"""Synthetic quick-commerce order simulator.

The idea: we decide the rules of the world ourselves (prep time, rider wait, travel,
handover), generate orders from those rules, then damage the file on purpose. Later the
ETA model has to rediscover the rules, and cleaning has to find the damage.
"""

import argparse
import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.config import data_path, load_config

CITIES = [
    ("Bengaluru", 12.97, 77.59),
    ("Mumbai", 19.07, 72.87),
    ("Delhi", 28.61, 77.20),
    ("Hyderabad", 17.38, 78.48),
]

# Relative number of orders per hour of day (night is quiet, lunch and dinner spike).
HOUR_DEMAND = np.array(
    [0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 0.8, 1.0, 0.9, 0.9, 1.1]
    + [1.8, 1.7, 1.1, 0.9, 1.0, 1.3, 1.6, 2.2, 2.4, 2.0, 1.2, 0.6]
)

# Road congestion by hour, 0 = empty roads, 1 = gridlock.
HOUR_TRAFFIC = np.array(
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.25, 0.50, 0.80, 0.70, 0.50, 0.50]
    + [0.55, 0.55, 0.50, 0.50, 0.60, 0.80, 0.90, 0.75, 0.60, 0.50, 0.30, 0.15]
)

KM_PER_DEG_LAT = 111.0


def build_world(sim: dict[str, Any], seed: int) -> pd.DataFrame:
    """Create the stores. Same seed gives the same stores, so the drift run reuses them."""
    rng = np.random.default_rng(seed)
    eff = sim["effects"]["prep"]
    rows = []
    zone_id = 0
    for city_idx in range(sim["n_cities"]):
        name, lat0, lon0 = CITIES[city_idx]
        for _ in range(sim["zones_per_city"]):
            zlat = lat0 + rng.uniform(-0.06, 0.06)
            zlon = lon0 + rng.uniform(-0.06, 0.06)
            for _ in range(sim["stores_per_zone"]):
                rows.append(
                    {
                        "city_idx": city_idx,
                        "city": name,
                        "zone_id": zone_id,
                        "store_lat": zlat + rng.uniform(-0.02, 0.02),
                        "store_lon": zlon + rng.uniform(-0.02, 0.02),
                        # some stores are just slower than others
                        "store_base_prep_min": eff["store_base_mean_min"]
                        * rng.lognormal(0, eff["store_base_sd_log"]),
                        # some stores are much busier than others
                        "popularity": rng.lognormal(0, 0.8),
                    }
                )
            zone_id += 1
    stores = pd.DataFrame(rows)
    stores.insert(0, "store_id", np.arange(len(stores)))
    return stores


def _rain_table(sim: dict[str, Any], rng: np.random.Generator, rain_day_prob: float) -> np.ndarray:
    """rain[city, day, hour] = True when it is raining. Rain comes as a block of hours."""
    n_days = sim["n_days"]
    rain = np.zeros((sim["n_cities"], n_days, 24), dtype=bool)
    for c in range(sim["n_cities"]):
        for d in range(n_days):
            if rng.random() < rain_day_prob:
                start = rng.integers(0, 24)
                length = rng.integers(sim["rain"]["min_hours"], sim["rain"]["max_hours"] + 1)
                rain[c, d, start : start + length] = True
    return rain


def _count_prior_orders(store_ids: np.ndarray, t_sec: np.ndarray, window_sec: int) -> np.ndarray:
    """For each order: how many orders hit the same store in the window just before it.

    Only looks backwards in time, so it is a legitimate feature at order time.
    """
    big = 10**9
    key = store_ids.astype(np.int64) * big + t_sec.astype(np.int64)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    left = np.searchsorted(sorted_key, sorted_key - window_sec, side="left")
    count_sorted = np.arange(len(sorted_key)) - left
    out = np.empty(len(key), dtype=np.int64)
    out[order] = count_sorted
    return out


def simulate_orders(
    sim: dict[str, Any],
    stores: pd.DataFrame,
    seed: int,
    day_offset: int = 0,
    rain_day_prob: float | None = None,
    riders_scale: float = 1.0,
    speed_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate clean orders plus a 'truth' table holding each delivery-time component."""
    rng = np.random.default_rng(seed)
    eff = sim["effects"]
    n = sim["n_orders"]
    n_days = sim["n_days"]
    peak_hours = set(sim["peak_hours"])
    if rain_day_prob is None:
        rain_day_prob = sim["rain"]["rain_day_prob"]

    # when orders arrive
    day_w = np.array(
        [sim["weekend_demand_multiplier"] if d % 7 >= 5 else 1.0 for d in range(n_days)]
    )
    day = rng.choice(n_days, size=n, p=day_w / day_w.sum())
    hour = rng.choice(24, size=n, p=HOUR_DEMAND / HOUR_DEMAND.sum())
    t_sec = day * 86400 + hour * 3600 + rng.integers(0, 3600, size=n)
    order_idx = np.argsort(t_sec, kind="stable")
    day, hour, t_sec = day[order_idx], hour[order_idx], t_sec[order_idx]

    # where they arrive
    pop = stores["popularity"].to_numpy()
    store = rng.choice(len(stores), size=n, p=pop / pop.sum())
    zone = stores["zone_id"].to_numpy()[store]
    city_idx = stores["city_idx"].to_numpy()[store]
    is_peak = np.isin(hour, list(peak_hours))

    # conditions at order time
    rain_tbl = _rain_table(sim, rng, rain_day_prob)
    is_rain = rain_tbl[city_idx, day, hour]
    traffic = np.clip(HOUR_TRAFFIC[hour] + rng.normal(0, 0.05, n), 0, 1)
    items = np.clip(1 + rng.poisson(3.5, n), 1, 25)

    # riders on shift per zone and hour
    n_zones = sim["n_cities"] * sim["zones_per_city"]
    r = eff["rider"]
    hours_axis = np.arange(n_days * 24) % 24
    boost = np.where(np.isin(hours_axis, list(peak_hours)), r["peak_rider_boost"], 1.0)
    zone_scale = np.exp(rng.normal(0, 0.2, n_zones))
    lam = r["riders_per_zone_mean"] * riders_scale * zone_scale[:, None] * boost[None, :]
    riders_tbl = np.maximum(rng.poisson(lam), 1)
    slot = day * 24 + hour
    riders = riders_tbl[zone, slot]

    # how busy was the zone in this hour, and the store in the last 30 minutes
    zone_slot = zone.astype(np.int64) * (n_days * 24) + slot
    zone_orders = np.bincount(zone_slot)[zone_slot]
    pressure = zone_orders / riders
    queued = _count_prior_orders(store, t_sec, 1800)

    # --- the four parts of a delivery ---
    p = eff["prep"]
    sigma = p["noise_sigma_log"]
    prep_noise = np.exp(rng.normal(-(sigma**2) / 2, sigma, n))  # mean 1
    prep = (
        (
            stores["store_base_prep_min"].to_numpy()[store]
            + p["per_item_min"] * items
            + p["per_queued_order_min"] * queued
        )
        * np.where(is_peak, p["peak_multiplier"], 1.0)
        * prep_noise
    )

    wait_mean = (
        r["base_wait_min"]
        + r["wait_per_pressure_min"] * pressure
        + r["rain_add_min"] * is_rain
        + r["rain_peak_interaction_min"] * (is_rain & is_peak)
    )
    rider_wait = rng.gamma(4.0, wait_mean / 4.0)

    d = eff["distance"]
    straight_km = np.clip(
        rng.gamma(d["gamma_shape"], d["gamma_scale_km"], n), d["min_km"], d["max_km"]
    )
    t = eff["travel"]
    road_km = straight_km * t["road_circuity"]
    speed = (
        t["speed_kmph"]
        * np.where(is_rain, t["rain_speed_multiplier"], 1.0)
        * (1 - t["traffic_slowdown"] * traffic)
        * speed_multiplier
    )
    ts = t["noise_sigma_log"]
    travel = road_km / speed * 60 * np.exp(rng.normal(-(ts**2) / 2, ts, n))

    h = eff["handover"]
    handover = rng.gamma(h["gamma_shape"], h["gamma_scale_min"], n)

    total = prep + rider_wait + travel + handover

    # turn the straight-line distance into a customer location
    bearing = rng.uniform(0, 2 * np.pi, n)
    store_lat = stores["store_lat"].to_numpy()[store]
    store_lon = stores["store_lon"].to_numpy()[store]
    cust_lat = store_lat + straight_km * np.cos(bearing) / KM_PER_DEG_LAT
    cust_lon = store_lon + straight_km * np.sin(bearing) / (
        KM_PER_DEG_LAT * np.cos(np.radians(store_lat))
    )

    start = pd.Timestamp(sim["start_date"]) + pd.Timedelta(days=day_offset)
    placed_at = start + pd.to_timedelta(t_sec, unit="s")

    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, n + 1) + day_offset * 10_000_000,
            "placed_at": placed_at,
            "city": stores["city"].to_numpy()[store],
            "zone_id": zone,
            "store_id": store,
            "store_lat": store_lat,
            "store_lon": store_lon,
            "cust_lat": cust_lat,
            "cust_lon": cust_lon,
            "item_count": items,
            "weather": np.where(is_rain, "rain", "clear"),
            "traffic_index": traffic.round(3),
            "riders_on_shift": riders,
            "delivery_time_min": total.round(2),
            # LEAKAGE TRAP: only known once the order is delivered. Never a legal feature.
            "delivered_at": placed_at + pd.to_timedelta(total, unit="min"),
        }
    )
    truth = pd.DataFrame(
        {
            "order_id": orders["order_id"],
            "prep_min": prep,
            "rider_wait_min": rider_wait,
            "travel_min": travel,
            "handover_min": handover,
            "total_min": total,
            "straight_km": straight_km,
            "store_queue_30m": queued,
            "zone_pressure": pressure,
            "is_rain": is_rain,
            "is_peak": is_peak,
        }
    )
    return orders, truth


def inject_defects(
    orders: pd.DataFrame, defects: dict[str, float], seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Damage the clean orders. Each defect hits its own rows, so the counts are easy to audit."""
    rng = np.random.default_rng(seed)
    df = orders.copy()
    n = len(df)
    perm = rng.permutation(n)
    pos = 0

    def take(rate: float) -> np.ndarray:
        nonlocal pos
        k = int(round(rate * n))
        rows = perm[pos : pos + k]
        pos += k
        return rows

    miss = take(defects["missing_weather_rate"])
    df.loc[df.index[miss], "weather"] = np.nan

    bad_dist = take(defects["impossible_distance_rate"])
    half = len(bad_dist) // 2
    df.loc[df.index[bad_dist[:half]], ["cust_lat", "cust_lon"]] = 0.0  # "null island"
    swap = df.index[bad_dist[half:]]
    df.loc[swap, ["cust_lat", "cust_lon"]] = df.loc[swap, ["cust_lon", "cust_lat"]].to_numpy()

    neg = take(defects["negative_time_rate"])
    df.loc[df.index[neg], "delivery_time_min"] *= -1

    n_dup = int(round(defects["duplicate_rate"] * n))
    dup_rows = rng.choice(n, size=n_dup, replace=False)
    df = pd.concat([df, df.iloc[dup_rows]], ignore_index=True)
    df = df.sort_values(["placed_at", "order_id"], kind="stable").reset_index(drop=True)

    manifest = {
        "clean_orders": n,
        "rows_after_duplicates": len(df),
        "missing_weather": len(miss),
        "impossible_distance": len(bad_dist),
        "negative_time": len(neg),
        "duplicate_rows": n_dup,
        "leakage_column": "delivered_at",
    }
    return df, manifest


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a raw file and lock it, so nothing edits it by accident (D1)."""
    if path.exists():
        os.chmod(path, stat.S_IWRITE)
    df.to_parquet(path, index=False)
    os.chmod(path, stat.S_IREAD)


def generate(
    cfg: dict[str, Any], out_dir: Path | None = None, drift: bool = False
) -> dict[str, Any]:
    """Build the raw files. drift=True writes the shifted regime used later for drift alerts."""
    sim = copy.deepcopy(cfg["simulator"])
    seed = cfg["seed"]
    out = out_dir or data_path("raw")
    out.mkdir(parents=True, exist_ok=True)

    stores = build_world(sim, seed)
    if drift:
        d = sim["drift"]
        # start on a Monday after the normal period so the weekend pattern lines up
        day_offset = -(-sim["n_days"] // 7) * 7
        sim["n_orders"] = d["n_orders"]
        sim["n_days"] = d["n_days"]
        orders, truth = simulate_orders(
            sim,
            stores,
            seed + 1,
            day_offset=day_offset,
            rain_day_prob=d["rain_day_prob"],
            riders_scale=d["riders_scale"],
            speed_multiplier=d["speed_multiplier"],
        )
        tag = "_drift"
    else:
        orders, truth = simulate_orders(sim, stores, seed)
        tag = ""

    damaged, manifest = inject_defects(orders, sim["defects"], seed + 2)
    _write_parquet(damaged, out / f"orders{tag}.parquet")
    _write_parquet(truth, out / f"orders_truth{tag}.parquet")
    if not drift:
        _write_parquet(stores, out / "stores.parquet")
    manifest["regime"] = "drift" if drift else "normal"
    manifest_path = out / f"defects_manifest{tag}.json"
    if manifest_path.exists():
        os.chmod(manifest_path, stat.S_IWRITE)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated raw order data.")
    parser.add_argument("--n-orders", type=int, help="override config for a quick run")
    args = parser.parse_args()
    cfg = load_config()
    if args.n_orders:
        cfg["simulator"]["n_orders"] = args.n_orders
    print(json.dumps(generate(cfg), indent=2))
    print(json.dumps(generate(cfg, drift=True), indent=2))


if __name__ == "__main__":
    main()
