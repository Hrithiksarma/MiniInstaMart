import json

import numpy as np
import pandas as pd
import pytest

from simulator.orders import build_world, generate, inject_defects, simulate_orders
from utils.config import load_config

N = 30_000


@pytest.fixture(scope="module")
def sim_cfg():
    cfg = load_config()
    cfg["simulator"]["n_orders"] = N
    return cfg


@pytest.fixture(scope="module")
def clean(sim_cfg):
    sim = sim_cfg["simulator"]
    stores = build_world(sim, sim_cfg["seed"])
    orders, truth = simulate_orders(sim, stores, sim_cfg["seed"])
    return sim, orders, truth


def test_same_seed_same_data(sim_cfg):
    sim = sim_cfg["simulator"]
    stores = build_world(sim, 7)
    a, _ = simulate_orders(sim, stores, 7)
    b, _ = simulate_orders(sim, stores, 7)
    pd.testing.assert_frame_equal(a, b)


def test_components_add_up_to_delivery_time(clean):
    _, orders, truth = clean
    parts = truth[["prep_min", "rider_wait_min", "travel_min", "handover_min"]].sum(axis=1)
    assert np.allclose(parts, truth["total_min"])
    # delivery_time_min is the same total, only rounded to 2 decimals
    assert np.allclose(orders["delivery_time_min"], truth["total_min"], atol=0.01)


def test_clean_data_has_no_defects(clean):
    _, orders, _ = clean
    assert orders["weather"].notna().all()
    assert (orders["delivery_time_min"] > 0).all()
    assert not orders.duplicated().any()


def test_delivery_times_look_realistic(clean):
    _, orders, _ = clean
    assert 15 < orders["delivery_time_min"].mean() < 40


def test_rain_and_peak_make_deliveries_slower(clean):
    _, _, truth = clean
    t = truth
    assert t.loc[t.is_rain, "total_min"].mean() > t.loc[~t.is_rain, "total_min"].mean()
    assert t.loc[t.is_peak, "total_min"].mean() > t.loc[~t.is_peak, "total_min"].mean()


def test_rain_peak_interaction_is_real(clean):
    """Rain at peak should hurt more than rain and peak effects added separately."""
    _, _, truth = clean
    t = truth
    base = t.loc[~t.is_rain & ~t.is_peak, "rider_wait_min"].mean()
    rain_only = t.loc[t.is_rain & ~t.is_peak, "rider_wait_min"].mean() - base
    peak_only = t.loc[~t.is_rain & t.is_peak, "rider_wait_min"].mean() - base
    both = t.loc[t.is_rain & t.is_peak, "rider_wait_min"].mean() - base
    assert both > rain_only + peak_only


def test_injected_defect_rates_match_config(clean, sim_cfg):
    sim, orders, _ = clean
    dirty, manifest = inject_defects(orders, sim["defects"], seed=1)
    rates = sim["defects"]
    # duplicates are copies of damaged rows too, so count defects on the original rows only
    first = dirty.drop_duplicates()

    assert first["weather"].isna().sum() == manifest["missing_weather"]
    assert manifest["missing_weather"] / N == pytest.approx(rates["missing_weather_rate"], abs=1e-4)

    assert (first["delivery_time_min"] < 0).sum() == manifest["negative_time"]
    assert manifest["negative_time"] / N == pytest.approx(rates["negative_time_rate"], abs=1e-4)

    assert manifest["impossible_distance"] / N == pytest.approx(
        rates["impossible_distance_rate"], abs=1e-4
    )
    assert len(dirty) - N == manifest["duplicate_rows"]
    assert dirty.duplicated().sum() == manifest["duplicate_rows"]  # exact copies


def test_leakage_column_reveals_the_target(clean):
    """If delivered_at were allowed as a feature, the model would get the answer for free."""
    _, orders, _ = clean
    implied = (orders["delivered_at"] - orders["placed_at"]).dt.total_seconds() / 60
    assert np.allclose(implied, orders["delivery_time_min"], atol=0.02)


def test_generate_writes_locked_raw_files(sim_cfg, tmp_path):
    manifest = generate(sim_cfg, out_dir=tmp_path)
    assert (tmp_path / "orders.parquet").exists()
    assert (tmp_path / "orders_truth.parquet").exists()
    assert (tmp_path / "stores.parquet").exists()
    saved = json.loads((tmp_path / "defects_manifest.json").read_text())
    assert saved["missing_weather"] == manifest["missing_weather"]
    # running again must still work even though the first run locked the files
    generate(sim_cfg, out_dir=tmp_path)


def test_drift_regime_is_visibly_different(sim_cfg, tmp_path):
    cfg = load_config()
    cfg["simulator"]["n_orders"] = 100_000
    cfg["simulator"]["drift"]["n_orders"] = 20_000
    cfg["simulator"]["drift"]["n_days"] = 18
    cfg["simulator"]["n_days"] = 90
    generate(cfg, out_dir=tmp_path)
    generate(cfg, out_dir=tmp_path, drift=True)
    normal = pd.read_parquet(tmp_path / "orders.parquet")
    drift = pd.read_parquet(tmp_path / "orders_drift.parquet")
    assert drift["delivery_time_min"].median() > normal["delivery_time_min"].median()
    assert drift["placed_at"].min() > normal["placed_at"].max()
