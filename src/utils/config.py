from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read config.yaml from the repo root (or from an explicit path)."""
    cfg_path = path or ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


def data_path(kind: str) -> Path:
    """Resolve one of the data folders (raw / interim / processed) as an absolute path."""
    return ROOT / load_config()["paths"][kind]
