"""Load the YAML config (scenarios + tolerances).

Kept trivial on purpose — the interesting content lives in the YAML so a
reviewer/controller can read and challenge thresholds without reading Python.
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_scenarios() -> dict:
    """scenario_id -> {desc, inhouse_shock, bbg_shock, convention_note, ...}."""
    return load_yaml("scenarios.yaml")


def load_tolerances() -> dict:
    """(trade_type, metric) tolerance rules; see tolerances.yaml for schema."""
    return load_yaml("tolerances.yaml")
