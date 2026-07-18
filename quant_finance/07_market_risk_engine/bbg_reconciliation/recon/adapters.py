"""Load the frozen snapshots that get reconciled.

The in-house side is produced by capture_inhouse.py (which drives
engine_adapter.collect_all() — the class layer mirroring your base + product
engine). The pack never imports the engine into pytest; it reads the CSV
snapshot. The BBG side is your export, normalised onto the same canonical schema.
"""

from pathlib import Path

import pandas as pd

from . import schema

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_inhouse() -> pd.DataFrame:
    """Frozen in-house snapshot: data/inhouse/inhouse_canonical.csv."""
    path = DATA_DIR / "inhouse" / "inhouse_canonical.csv"
    if not path.exists():
        return schema.empty_canonical()
    return schema.validate_canonical(pd.read_csv(path), source="inhouse")


def load_bbg() -> pd.DataFrame:
    """Bloomberg export normalised to canonical: data/bbg/bbg_canonical.csv.

    TODO(you): if the raw BBG dump uses its own scenario names, remap them to the
    canonical scenario_id here before validating (they must match to join)."""
    path = DATA_DIR / "bbg" / "bbg_canonical.csv"
    if not path.exists():
        return schema.empty_canonical()
    return schema.validate_canonical(pd.read_csv(path), source="bbg")
