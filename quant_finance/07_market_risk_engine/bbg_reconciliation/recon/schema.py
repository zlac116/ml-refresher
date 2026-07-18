"""Canonical schema for the reconciliation pack.

Everything — in-house engine output AND Bloomberg export — is normalised into
ONE tidy long-format table before anything is compared. One row per
(trade_id, scenario_id, metric). This is the single contract the rest of the
pack depends on; adapters exist only to map each source onto it.

Grain
-----
    trade_id  | trade_type | ccy | notional | scenario_id | metric   | value
    IRS_001   | irs        | GBP | 50e6     | BASE        | base_pv  | 123456.7
    IRS_001   | irs        | GBP | 50e6     | PAR_UP_1BP  | stressed_pv | 118234.1
    IRS_001   | irs        | GBP | 50e6     | PAR_UP_1BP  | impact   | -5222.6

Conventions (make these explicit and identical on BOTH sides)
    - value is a PV / impact in `ccy` (reporting-ccy conversion, if any, is done
      upstream in the adapter, using the FX stated in scenarios.yaml).
    - PV sign: receiver / long = positive.
    - impact = stressed_pv - base_pv (signed).
    - metric BASE row: stressed_pv == base_pv, impact == 0.
"""

import pandas as pd

# --- column names (import these everywhere; never hard-code strings) ---------
TRADE_ID = "trade_id"
TRADE_TYPE = "trade_type"
CCY = "ccy"
NOTIONAL = "notional"
SCENARIO_ID = "scenario_id"
METRIC = "metric"
VALUE = "value"

KEY_COLS = [TRADE_ID, SCENARIO_ID, METRIC]
CANONICAL_COLS = [TRADE_ID, TRADE_TYPE, CCY, NOTIONAL, SCENARIO_ID, METRIC, VALUE]

# --- controlled vocabularies -------------------------------------------------
METRIC_BASE_PV = "base_pv"
METRIC_STRESSED_PV = "stressed_pv"
METRIC_IMPACT = "impact"
METRICS = (METRIC_BASE_PV, METRIC_STRESSED_PV, METRIC_IMPACT)

# The recognised trade types for this pack. Keep in sync with tolerances.yaml.
# MINIMAL scope: FX forwards & swaps only. Add types here as you extend.
TRADE_TYPES = (
    "fx_forward",
    "fx_swap",
)


def empty_canonical() -> pd.DataFrame:
    """Return an empty frame with the canonical columns and dtypes."""
    return pd.DataFrame(columns=CANONICAL_COLS)


def validate_canonical(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Fail loudly if a source has not been normalised correctly.

    Cheap contract check run at the boundary of every adapter — a mislabelled
    column here is the difference between a real break and a phantom one.
    """
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{source}] missing canonical columns: {missing}")

    bad_metric = set(df[METRIC].unique()) - set(METRICS)
    if bad_metric:
        raise ValueError(f"[{source}] unknown metric(s): {sorted(bad_metric)}")

    bad_type = set(df[TRADE_TYPE].unique()) - set(TRADE_TYPES)
    if bad_type:
        raise ValueError(f"[{source}] unknown trade_type(s): {sorted(bad_type)}")

    dupes = df.duplicated(subset=KEY_COLS)
    if dupes.any():
        raise ValueError(
            f"[{source}] {dupes.sum()} duplicate rows on {KEY_COLS}; "
            "each (trade, scenario, metric) must be unique."
        )
    return df[CANONICAL_COLS].copy()
