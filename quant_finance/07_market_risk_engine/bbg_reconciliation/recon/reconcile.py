"""Join in-house vs BBG, difference, and classify breaks.

This is generic and source-agnostic: it only ever sees the canonical schema.
Output is a wide recon frame, one row per (trade_id, scenario_id, metric):

    inhouse | bbg | diff_abs | diff_rel | diff_bp | tol_basis | threshold | status

Status ∈ {PASS, WARN, FAIL, N/A}.  N/A = one side missing (a coverage break,
not a value break — flagged separately so it's never silently a pass).

Scope: the recon is bounded to whatever trade_ids appear in the BBG input. An
in-house trade never sent to BBG for this sample is out of scope and excluded
entirely — it is NOT a coverage break. A trade_id BBG has but in-house doesn't
still surfaces as a genuine N/A coverage break.
"""

import numpy as np
import pandas as pd

from . import schema

PASS, WARN, FAIL, NA = "PASS", "WARN", "FAIL", "N/A"


def merge_sources(inhouse: pd.DataFrame, bbg: pd.DataFrame) -> pd.DataFrame:
    """Outer-join the two canonical frames on (trade, scenario, metric),
    scoped to trade_ids present in `bbg`.

    In-house trades outside the BBG sample are dropped before the join so they
    never appear as spurious N/A rows. A BBG trade_id with no in-house match at
    all still comes through as N/A — a real coverage break.
    """
    bbg_trade_ids = bbg[schema.TRADE_ID].unique()
    scoped_inhouse = inhouse[inhouse[schema.TRADE_ID].isin(bbg_trade_ids)]

    left = scoped_inhouse.rename(columns={schema.VALUE: "inhouse"})
    right = bbg.rename(columns={schema.VALUE: "bbg"})[schema.KEY_COLS + ["bbg"]]
    merged = left.merge(right, on=schema.KEY_COLS, how="outer")
    # Recover meta (trade_type/ccy/notional) for BBG-only rows from either side.
    # (Both sides should carry identical meta; in-house wins where present.)
    return merged


def _difference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["diff_abs"] = df["inhouse"] - df["bbg"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["diff_rel"] = df["diff_abs"] / df["bbg"].abs()
        df["diff_bp"] = 1e4 * df["diff_abs"] / df[schema.NOTIONAL].abs()
    return df


def _select_tol(trade_type: str, metric: str, tolerances: dict) -> dict:
    """Resolve the tolerance rule for a (trade_type, metric).

    Lookup order: exact (trade_type.metric) -> trade_type.default -> global
    default. A rule is a dict: {basis, warn, fail, abs_floor}.
      basis     : 'bp_notional' | 'abs' | 'rel'   (which diff column to test)
      warn/fail : thresholds on that basis (bp, ccy, or fraction)
      abs_floor : ignore breaks smaller than this in ccy (kills divide-by-tiny)
    """
    tt = tolerances.get(trade_type, {})
    rule = tt.get(metric) or tt.get("default") or tolerances.get("default")
    if rule is None:
        raise KeyError(f"no tolerance for ({trade_type}, {metric}) and no default")
    return rule


def _basis_value(row: pd.Series, basis: str) -> float:
    return {
        "bp_notional": abs(row["diff_bp"]),
        "abs": abs(row["diff_abs"]),
        "rel": abs(row["diff_rel"]),
    }[basis]


def classify(merged: pd.DataFrame, tolerances: dict) -> pd.DataFrame:
    """Add tol_basis / threshold / status columns."""
    df = _difference(merged)
    basis_col, thr_col, status_col = [], [], []
    for _, row in df.iterrows():
        # Missing on either side -> coverage break, not a value comparison.
        if pd.isna(row["inhouse"]) or pd.isna(row["bbg"]):
            basis_col.append("coverage"); thr_col.append(np.nan); status_col.append(NA)
            continue
        rule = _select_tol(row[schema.TRADE_TYPE], row[schema.METRIC], tolerances)
        basis = rule["basis"]
        metric_val = _basis_value(row, basis)
        # Absolute-materiality floor: tiny ccy diffs never fail regardless of bp.
        if abs(row["diff_abs"]) <= rule.get("abs_floor", 0.0):
            status = PASS
        elif metric_val > rule["fail"]:
            status = FAIL
        elif metric_val > rule["warn"]:
            status = WARN
        else:
            status = PASS
        basis_col.append(basis); thr_col.append(rule["fail"]); status_col.append(status)
    df["tol_basis"] = basis_col
    df["threshold"] = thr_col
    df["status"] = status_col
    return df


def run(inhouse: pd.DataFrame, bbg: pd.DataFrame, tolerances: dict) -> pd.DataFrame:
    """End-to-end: merge -> difference -> classify. The pack's core call."""
    return classify(merge_sources(inhouse, bbg), tolerances)


def summary(recon: pd.DataFrame) -> pd.DataFrame:
    """Pass-rate matrix by trade_type x metric — the top of the pack."""
    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        return pd.Series({
            "n": n,
            "pass": (g["status"] == PASS).sum(),
            "warn": (g["status"] == WARN).sum(),
            "fail": (g["status"] == FAIL).sum(),
            "na": (g["status"] == NA).sum(),
            "max_abs_bp": g["diff_bp"].abs().max(),
        })
    return (
        recon.groupby([schema.TRADE_TYPE, schema.METRIC], dropna=False)
        .apply(agg, include_groups=False)
        .reset_index()
    )
