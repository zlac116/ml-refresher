"""Recon is scoped to trade_ids present in the BBG input.

An in-house trade never sent to BBG must be excluded entirely (not surfaced as
N/A) — it's out of sample scope, not a coverage break. A BBG trade_id with no
in-house match at all must still surface as a genuine N/A coverage break.
"""

import pandas as pd

from recon import reconcile, schema


def _row(trade_id, trade_type, scenario_id, metric, value, notional=1_000_000, ccy="GBP"):
    return {
        schema.TRADE_ID: trade_id, schema.TRADE_TYPE: trade_type, schema.CCY: ccy,
        schema.NOTIONAL: notional, schema.SCENARIO_ID: scenario_id,
        schema.METRIC: metric, schema.VALUE: value,
    }


def test_inhouse_only_trade_is_excluded_not_na():
    inhouse = pd.DataFrame([
        _row("IN_SCOPE", "fx_forward", "BASE", schema.METRIC_BASE_PV, 100.0),
        _row("NOT_SENT_TO_BBG", "fx_forward", "BASE", schema.METRIC_BASE_PV, 200.0),
    ])
    bbg = pd.DataFrame([
        _row("IN_SCOPE", "fx_forward", "BASE", schema.METRIC_BASE_PV, 101.0),
    ])
    merged = reconcile.merge_sources(inhouse, bbg)
    assert set(merged[schema.TRADE_ID]) == {"IN_SCOPE"}


def test_bbg_only_trade_still_surfaces_as_na():
    inhouse = pd.DataFrame([
        _row("IN_SCOPE", "fx_forward", "BASE", schema.METRIC_BASE_PV, 100.0),
    ])
    bbg = pd.DataFrame([
        _row("IN_SCOPE", "fx_forward", "BASE", schema.METRIC_BASE_PV, 101.0),
        _row("MISSING_FROM_INHOUSE", "fx_forward", "BASE", schema.METRIC_BASE_PV, 50.0),
    ])
    tolerances = {"default": {"basis": "bp_notional", "warn": 1.0, "fail": 5.0, "abs_floor": 0.0}}
    recon = reconcile.run(inhouse, bbg, tolerances)
    missing = recon[recon[schema.TRADE_ID] == "MISSING_FROM_INHOUSE"]
    assert len(missing) == 1
    assert missing.iloc[0]["status"] == reconcile.NA
