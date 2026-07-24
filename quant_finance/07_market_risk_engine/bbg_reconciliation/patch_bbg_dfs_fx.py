"""Patch the engine's DF npz with BBG discount factors, scoped to the exact
"Deal No"s present in the BBG data file (TRS/repo gilt-leg curves are handled
separately — this script covers FX only, via whichever deals are in the BBG
file).

    python patch_bbg_dfs_fx.py

Produces a NEW npz (never overwrites the original) with only the specific
month-index slots touched by those deals' cashflows replaced by BBG's own DF.
Feed the output into the real engine's base + stress run (scoped to just the
sample trades) to get BBG-consistent base and stressed PVs through the
engine's own mechanics.

Array indexing is confirmed 0-based: month 1 -> arr[0].

TODO before running — confirm against the real files, or the patch silently
lands in the wrong slot:
  - date_to_month_index(): the calendar-month bucketing rule itself (day-based
    vs calendar-month diff, rounding vs floor) — indexing base is settled
  - BBG_DATE_COL / BBG_DF_COL: exact column names in the BBG data file
  - CCY_TO_CURVE: extend for any currency beyond GBP/USD/EUR present in the
    sample trades
"""

import numpy as np
import pandas as pd

AS_OF = pd.Timestamp("2026-07-18")          # TODO: confirm engine's actual as_of date
NPZ_PATH = "data/inhouse/discount_factors.npz"
CASHFLOWS_PATH = "data/inhouse/cashflows.csv"
BBG_DF_PATH = "data/bbg/bbg_dfs_fx.csv"
OUT_PATH = "data/inhouse/discount_factors_bbg_fx.npz"

# Cashflow file columns — confirmed
CF_DEAL_COL = "Deal No"
CF_DATE_COL = "Cash Flow Date"
CF_CCY_COL = "CCY"                          # TODO: confirm exact column name in cashflow file

# BBG data file columns — Deal No and CCY confirmed; date/DF columns TODO confirm
BBG_DEAL_COL = "Deal No"
BBG_CCY_COL = "CCY"                         # curve currency
BBG_DATE_COL = "Cash Flow Date"             # TODO: confirm matches cashflow file's date col
BBG_DF_COL = "BBG DF"                       # TODO: confirm exact column name

# CCY -> npz curve key. GBP is named after SONIA, not "base_CCY_GBP".
CCY_TO_CURVE = {
    "GBP": "base_SONIA",
    "USD": "base_CCY_USD",
    "EUR": "base_CCY_EUR",
}


def date_to_month_index(cf_date: pd.Timestamp, as_of: pd.Timestamp) -> int:
    """0-based array index: month 1 (first bucket after as_of) -> arr[0].
    TODO: the calendar-month-diff bucketing rule itself still needs
    confirming against the engine's own source (day-based vs calendar-month,
    rounding vs floor)."""
    months_elapsed = (cf_date.year - as_of.year) * 12 + (cf_date.month - as_of.month)
    month_number = max(1, months_elapsed)   # month 1 = first bucket, no month 0
    return month_number - 1                 # 0-based index


def build_overrides(cashflows: pd.DataFrame, bbg_dfs: pd.DataFrame) -> dict:
    """{curve_name: {month_index: bbg_df}}, scoped to Deal Nos present in the
    BBG file. Joins on (Deal No, Cash Flow Date, CCY) — date alone is not a
    unique key, since an FX swap's GBP and EUR legs commonly settle on the
    same date for the same deal. Raises on a collision (two cashflows landing
    in the same bucket with different BBG DFs) and warns on any BBG deal that
    never matched a cashflow row (a silent-gap risk otherwise)."""
    bbg_deal_nos = set(bbg_dfs[BBG_DEAL_COL].unique())
    cf = cashflows[cashflows[CF_DEAL_COL].isin(bbg_deal_nos)]

    overrides: dict = {}
    matched_deal_nos = set()
    for _, row in cf.iterrows():
        match = bbg_dfs[
            (bbg_dfs[BBG_DEAL_COL] == row[CF_DEAL_COL])
            & (bbg_dfs[BBG_DATE_COL] == row[CF_DATE_COL])
            & (bbg_dfs[BBG_CCY_COL] == row[CF_CCY_COL])
        ]
        if match.empty:
            raise ValueError(
                f"No BBG DF for Deal No {row[CF_DEAL_COL]} / {row[CF_CCY_COL]} "
                f"on {row[CF_DATE_COL].date()}"
            )
        matched_deal_nos.add(row[CF_DEAL_COL])

        ccy = row[CF_CCY_COL]
        if ccy not in CCY_TO_CURVE:
            raise KeyError(f"No curve mapping for CCY {ccy!r} — extend CCY_TO_CURVE")
        curve = CCY_TO_CURVE[ccy]

        idx = date_to_month_index(row[CF_DATE_COL], AS_OF)
        val = match[BBG_DF_COL].iloc[0]
        existing = overrides.setdefault(curve, {})
        if idx in existing and not np.isclose(existing[idx], val):
            raise ValueError(f"Collision: {curve} month {idx} already {existing[idx]}, new {val}")
        existing[idx] = val

    unmatched = bbg_deal_nos - matched_deal_nos
    if unmatched:
        print(f"WARNING: {len(unmatched)} BBG Deal No(s) never matched a cashflow row: {unmatched}")

    return overrides


def patch_npz(npz_path: str, overrides: dict, out_path: str) -> None:
    dfs = dict(np.load(npz_path))
    for curve, month_map in overrides.items():
        if curve not in dfs:
            raise KeyError(f"No curve array named {curve!r} in npz — actual keys: {list(dfs.keys())}")
        arr = dfs[curve].copy()
        for month_idx, bbg_val in month_map.items():
            arr[month_idx] = bbg_val        # 0-based index, from date_to_month_index()
        dfs[curve] = arr
    np.savez(out_path, **dfs)


def main() -> None:
    cashflows = pd.read_csv(CASHFLOWS_PATH, parse_dates=[CF_DATE_COL])
    bbg_dfs = pd.read_csv(BBG_DF_PATH, parse_dates=[BBG_DATE_COL])
    overrides = build_overrides(cashflows, bbg_dfs)
    patch_npz(NPZ_PATH, overrides, OUT_PATH)
    n_slots = sum(len(v) for v in overrides.values())
    print(f"Patched {n_slots} slots across {list(overrides)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
