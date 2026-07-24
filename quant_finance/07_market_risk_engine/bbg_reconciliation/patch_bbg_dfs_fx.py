"""Patch the engine's DF npz with BBG discount factors at FX swap/forward
cashflow dates only (TRS/repo gilt-leg curves are handled separately).

    python patch_bbg_dfs_fx.py

Produces a NEW npz (never overwrites the original) with only the specific
month-index slots touched by FX swap/forward cashflows replaced by BBG's own
DF at that exact date. Feed the output into the real engine's base + stress
run (scoped to just the sample trades) to get BBG-consistent base and
stressed PVs through the engine's own mechanics.

TODO before running — confirm against the engine's actual source, or the
patch silently lands in the wrong slot:
  - date_to_month_index(): the exact bucketing formula
  - 0- vs 1-indexed array (month 1 -> arr[0] or arr[1]?)
  - npz key names (run: list(np.load(NPZ_PATH).keys()))
"""

import numpy as np
import pandas as pd

AS_OF = pd.Timestamp("2026-07-18")          # TODO: confirm engine's actual as_of date
NPZ_PATH = "data/inhouse/discount_factors.npz"
CASHFLOWS_PATH = "data/inhouse/cashflows.csv"   # cols: trade_id, instrument, ccy, cashflow_date, amount
BBG_DF_PATH = "data/bbg/bbg_dfs_fx.csv"         # cols: ccy, date, bbg_df
OUT_PATH = "data/inhouse/discount_factors_bbg_fx.npz"


def date_to_month_index(cf_date: pd.Timestamp, as_of: pd.Timestamp) -> int:
    """TODO: MUST match the engine's own bucketing exactly — confirm the
    formula and indexing before trusting this."""
    return (cf_date.year - as_of.year) * 12 + (cf_date.month - as_of.month)


def build_overrides(cashflows: pd.DataFrame, bbg_dfs: pd.DataFrame) -> dict:
    """{ccy: {month_index: bbg_df}} for FX swap/forward cashflows only, with
    collision detection (two cashflows landing in the same bucket with
    different BBG DFs)."""
    fx_cf = cashflows[cashflows["instrument"].isin(["FX_FORWARD", "FX_SWAP"])]
    overrides: dict = {}
    for _, row in fx_cf.iterrows():
        match = bbg_dfs[(bbg_dfs["ccy"] == row["ccy"]) & (bbg_dfs["date"] == row["cashflow_date"])]
        if match.empty:
            raise ValueError(f"No BBG DF for {row['ccy']} {row['cashflow_date'].date()}")
        idx = date_to_month_index(row["cashflow_date"], AS_OF)
        val = match["bbg_df"].iloc[0]
        existing = overrides.setdefault(row["ccy"], {})
        if idx in existing and not np.isclose(existing[idx], val):
            raise ValueError(
                f"Collision: {row['ccy']} month {idx} already has {existing[idx]}, new value {val}"
            )
        existing[idx] = val
    return overrides


def patch_npz(npz_path: str, overrides: dict, out_path: str) -> None:
    dfs = dict(np.load(npz_path))
    for ccy, month_map in overrides.items():
        if ccy not in dfs:
            raise KeyError(f"No curve for {ccy} — actual keys: {list(dfs.keys())}")
        arr = dfs[ccy].copy()
        for month_idx, bbg_val in month_map.items():
            arr[month_idx] = bbg_val          # TODO: confirm 0- vs 1-indexing against engine
        dfs[ccy] = arr
    np.savez(out_path, **dfs)


def main() -> None:
    cashflows = pd.read_csv(CASHFLOWS_PATH, parse_dates=["cashflow_date"])
    bbg_dfs = pd.read_csv(BBG_DF_PATH, parse_dates=["date"])
    overrides = build_overrides(cashflows, bbg_dfs)
    patch_npz(NPZ_PATH, overrides, OUT_PATH)
    n_slots = sum(len(v) for v in overrides.values())
    print(f"Patched {n_slots} slots across {list(overrides)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
