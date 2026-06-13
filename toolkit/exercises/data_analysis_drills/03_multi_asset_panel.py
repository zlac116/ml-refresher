"""
DRILL 3 — Multi-Asset Panel
===========================

OBJECTIVE
    Convert a long-format ticker/date/close DataFrame into a wide panel
    indexed by date with one column per ticker, then compute per-ticker
    annualised returns and ann vol.

ESTIMATED TIME
    20 min

TOPICS
    pandas.DataFrame.pivot, .melt
    pandas axis-aware aggregations (.mean(axis=0))
    pct_change on a wide frame

EXPECTED OUTPUT
    long rows:           400
    wide shape:          (100, 4)
    AAA final:           91.85
    DDD final:           118.70
    AAA ann vol:         21.69 %
    mean ann return AAA: -21.39 %
    mean ann return DDD: 48.99 %

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
tickers = ["AAA", "BBB", "CCC", "DDD"]
n_days = 100
dates = pd.bdate_range("2024-01-01", periods=n_days)

_records = []
for t in tickers:
    drift = 0.0008 if t == "AAA" else 0.0002
    r = np.random.normal(drift, 0.015, n_days)
    px = 100.0 * (1 + r).cumprod()
    for d, p in zip(dates, px):
        _records.append({"date": d, "ticker": t, "close": p})
long_df = pd.DataFrame(_records)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide: index=date, columns=ticker, values=close. Sorted index."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def per_ticker_ann_vol(wide: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Annualised vol of daily returns for each column.

    Returns a Series indexed by ticker.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def per_ticker_mean_ann_return(wide: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Mean daily simple return * periods_per_year, per ticker."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assert len(long_df) == 400

    wide = to_wide(long_df)
    assert wide.shape == (100, 4), wide.shape
    assert list(wide.columns) == ["AAA", "BBB", "CCC", "DDD"]
    assert abs(wide["AAA"].iloc[-1] -  91.8498) < 1e-3
    assert abs(wide["DDD"].iloc[-1] - 118.7009) < 1e-3
    assert wide.index.is_monotonic_increasing

    vols = per_ticker_ann_vol(wide)
    assert abs(vols["AAA"] - 0.2169) < 1e-3, f"AAA vol off: {vols['AAA']}"

    mean_ret = per_ticker_mean_ann_return(wide)
    assert abs(mean_ret["AAA"] - (-0.213870)) < 1e-3
    assert abs(mean_ret["DDD"] - ( 0.489988)) < 1e-3

    print(f"long rows:           {len(long_df)}")
    print(f"wide shape:          {wide.shape}")
    print(f"AAA final:           {wide['AAA'].iloc[-1]:.2f}")
    print(f"DDD final:           {wide['DDD'].iloc[-1]:.2f}")
    print(f"AAA ann vol:         {vols['AAA'] * 100:.2f} %")
    print(f"mean ann return AAA: {mean_ret['AAA'] * 100:.2f} %")
    print(f"mean ann return DDD: {mean_ret['DDD'] * 100:.2f} %")
    print("\n✓ All checks passed.")
