"""
DRILL 9 — Complex GroupBy (per-group rolling, per-date z-score, ranks)
======================================================================

OBJECTIVE
    On a tidy long-format (date, ticker, sector, ret) frame:
      1. Per-ticker 20-day rolling annualised vol (groupby + .transform + rolling).
      2. Per-(date, sector) cross-sectional z-score of return.
      3. Per-ticker time-series rank of returns.

ESTIMATED TIME
    20 min

TOPICS
    groupby(...).transform(lambda s: s.rolling(...))
    groupby([date, sector]).transform — multi-key grouping
    .rank() — default ordinal; pct=True for percentile

EXPECTED OUTPUT
    rvol_20 notna count:    1155
    AAA last rvol:          0.234566
    EEE last rvol:          0.271632
    sector_z mean (~0):     0.00e+00
    sector_z notna:         1250
    AAA max rank:           250

GRADING
    All asserts must pass. No explicit Python loops over tickers/dates.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
_tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
_n = 250
_dates = pd.bdate_range("2024-01-01", periods=_n)
_sectors = {"AAA": "Tech", "BBB": "Tech", "CCC": "Fin", "DDD": "Fin", "EEE": "Energy"}

_records = []
for t in _tickers:
    rr = np.random.normal(0.0005, 0.015, _n)
    for d, r in zip(_dates, rr):
        _records.append({"date": d, "ticker": t, "sector": _sectors[t], "ret": r})
df = pd.DataFrame(_records).sort_values(["ticker", "date"]).reset_index(drop=True)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def per_ticker_rolling_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Per-ticker 20-day rolling annualised vol of df['ret'].

    Aligned with df's index. First (window-1) rows per ticker are NaN.
    Hint: df.groupby('ticker')['ret'].transform(lambda s: ...)
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def per_date_sector_zscore(df: pd.DataFrame) -> pd.Series:
    """For each (date, sector) cell, return the cross-sectional z-score of 'ret'
    (subtract group mean, divide by group std).

    Use groupby on TWO keys + transform.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def per_ticker_rank(df: pd.DataFrame) -> pd.Series:
    """Per-ticker rank of df['ret'] (1 = smallest). Use groupby + .rank()."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rvol = per_ticker_rolling_vol(df)
    assert rvol.notna().sum() == 1155, rvol.notna().sum()
    aaa_last = rvol[df["ticker"] == "AAA"].iloc[-1]
    eee_last = rvol[df["ticker"] == "EEE"].iloc[-1]
    assert abs(aaa_last - 0.234566) < 1e-4
    assert abs(eee_last - 0.271632) < 1e-4

    z = per_date_sector_zscore(df)
    assert len(z) == len(df)
    # Within-group mean must be ~0 everywhere with enough members
    assert abs(z.mean()) < 1e-8
    # Each (date, sector) group with >=2 members has std=1 of its z-scores
    by_grp = df.assign(z=z).groupby(["date", "sector"])["z"].std()
    by_grp_ok = by_grp.dropna()
    assert np.allclose(by_grp_ok, 1.0, atol=1e-6)

    rk = per_ticker_rank(df)
    assert len(rk) == len(df)
    assert rk[df["ticker"] == "AAA"].max() == 250
    assert rk[df["ticker"] == "AAA"].min() == 1

    print(f"rvol_20 notna count:    {rvol.notna().sum()}")
    print(f"AAA last rvol:          {aaa_last:.6f}")
    print(f"EEE last rvol:          {eee_last:.6f}")
    print(f"sector_z mean (~0):     {z.mean():.2e}")
    print(f"sector_z notna:         {z.notna().sum()}")
    print(f"AAA max rank:           {int(rk[df['ticker']=='AAA'].max())}")
    print("\n✓ All checks passed.")
