"""
DRILL 6 — GroupBy + Named Aggregations
======================================

OBJECTIVE
    Aggregate a trades blotter by sector and by (sector, country) using the
    pandas named-agg syntax, then use .transform to compute within-group
    demeaned returns.

ESTIMATED TIME
    20 min

TOPICS
    pandas.DataFrame.groupby
    Named aggregation: .agg(out=('col', 'func'), ...)
    .transform vs .apply: when to use which

EXPECTED OUTPUT
    n trades:            1000
    Energy n / mean ret: 319 / 0.001718
    Tech total notional: 2.025849e+09
    Tech-US mean ret:    -0.000113
    Demeaned mean:       0.0 (machine epsilon)
    Demeaned |max|:      0.050703

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "sector":   np.random.choice(["Tech", "Fin", "Energy"], n),
    "country":  np.random.choice(["US", "UK", "JP"], n),
    "ret":      np.random.normal(0.0005, 0.015, n),
    "notional": np.random.uniform(1e6, 1e7, n),
})


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Group by sector, return a frame with columns
        n_trades, mean_ret, total_notional.

    Use the canonical named-aggregation syntax (Series-style):
        .agg(n_trades=('ret', 'size'), ...)
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def by_sector_country(df: pd.DataFrame) -> pd.DataFrame:
    """Group by [sector, country], compute mean of 'ret', UNSTACK country.

    Returns a frame indexed by sector with columns ['JP','UK','US'].
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def demean_within_sector(df: pd.DataFrame) -> pd.Series:
    """Return df['ret'] - per-sector-mean, aligned with df's index.

    Use .groupby(...)['ret'].transform('mean') — NOT .apply, NOT .agg.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sec = by_sector(df)
    assert set(sec.columns) == {"n_trades", "mean_ret", "total_notional"}
    assert sec.loc["Energy", "n_trades"] == 319
    assert abs(sec.loc["Energy", "mean_ret"] - 0.001718) < 1e-5
    assert abs(sec.loc["Tech",   "total_notional"] - 2.025849e9) < 1e3

    sc = by_sector_country(df)
    assert list(sc.columns) == ["JP", "UK", "US"], sc.columns.tolist()
    assert abs(sc.loc["Tech", "US"] - -0.000113) < 1e-5

    dmean = demean_within_sector(df)
    assert len(dmean) == len(df)
    assert abs(dmean.mean()) < 1e-10
    assert abs(dmean.abs().max() - 0.050703) < 1e-5

    print(f"n trades:            {len(df)}")
    print(f"Energy n / mean ret: {sec.loc['Energy','n_trades']} / "
          f"{sec.loc['Energy','mean_ret']:.6f}")
    print(f"Tech total notional: {sec.loc['Tech','total_notional']:.6e}")
    print(f"Tech-US mean ret:    {sc.loc['Tech','US']:.6f}")
    print(f"Demeaned mean:       {dmean.mean():.2e}")
    print(f"Demeaned |max|:      {dmean.abs().max():.6f}")
    print("\n✓ All checks passed.")
