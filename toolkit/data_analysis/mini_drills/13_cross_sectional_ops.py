"""
DRILL 13 — Cross-Sectional Operations (per-date normalisation, ranks, top/bottom)
================================================================================

OBJECTIVE
    On a wide (dates × assets) returns DataFrame, run the canonical
    cross-sectional moves used in factor / stat-arb research:
      1. Per-date z-score across assets (axis=1 normalisation).
      2. Per-date percentile rank (1..n).
      3. Build a long-top-quintile, short-bottom-quintile, equal-weight
         daily portfolio return series.

ESTIMATED TIME
    20 min

TOPICS
    Series/DataFrame .sub(.., axis=0) and .div(.., axis=0) — broadcasting along axis
    .rank(axis=1), .rank(axis=1, pct=True)
    .shift(-1) for next-day forward returns (signal-aligned)

CONVENTION
    Cross-sectional means PER DATE across assets — axis=1.
    Top-quintile means percentile rank > 0.8, bottom is <= 0.2.

EXPECTED OUTPUT
    xs_z shape:           (50, 20)
    xs_z row 0 mean (~0): 0.00e+00
    xs_z row 0 std (=1):  1.000000
    xs_rank min/max:      1 / 20
    top-quintile picks:   4 per day (20 * 0.2)
    strat mean ret:       -0.002204
    strat std:            0.008096

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
_n_days, _n_assets = 50, 20
_dates = pd.bdate_range("2024-01-01", periods=_n_days)
_cols = [f"A{i:02d}" for i in range(_n_assets)]
rets = pd.DataFrame(np.random.normal(0.0005, 0.015, (_n_days, _n_assets)),
                    index=_dates, columns=_cols)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def cross_sectional_zscore(rets: pd.DataFrame) -> pd.DataFrame:
    """Per-date (row) z-score across assets: (x - row_mean) / row_std.

    Use .sub(..., axis=0) and .div(..., axis=0). Same shape as input.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def cross_sectional_rank(rets: pd.DataFrame) -> pd.DataFrame:
    """Per-date ordinal rank across assets, 1..n_assets. Use .rank(axis=1)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def long_short_quintile_return(rets: pd.DataFrame) -> pd.Series:
    """Each day, equal-weight long the top quintile (pct rank > 0.8) and
    equal-weight short the bottom quintile (pct rank <= 0.2). Hold for one
    day (so use shift(-1) on rets to look at NEXT-day returns for the
    portfolio P&L). Return the per-day strategy return Series.

    Final row will be NaN (no next-day return available).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    z = cross_sectional_zscore(rets)
    assert z.shape == rets.shape
    assert abs(z.iloc[0].mean()) < 1e-10
    assert abs(z.iloc[0].std() - 1.0) < 1e-10

    rk = cross_sectional_rank(rets)
    assert rk.shape == rets.shape
    assert rk.iloc[0].min() == 1
    assert rk.iloc[0].max() == 20

    strat = long_short_quintile_return(rets)
    assert isinstance(strat, pd.Series)
    assert len(strat) == _n_days
    assert pd.isna(strat.iloc[-1]), "last row should be NaN (no next-day return)"
    s = strat.dropna()
    assert abs(s.mean() - -0.002204) < 1e-4, f"mean: {s.mean()}"
    assert abs(s.std()  -  0.008096) < 1e-4

    print(f"xs_z shape:           {z.shape}")
    print(f"xs_z row 0 mean (~0): {z.iloc[0].mean():.2e}")
    print(f"xs_z row 0 std (=1):  {z.iloc[0].std():.6f}")
    print(f"xs_rank min/max:      {int(rk.iloc[0].min())} / {int(rk.iloc[0].max())}")
    print(f"top-quintile picks:   4 per day (20 * 0.2)")
    print(f"strat mean ret:       {s.mean():.6f}")
    print(f"strat std:            {s.std():.6f}")
    print("\n✓ All checks passed.")
