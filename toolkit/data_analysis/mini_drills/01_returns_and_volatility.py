"""
DRILL 1 — Returns and Volatility
=================================

OBJECTIVE
    Given a daily price series, compute simple returns, annualised volatility,
    and a rolling 20-day annualised vol.

ESTIMATED TIME
    15 min

TOPICS
    pandas.Series.pct_change, .rolling, .ewm
    numpy.std, numpy.sqrt
    Annualisation convention: sigma_ann = sigma_daily * sqrt(252)

EXPECTED OUTPUT (when implemented correctly)
    n returns:           251
    daily mean:          0.000431
    daily std:           0.011624
    annualised vol:      18.45 %
    last 20d rvol:       19.68 %
    final price:         110.27

GRADING
    Run this script. All asserts must pass and printed numbers must match.
"""
import numpy as np
import pandas as pd

# ── synthetic data ─────────────────────────────────────────────────────────
np.random.seed(42)
n_days = 252
dates = pd.bdate_range("2024-01-01", periods=n_days)
returns_true = np.random.normal(0.0005, 0.012, n_days)
prices = 100.0 * (1 + returns_true).cumprod()
df = pd.DataFrame({"close": prices}, index=dates)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def daily_returns(df: pd.DataFrame) -> pd.Series:
    """Return the daily simple returns from df['close']. Drop the first NaN.

    Length should be len(df) - 1. Use pandas idiomatic .pct_change().
    """
    return df["close"].pct_change().iloc[1:]


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def annualised_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised volatility = sample std * sqrt(periods_per_year)."""
    return returns.std() * np.sqrt(periods_per_year)


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def rolling_vol(returns: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """20-day rolling annualised vol. Leading rows are NaN. Use .rolling()."""
    return (returns * np.sqrt(periods_per_year)).rolling(window=window).std()


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rets = daily_returns(df)
    assert len(rets) == n_days - 1, f"len mismatch: {len(rets)}"
    assert abs(rets.mean() - 0.000431) < 1e-5
    assert abs(rets.std() - 0.011624) < 1e-5

    ann_vol = annualised_vol(rets)
    assert abs(ann_vol - 0.184518) < 1e-4, f"ann vol off: {ann_vol}"

    rvol = rolling_vol(rets)
    assert rvol.isna().sum() >= 19, "rolling should produce >= window-1 NaNs"
    assert abs(rvol.dropna().iloc[-1] - 0.196847) < 1e-3

    print(f"n returns:           {len(rets)}")
    print(f"daily mean:          {rets.mean():.6f}")
    print(f"daily std:           {rets.std():.6f}")
    print(f"annualised vol:      {ann_vol * 100:.2f} %")
    print(f"last 20d rvol:       {rvol.dropna().iloc[-1] * 100:.2f} %")
    print(f"final price:         {df['close'].iloc[-1]:.2f}")
    print("\n✓ All checks passed.")
