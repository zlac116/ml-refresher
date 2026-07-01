"""
PROJECT — numpy + scipy: Equity Curves, Drawdowns, Bootstrap Sharpe
====================================================================

OBJECTIVE
    On the BTC/ETH/SOL/BNB hourly dataset:
      1. Build per-symbol equity curves from log returns.
      2. Compute max drawdown via running-max (vectorised, no for loops).
      3. Characterise return distributions (skew + excess kurtosis).
      4. Bootstrap a 95% CI on BTC's annualised Sharpe.

ESTIMATED TIME
    30 min

TOPICS
    np.cumsum + np.exp for cumulative-product-style equity
    np.maximum.accumulate (NOT a for loop)
    scipy.stats.skew, scipy.stats.kurtosis (excess by default)
    Block bootstrap with np.random.choice + np.percentile

EXPECTED OUTPUT
    equity shape:       (17520, 4)
    BTC final equity:   1.1607
    BTC max drawdown:   0.5008
    ETH max drawdown:   0.6528
    BTC return moments: mean≈9e-6, std≈0.0051, skew≈-0.17, ex-kurt≈9.42
    BTC Sharpe (h):     0.1575
    BTC Sharpe 95% CI:  [-1.17, 1.51]

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd
from scipy import stats

DATA = "/home/zlac116/Code/learning/ml-revision/data/crypto_hourly.parquet"
df = pd.read_parquet(DATA).sort_values(["symbol", "ts"]).reset_index(drop=True)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def equity_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-symbol equity curves from hourly closes.

    Returns a wide DataFrame indexed by ts, columns = symbols, starting at 1.0.
    Use log returns + cumsum + exp (avoids floating-point drift of (1+r).cumprod()).
    """
    # TODO: implement
    #   1. per-symbol logret = log(close).diff()
    #   2. pivot to wide (ts, symbol) and fillna(0) on first row per symbol
    #   3. equity = exp(cumsum(logret))
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def max_drawdown(equity: pd.DataFrame) -> pd.Series:
    """Maximum drawdown PER SYMBOL: -min((equity - running_max) / running_max).

    MUST use np.maximum.accumulate — no for loops.
    Returns positive numbers (= magnitudes).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def return_moments(df: pd.DataFrame, symbol: str) -> dict:
    """Return dict with keys: mean, std, skew, ex_kurt for the given symbol's
    log returns. Use scipy.stats.skew and scipy.stats.kurtosis (default = excess).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def bootstrap_sharpe_ci(returns: np.ndarray, n_boot: int = 1000,
                        periods_per_year: int = 365 * 24, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap (resample with replacement) the annualised Sharpe ratio.

    Returns (point_estimate, ci_low, ci_high) at 95% confidence.
    Use np.random.choice(returns, len(returns), replace=True) inside a loop.
    Sharpe = mean / std * sqrt(periods_per_year).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    equity = equity_curves(df)
    assert equity.shape == (17520, 4), equity.shape
    assert abs(equity["BTC"].iloc[-1] - 1.1607) < 1e-3

    dd = max_drawdown(equity)
    assert abs(dd["BTC"] - 0.5008) < 1e-3
    assert abs(dd["ETH"] - 0.6528) < 1e-3
    assert (dd > 0).all()

    m = return_moments(df, "BTC")
    assert abs(m["std"]  - 0.005056) < 1e-5
    assert abs(m["skew"] - -0.1704) < 1e-3
    assert abs(m["ex_kurt"] - 9.4185) < 1e-2

    btc_r = df[df.symbol == "BTC"]["close"].pct_change().dropna().apply(np.log1p).values
    s, lo, hi = bootstrap_sharpe_ci(btc_r)
    assert lo < s < hi
    # CI width should be plausible (a few units around 0)
    assert 0.5 < (hi - lo) < 5.0

    print(f"equity shape:       {equity.shape}")
    print(f"BTC final equity:   {equity['BTC'].iloc[-1]:.4f}")
    print(f"BTC max drawdown:   {dd['BTC']:.4f}")
    print(f"ETH max drawdown:   {dd['ETH']:.4f}")
    print(f"BTC return moments: mean={m['mean']:.6f}, std={m['std']:.6f}, "
          f"skew={m['skew']:.4f}, ex-kurt={m['ex_kurt']:.4f}")
    print(f"BTC Sharpe (h):     {s:.4f}")
    print(f"BTC Sharpe 95% CI:  [{lo:.4f}, {hi:.4f}]")
    print("\n✓ All checks passed.")
