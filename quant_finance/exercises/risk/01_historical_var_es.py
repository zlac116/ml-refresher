"""
RISK 1 — Historical VaR + Expected Shortfall
============================================

OBJECTIVE
    On a 1000-day P&L return series with two large injected losses, compute:
      1. 99% and 95% historical 1-day Value at Risk (VaR).
      2. 99% and 95% Expected Shortfall (a.k.a. CVaR, AVaR).
      3. Translate both to $ amounts on a $10M notional.

ESTIMATED TIME
    15 min

TOPICS
    Historical VaR = -percentile(returns, 100*(1-conf))   (sign convention: loss>0)
    Expected Shortfall = - mean of returns at or below -VaR
    Why ES is "coherent" (sub-additive) but VaR is not
    Basel III shift from VaR to ES for trading-book capital (FRTB)

REAL-WORLD NOTE
    Production VaR uses 1y / 250-day rolling window (more for ES). Tail
    estimates with n<500 are unreliable. ES is typically 1.25-1.5x VaR
    for plausible loss distributions.

REFERENCE
    Jorion, "Value at Risk", 3rd ed.; BIS d352 (FRTB ES requirements).

EXPECTED OUTPUT  (seed=42, 1000 daily returns N(0.0005, 0.012), 2 tail injects)
    99% historical VaR  = 0.025887
    95% historical VaR  = 0.018112
    99% historical ES   = 0.035341
    95% historical ES   = 0.024104
    $ VaR99 at $10M     = 258871.51
    $ ES99 at $10M      = 353409.39

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
_n = 1000
_r = np.random.normal(0.0005, 0.012, _n)
_r[-1] = -0.06
_r[-2] = -0.05
returns = pd.Series(_r, name="ret")


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def historical_var(returns: pd.Series, conf: float) -> float:
    """1-day historical VaR at `conf` (e.g. 0.99). Returns the LOSS magnitude
    (positive). Use np.percentile with 100*(1-conf).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def historical_es(returns: pd.Series, conf: float) -> float:
    """Expected Shortfall at `conf`: -mean of returns at or below -VaR(conf).
    Returns the LOSS magnitude.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def dollar_var(returns: pd.Series, conf: float, notional: float) -> float:
    """notional * historical_var(returns, conf)."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    v99 = historical_var(returns, 0.99)
    v95 = historical_var(returns, 0.95)
    assert abs(v99 - 0.025887) < 1e-5, v99
    assert abs(v95 - 0.018112) < 1e-5
    # ES > VaR always
    assert v99 > v95

    es99 = historical_es(returns, 0.99)
    es95 = historical_es(returns, 0.95)
    assert abs(es99 - 0.035341) < 1e-5
    assert abs(es95 - 0.024104) < 1e-5
    assert es99 > v99 and es95 > v95     # ES > VaR by construction

    dv99 = dollar_var(returns, 0.99, 10_000_000)
    assert abs(dv99 - 258871.51) < 1e-1

    print(f"99% historical VaR  = {v99:.6f}")
    print(f"95% historical VaR  = {v95:.6f}")
    print(f"99% historical ES   = {es99:.6f}")
    print(f"95% historical ES   = {es95:.6f}")
    print(f"$ VaR99 at $10M     = {dv99:.2f}")
    print(f"$ ES99 at $10M      = {es99 * 10_000_000:.2f}")
    print("\n✓ All checks passed.")
