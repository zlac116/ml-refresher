"""
PORT 3 — Backtest: Buy-and-Hold vs Monthly Rebalanced Equal-Weight
==================================================================

OBJECTIVE
    Backtest two strategies on a 252-day random-return universe of 4 assets:
      1. Buy-and-hold equal weight (set 25/25/25/25 on day 0, no rebalancing —
         weights DRIFT as relative prices move).
      2. Equal weight rebalanced at every business-month-end.

    For each compute: total return, Sharpe, max drawdown.

ESTIMATED TIME
    20 min

TOPICS
    Wealth trajectory: holdings[i] = holdings[i] * (1 + ret[i]) per day
    Drift: weights move away from target between rebalances
    Rebalance event: redistribute total wealth to target weights (no costs)
    Max drawdown: min((value - running_max) / running_max)

REAL-WORLD NOTE
    Real backtests must subtract transaction costs (5-10 bps on rebalance),
    enforce settlement (T+1, T+2), and handle dividends/corporate actions.
    Here we backtest gross of all of those for clarity.

REFERENCE
    Sharpe (1966), "Mutual Fund Performance"; Maillard et al (2010).

EXPECTED OUTPUT  (seed=42, 252 days, 4 assets, 12 month-end rebalances)
    buy-hold total ret:    0.199255
    rebal    total ret:    0.205482
    buy-hold Sharpe:       1.868992
    rebal    Sharpe:       1.927606
    buy-hold max DD:       -0.058020
    rebal    max DD:       -0.057223
    n rebalance days:      12

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
_n_days, _n_assets = 252, 4
_dates = pd.bdate_range("2024-01-01", periods=_n_days)
rets = pd.DataFrame(np.random.normal(0.0005, 0.012, (_n_days, _n_assets)),
                    index=_dates, columns=["A", "B", "C", "D"])


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def buy_and_hold_wealth(rets: pd.DataFrame, init_weights: np.ndarray) -> pd.Series:
    """Hold init_weights (must sum to 1) on day 0, then no rebalancing.

    wealth_t = sum_i (init_weights[i] * (1 + ret_path[i, :t]).cumprod())
    Returns a wealth Series indexed by date, starting at value 1 at day 0
    (or first day's already-applied return — be consistent).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def month_end_dates(dates: pd.DatetimeIndex) -> set:
    """Return the LAST business day of each calendar month present in `dates`.

    Hint: groupby on dates.to_period('M'), pick max date per group.
    Return as a set (or frozenset) for fast 'in' lookup.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def rebalanced_wealth(rets: pd.DataFrame, target_weights: np.ndarray,
                      rebal_dates: set) -> pd.Series:
    """Maintain holdings array; apply daily returns; at each rebal day,
    redistribute total wealth to target_weights.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def sharpe_ratio(daily_pct_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe = mean / std * sqrt(periods_per_year)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def max_drawdown(wealth: pd.Series) -> float:
    """min over t of (wealth_t - running_max_t) / running_max_t. Negative number."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    target = np.array([0.25, 0.25, 0.25, 0.25])

    bh = buy_and_hold_wealth(rets, target)
    assert isinstance(bh, pd.Series)
    assert abs(bh.iloc[-1] - 1.199255) < 1e-4

    rebal = month_end_dates(_dates)
    assert len(rebal) == 12

    rb = rebalanced_wealth(rets, target, rebal)
    assert abs(rb.iloc[-1] - 1.205482) < 1e-4

    bh_ret = bh.pct_change().dropna()
    rb_ret = rb.pct_change().dropna()

    s_bh = sharpe_ratio(bh_ret)
    s_rb = sharpe_ratio(rb_ret)
    assert abs(s_bh - 1.868992) < 1e-4
    assert abs(s_rb - 1.927606) < 1e-4

    dd_bh = max_drawdown(bh)
    dd_rb = max_drawdown(rb)
    assert abs(dd_bh - -0.058020) < 1e-4
    assert abs(dd_rb - -0.057223) < 1e-4

    print(f"buy-hold total ret:    {bh.iloc[-1] - 1:.6f}")
    print(f"rebal    total ret:    {rb.iloc[-1] - 1:.6f}")
    print(f"buy-hold Sharpe:       {s_bh:.6f}")
    print(f"rebal    Sharpe:       {s_rb:.6f}")
    print(f"buy-hold max DD:       {dd_bh:.6f}")
    print(f"rebal    max DD:       {dd_rb:.6f}")
    print(f"n rebalance days:      {len(rebal)}")
    print("\n✓ All checks passed.")
