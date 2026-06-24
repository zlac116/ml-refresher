"""
PROJECT — matplotlib: One-Page Strategy Tearsheet
==================================================

OBJECTIVE
    Build a 2×3 subplot figure that summarises a strategy's performance:

      Panel 1: Equity curve (strategy vs benchmark, log-scale optional)
      Panel 2: Drawdown curve with worst-DD annotation
      Panel 3: Rolling 30-day Sharpe
      Panel 4: Return distribution (histogram)
      Panel 5: Monthly return heatmap (year × month)
      Panel 6: Strategy vs benchmark scatter (β + R² annotated)

ESTIMATED TIME
    30 min

TOPICS
    plt.subplots(2, 3, figsize=(15, 9))
    Ax.fill_between for drawdown shading
    pivot_table → imshow heatmap
    Annotation via ax.annotate or ax.text
    plt.tight_layout + plt.savefig

REAL-WORLD NOTE
    Synthetic strategy + benchmark (deterministic seed). The point is the
    PLOTTING patterns — substitute your own returns and the same code works.

EXPECTED OUTPUT
    n days:           756
    strat sharpe:     0.3733
    strat final eq:   1.1709
    strat max DD:    -22.62 %
    figure saved to: /tmp/tearsheet.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
N_DAYS = 252 * 3
DATES = pd.bdate_range("2023-01-01", periods=N_DAYS)
strat_rets = pd.Series(np.random.normal(0.0005, 0.012, N_DAYS), index=DATES, name="strat")
bench_rets = pd.Series(np.random.normal(0.0003, 0.010, N_DAYS), index=DATES, name="bench")


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def equity_and_drawdown(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (equity, drawdown) where:
        equity     = (1 + returns).cumprod()
        drawdown   = (equity - running_max) / running_max     (negative values)
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def rolling_sharpe(returns: pd.Series, window: int = 30, periods_per_year: int = 252) -> pd.Series:
    """rolling_mean / rolling_std * sqrt(periods_per_year). NaN for first window-1."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def monthly_return_pivot(returns: pd.Series) -> pd.DataFrame:
    """Pivot returns into a (year × month) matrix of monthly compounded returns.

    Hint: groupby([year, month]).apply(lambda r: (1+r).prod()-1).unstack(level='month')
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def render_tearsheet(strat: pd.Series, bench: pd.Series, save_path: str) -> None:
    """Build the 2×3 tearsheet and save it to `save_path` (PNG).

    Panels (top-left → bottom-right):
      (0,0) Equity curve   — strat & bench, with legend
      (0,1) Drawdown curve — fill_between zero and dd, annotate worst point
      (0,2) Rolling Sharpe — 30-day window
      (1,0) Return histogram (strat), with bin count e.g. 40
      (1,1) Monthly return heatmap — use imshow + annotate cells; add a colorbar
      (1,2) Scatter strat vs bench daily returns + best-fit line

    Use plt.tight_layout(); plt.close('all') at the end.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    eq, dd = equity_and_drawdown(strat_rets)
    assert len(eq) == N_DAYS
    assert abs(eq.iloc[-1] - 1.1709) < 1e-3
    assert abs(dd.min() - -0.2262) < 1e-3
    assert (dd <= 0).all()  # drawdowns are non-positive by construction

    rs = rolling_sharpe(strat_rets)
    assert len(rs) == N_DAYS
    assert rs.isna().sum() >= 29   # first 29 rows NaN

    mp = monthly_return_pivot(strat_rets)
    assert mp.shape[1] == 12      # 12 columns for months
    assert mp.shape[0] >= 3       # at least 3 years

    out = "/tmp/tearsheet.png"
    render_tearsheet(strat_rets, bench_rets, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 20_000   # non-trivial 2x3 figure

    sharpe = strat_rets.mean() / strat_rets.std() * np.sqrt(252)
    print(f"n days:           {N_DAYS}")
    print(f"strat sharpe:     {sharpe:.4f}")
    print(f"strat final eq:   {eq.iloc[-1]:.4f}")
    print(f"strat max DD:    {dd.min()*100:.2f} %")
    print(f"figure saved to: {out}")
    print("\n✓ All checks passed.")
