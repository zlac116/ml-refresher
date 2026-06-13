"""
DRILL 8 — Rolling Windows + Subplots
====================================

OBJECTIVE
    From a daily price series build two SMAs (20, 50), a 20-day rolling
    annualised vol, and an expanding mean. Render a 2x1 subplot figure:
    top = price + both SMAs; bottom = rolling vol on its own axis.

ESTIMATED TIME
    20 min

TOPICS
    pandas rolling vs expanding (when to pick which)
    matplotlib subplots layout, twin axes
    SMA crossover signal (boolean Series sum)

EXPECTED OUTPUT
    price last:          129.23
    SMA-20 last:         131.09
    SMA-50 last:         129.95
    20d rvol last:       19.38 %
    expanding mean:      110.71
    days SMA20 > SMA50:  300
    figure saved to:     /tmp/drill08_subplots.png

GRADING
    All asserts must pass and PNG must be non-trivial.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500
dates = pd.bdate_range("2023-01-01", periods=n)
_rets = pd.Series(np.random.normal(0.0005, 0.012, n), index=dates, name="ret")
prices = (100.0 * (1 + _rets).cumprod()).rename("close")


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def add_smas(prices: pd.Series) -> pd.DataFrame:
    """Return a DataFrame with columns ['close', 'sma_20', 'sma_50']
    using prices.rolling(window).mean().
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def rolling_ann_vol(returns: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """20-day rolling annualised vol of `returns`."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def crossover_days(sma_fast: pd.Series, sma_slow: pd.Series) -> int:
    """Count days where sma_fast > sma_slow (drop NaNs first)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def render(price_smas: pd.DataFrame, rvol: pd.Series, path: str) -> None:
    """2x1 subplot figure:
        - top:    price + sma_20 + sma_50, with a legend
        - bottom: rvol as %, with a horizontal axhline at its mean
    Use plt.subplots(2, 1, sharex=True, figsize=(10, 6)).
    Save with plt.savefig(path); plt.close('all') at end.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    smas = add_smas(prices)
    assert list(smas.columns) == ["close", "sma_20", "sma_50"]
    assert abs(smas["close"].iloc[-1]  - 129.2309) < 1e-3
    assert abs(smas["sma_20"].iloc[-1] - 131.0877) < 1e-3
    assert abs(smas["sma_50"].iloc[-1] - 129.9535) < 1e-3

    rvol = rolling_ann_vol(_rets)
    assert abs(rvol.iloc[-1] - 0.1938) < 1e-3

    days = crossover_days(smas["sma_20"], smas["sma_50"])
    assert days == 300, f"expected 300, got {days}"

    out = "/tmp/drill08_subplots.png"
    render(smas, rvol, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5000

    print(f"price last:          {smas['close'].iloc[-1]:.2f}")
    print(f"SMA-20 last:         {smas['sma_20'].iloc[-1]:.2f}")
    print(f"SMA-50 last:         {smas['sma_50'].iloc[-1]:.2f}")
    print(f"20d rvol last:       {rvol.iloc[-1] * 100:.2f} %")
    print(f"expanding mean:      {smas['close'].expanding().mean().iloc[-1]:.2f}")
    print(f"days SMA20 > SMA50:  {days}")
    print(f"figure saved to:     {out}")
    print("\n✓ All checks passed.")
