"""
DRILL 7 — Outlier Detection + Winsorization
===========================================

OBJECTIVE
    Detect outliers using both the z-score and Tukey IQR rules; winsorize
    a series at the 1st/99th percentiles.

ESTIMATED TIME
    15 min

TOPICS
    Series arithmetic for z-scores
    .quantile, .clip (canonical pandas winsorization)
    Boolean indexing / .abs()

EXPECTED OUTPUT
    |z| > 3 outliers:    10
    IQR-rule outliers:   18
    raw max:             8.938284
    winsor max:          3.878260
    winsor mean:         0.062748

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
_clean = np.random.normal(0, 1, 1000)
_clean[::100] = _clean[::100] + 8.0      # 10 outliers injected at indices 0,100,200,…
s = pd.Series(_clean, name="x")


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def count_zscore_outliers(s: pd.Series, threshold: float = 3.0) -> int:
    """Count points with |z-score| > threshold. z = (x - mean) / std."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def count_iqr_outliers(s: pd.Series, k: float = 1.5) -> int:
    """Count points outside [Q1 - k*IQR, Q3 + k*IQR]. Tukey's fences."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def winsorize(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Clip to [lower_q quantile, upper_q quantile]. Use .clip(...)."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n_z = count_zscore_outliers(s)
    assert n_z == 10, f"expected 10 z-outliers, got {n_z}"

    n_iqr = count_iqr_outliers(s)
    assert n_iqr == 18, f"expected 18 IQR-outliers, got {n_iqr}"

    w = winsorize(s)
    assert len(w) == len(s)
    assert abs(w.max() - 3.878260) < 1e-4, w.max()
    assert abs(w.mean() - 0.062748) < 1e-4
    # winsor reduces the max but never moves it below the 99th percentile of s
    assert w.max() < s.max()

    print(f"|z| > 3 outliers:    {n_z}")
    print(f"IQR-rule outliers:   {n_iqr}")
    print(f"raw max:             {s.max():.6f}")
    print(f"winsor max:          {w.max():.6f}")
    print(f"winsor mean:         {w.mean():.6f}")
    print("\n✓ All checks passed.")
