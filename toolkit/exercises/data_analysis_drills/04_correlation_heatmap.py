"""
DRILL 4 — Correlation Matrix + Heatmap
=======================================

OBJECTIVE
    Compute the correlation matrix of a 4-asset returns frame and render it
    as an annotated heatmap with matplotlib's imshow.

ESTIMATED TIME
    20 min

TOPICS
    pandas.DataFrame.corr / .cov
    matplotlib.pyplot.imshow + Axes.text annotations + colorbar

EXPECTED OUTPUT
    corr A-B:            0.2904
    corr A-D:           -0.2920
    corr C-D:           -0.1013
    cov diag:            [1.4586, 1.3092, 1.0597, 1.2725]
    figure saved to:     /tmp/drill04_corr.png

GRADING
    All asserts must pass. PNG should render with 4x4 annotated cells.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500
_factor = np.random.normal(0, 1, n)
_idio = np.random.normal(0, 1, (n, 4))
_loadings = np.array([0.8, 0.6, 0.3, -0.5])
_returns = _loadings[None, :] * _factor[:, None] + _idio
df = pd.DataFrame(_returns, columns=list("ABCD"))


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix. Use the pandas idiomatic method."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def plot_heatmap(corr: pd.DataFrame, path: str) -> None:
    """Render `corr` as an imshow heatmap with annotations + colorbar.

    Requirements:
      - vmin=-1, vmax=1, cmap='coolwarm' or 'RdBu_r'
      - Each cell text-annotated with value to 2 d.p.
      - x/y tick labels = column names
      - Save with plt.savefig(path); call plt.close('all') at end.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    corr = correlation_matrix(df)
    assert corr.shape == (4, 4)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert abs(corr.loc["A", "B"] -  0.2904) < 1e-3
    assert abs(corr.loc["A", "D"] - -0.2920) < 1e-3
    assert abs(corr.loc["C", "D"] - -0.1013) < 1e-3
    # Symmetric
    assert np.allclose(corr.values, corr.values.T)

    cov = covariance_matrix(df)
    diag = cov.values.diagonal()
    expected_diag = np.array([1.458588, 1.309198, 1.059672, 1.272523])
    assert np.allclose(diag, expected_diag, atol=1e-4), diag

    out = "/tmp/drill04_corr.png"
    plot_heatmap(corr, out)
    import os
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000  # non-trivial PNG

    print(f"corr A-B:            {corr.loc['A','B']:.4f}")
    print(f"corr A-D:            {corr.loc['A','D']:.4f}")
    print(f"corr C-D:            {corr.loc['C','D']:.4f}")
    print(f"cov diag:            {[round(x, 4) for x in diag]}")
    print(f"figure saved to:     {out}")
    print("\n✓ All checks passed.")
