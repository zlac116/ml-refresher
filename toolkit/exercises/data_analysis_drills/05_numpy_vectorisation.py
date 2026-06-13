"""
DRILL 5 — NumPy Vectorisation
=============================

OBJECTIVE
    No for-loops over portfolios or assets. Compute per-portfolio P&L,
    capped returns, and quadratic-form variances using broadcasting,
    np.where, and np.einsum.

ESTIMATED TIME
    15 min

TOPICS
    numpy broadcasting (rows × columns)
    np.where for conditional element-wise transforms
    np.einsum 'ij,jk,ik->i' for batched x' S x

EXPECTED OUTPUT
    pnl shape:           (1000,)
    pnl mean:            -0.000181
    pnl std:             0.003333
    pnl p5/p95:          -0.005574 / 0.005428
    capped sum:          0.040877
    median port var:     1.24e-05

GRADING
    All asserts must pass. No `for` keyword in your function bodies.
"""
import numpy as np

np.random.seed(42)
n_ports, n_assets = 1000, 50
weights = np.random.normal(0, 1, (n_ports, n_assets))
weights /= np.abs(weights).sum(axis=1, keepdims=True)        # gross = 1 each row
returns = np.random.normal(0.0005, 0.02, n_assets)
cov = np.eye(n_assets) * 0.0004


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def portfolio_pnl(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Per-portfolio P&L. Shape (n_ports,). Use matrix multiply or @ — no loops."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def cap_returns(returns: np.ndarray, cap: float = 0.02) -> np.ndarray:
    """Clip each return to [-cap, +cap]. Use np.where or np.clip; no loops."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def portfolio_variances(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Batched quadratic form w_i' Σ w_i for every portfolio i.

    Shape (n_ports,). Use np.einsum 'ij,jk,ik->i'. No loops.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pnl = portfolio_pnl(weights, returns)
    assert pnl.shape == (n_ports,), pnl.shape
    assert abs(pnl.mean() - -0.000181) < 1e-5
    assert abs(pnl.std()  -  0.003333) < 1e-5
    assert abs(np.percentile(pnl,  5) - -0.005574) < 1e-4
    assert abs(np.percentile(pnl, 95) -  0.005428) < 1e-4

    capped = cap_returns(returns, cap=0.02)
    assert capped.shape == returns.shape
    assert (capped >= -0.02 - 1e-12).all() and (capped <= 0.02 + 1e-12).all()
    assert abs(capped.sum() - 0.040877) < 1e-4

    var = portfolio_variances(weights, cov)
    assert var.shape == (n_ports,)
    assert (var > 0).all()
    assert abs(np.percentile(var, 50) - 1.242e-5) < 1e-7, np.percentile(var, 50)

    import inspect
    for fn in (portfolio_pnl, cap_returns, portfolio_variances):
        src = inspect.getsource(fn)
        assert "for " not in src and " for(" not in src, f"loops detected in {fn.__name__}"

    print(f"pnl shape:           {pnl.shape}")
    print(f"pnl mean:            {pnl.mean():.6f}")
    print(f"pnl std:             {pnl.std():.6f}")
    print(f"pnl p5/p95:          {np.percentile(pnl, 5):.6f} / {np.percentile(pnl, 95):.6f}")
    print(f"capped sum:          {capped.sum():.6f}")
    print(f"median port var:     {np.percentile(var, 50):.2e}")
    print("\n✓ All checks passed.")
