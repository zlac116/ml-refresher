"""
PORT 1 — Mean-Variance Optimisation (long-only, target return)
==============================================================

OBJECTIVE
    Solve the classic Markowitz problem for 4 risky assets:
        min   0.5 * w' Σ w
        s.t.  w' mu = target_return
              sum(w) = 1
              0 <= w_i <= 1     (long-only, fully invested)
    Use scipy.optimize.minimize with SLSQP.

ESTIMATED TIME
    20 min

TOPICS
    Markowitz (1952) mean-variance frontier
    SLSQP for equality + inequality constraints (canonical scipy choice)
    Convexity: the problem is QP with linear constraints — single global optimum

REAL-WORLD NOTE
    Real allocators add: sector caps, turnover penalty, transaction costs,
    Black-Litterman views. SLSQP scales to ~100 assets; bigger problems
    use cvxpy or commercial QP solvers (OSQP).

REFERENCE
    Markowitz, "Portfolio Selection", J. of Finance, 1952.
    scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html

EXPECTED OUTPUT  (mu=[8,10,12,7]%, target=9%)
    weights:        [0.285098, 0.208800, 0.217700, 0.288402]
    sum w:          1.0000000
    portfolio ret:  0.090000
    portfolio vol:  0.106389
    Sharpe (rf=3%): 0.563971

GRADING
    All asserts must pass; sum of weights must equal 1 exactly.
"""
import numpy as np
from scipy.optimize import minimize


MU = np.array([0.08, 0.10, 0.12, 0.07])
VOLS = np.array([0.15, 0.20, 0.25, 0.12])
CORR = np.array([
    [1.0, 0.2, 0.1, 0.0],
    [0.2, 1.0, 0.3, 0.1],
    [0.1, 0.3, 1.0, 0.2],
    [0.0, 0.1, 0.2, 1.0],
])
COV = np.outer(VOLS, VOLS) * CORR


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def optimise_min_variance_target(mu: np.ndarray, cov: np.ndarray,
                                 target_return: float) -> np.ndarray:
    """Solve the Markowitz problem.

    - Objective: w @ cov @ w   (no factor of 0.5; SLSQP doesn't care)
    - Equality constraints: sum(w) - 1 = 0   AND  w @ mu - target_return = 0
    - Bounds: 0 <= w_i <= 1
    - Initial guess: equal weights
    - method='SLSQP'

    Returns the optimal weight vector.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def portfolio_return_and_vol(w: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    """Return (port_return, port_vol)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def sharpe(port_return: float, port_vol: float, rf: float = 0.03) -> float:
    """(port_return - rf) / port_vol."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    w = optimise_min_variance_target(MU, COV, target_return=0.09)
    assert w.shape == (4,)
    expected = np.array([0.285098, 0.208800, 0.217700, 0.288402])
    assert np.allclose(w, expected, atol=1e-4), w
    assert abs(w.sum() - 1.0) < 1e-8
    assert (w >= -1e-8).all() and (w <= 1 + 1e-8).all()

    pr, pv = portfolio_return_and_vol(w, MU, COV)
    assert abs(pr - 0.09)     < 1e-6
    assert abs(pv - 0.106389) < 1e-4

    sh = sharpe(pr, pv, rf=0.03)
    assert abs(sh - 0.563971) < 1e-4

    print(f"weights:        {w.round(6).tolist()}")
    print(f"sum w:          {w.sum():.7f}")
    print(f"portfolio ret:  {pr:.6f}")
    print(f"portfolio vol:  {pv:.6f}")
    print(f"Sharpe (rf=3%): {sh:.6f}")
    print("\n✓ All checks passed.")
