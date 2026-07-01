"""
PORT 2 — Risk Parity Portfolio
==============================

OBJECTIVE
    Find weights such that EACH asset contributes equal RISK to the portfolio
    (risk contribution = w_i * (Σw)_i / sigma_p).

    Use scipy.optimize.minimize on the sum-of-squared-deviations from the
    equal-RC target.

ESTIMATED TIME
    20 min

TOPICS
    Risk contribution: RC_i = w_i * MVaR_i  where MVaR_i = (cov @ w)_i / sigma_p
    Sum RC_i = sigma_p (Euler theorem on homogeneous functions)
    Target: RC_i = sigma_p / n for all i

REAL-WORLD NOTE
    Risk parity (a.k.a. equal risk contribution, ERC) underpins All Weather
    funds (Bridgewater) and many alternative-risk-premium strategies.
    Maillard-Roncalli-Teiletche (2010) is the canonical ERC paper.
    Convexity NOT guaranteed for arbitrary covariance — initialize at 1/n.

REFERENCE
    Maillard, Roncalli, Teiletche, "On the properties of equally-weighted risk
    contribution portfolios", J. of Portfolio Management, 2010.

EXPECTED OUTPUT  (same 4-asset cov as PORT 1)
    weights:        [0.291203, 0.191552, 0.153241, 0.364005]
    sum w:          1.0000000
    portfolio vol:  0.098184
    risk contribs:  [0.024546, 0.024546, 0.024546, 0.024546]  (all equal)
    rc spread:      < 1e-6

GRADING
    All asserts must pass; rc spread must be < 1e-5.
"""
import numpy as np
from scipy.optimize import minimize


VOLS = np.array([0.15, 0.20, 0.25, 0.12])
CORR = np.array([
    [1.0, 0.2, 0.1, 0.0],
    [0.2, 1.0, 0.3, 0.1],
    [0.1, 0.3, 1.0, 0.2],
    [0.0, 0.1, 0.2, 1.0],
])
COV = np.outer(VOLS, VOLS) * CORR


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """RC_i = w_i * (cov @ w)_i / sigma_p   where sigma_p = sqrt(w' cov w).
    Returns a length-n vector. Should sum to sigma_p (Euler decomposition).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def risk_parity_objective(w: np.ndarray, cov: np.ndarray) -> float:
    """Sum of squared deviations of RC from equal target (sigma_p / n)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def solve_risk_parity(cov: np.ndarray) -> np.ndarray:
    """Find weights such that risk contributions are equal.
    - Equality constraint: sum(w) - 1 = 0
    - Bounds: w_i in [1e-8, 1]   (long-only, strictly positive)
    - method='SLSQP', options ftol=1e-12, maxiter=1000
    - Initial guess: 1/n equal weights
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    w = solve_risk_parity(COV)
    assert w.shape == (4,)
    expected = np.array([0.291203, 0.191552, 0.153241, 0.364005])
    assert np.allclose(w, expected, atol=1e-4), w
    assert abs(w.sum() - 1.0) < 1e-8

    sigma_p = float(np.sqrt(w @ COV @ w))
    assert abs(sigma_p - 0.098184) < 1e-4

    rc = risk_contributions(w, COV)
    # Each RC must be approximately equal
    assert (rc.max() - rc.min()) < 1e-5, f"rc spread: {rc.max()-rc.min()}"
    # Sum of RCs = portfolio vol
    assert abs(rc.sum() - sigma_p) < 1e-10

    print(f"weights:        {w.round(6).tolist()}")
    print(f"sum w:          {w.sum():.7f}")
    print(f"portfolio vol:  {sigma_p:.6f}")
    print(f"risk contribs:  {rc.round(6).tolist()}  (all equal)")
    print(f"rc spread:      {rc.max() - rc.min():.2e}")
    print("\n✓ All checks passed.")
