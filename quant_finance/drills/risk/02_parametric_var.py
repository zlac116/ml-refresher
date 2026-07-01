"""
RISK 2 — Parametric VaR from a Covariance Matrix
================================================

OBJECTIVE
    For a 3-asset portfolio with weights, asset annualised vols and
    correlation matrix:
      1. Compute the portfolio daily vol.
      2. Compute 99% parametric VaR (normal, no mean adjustment).
      3. Compute COMPONENT VaR — each asset's marginal contribution.
         Sum of component VaR must equal total VaR.

ESTIMATED TIME
    20 min

TOPICS
    Portfolio vol = sqrt(w' * Sigma * w)
    Parametric VaR = z_alpha * sigma_port   (with mean=0 assumption)
    Marginal VaR_i = (Sigma * w)_i / sigma_port
    Component VaR_i = w_i * MVaR_i * z_alpha
    sum_i Component VaR_i = total VaR  (Euler decomposition)

REAL-WORLD NOTE
    Parametric VaR underestimates tails for fat-tailed assets (equities,
    crypto). Historical or Monte Carlo VaR with full revaluation is the
    Basel-compliant standard for trading books (FRTB ES).

REFERENCE
    Jorion, "Value at Risk" ch. 7; RiskMetrics technical doc.

EXPECTED OUTPUT  (w=[0.4,0.4,0.2], vol_ann=[20%,15%,30%], corr 3x3, conf=99%)
    portfolio daily vol = 0.008656
    99% parametric VaR  = 0.020136
    $ VaR99 at $5M      = 100680.51
    sum of components   = 0.020136   (= total VaR)

GRADING
    All asserts must pass. Sum of component VaR must equal total VaR exactly.
"""
import numpy as np
from scipy.stats import norm


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def covariance_matrix(daily_vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Sigma_ij = sigma_i * sigma_j * rho_ij.
    Vectorised: np.outer(vols, vols) * corr.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    """sqrt(w' Σ w)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def parametric_var(weights: np.ndarray, cov: np.ndarray, conf: float) -> float:
    """z_conf * portfolio_vol(weights, cov)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def component_var(weights: np.ndarray, cov: np.ndarray, conf: float) -> np.ndarray:
    """Component VaR: w_i * (Sigma @ w)_i / sigma_port * z_conf.
    Sum of components must equal parametric_var.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    w = np.array([0.4, 0.4, 0.2])
    ann_vol = np.array([0.20, 0.15, 0.30])
    daily_vol = ann_vol / np.sqrt(252)
    corr = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])

    cov = covariance_matrix(daily_vol, corr)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)               # symmetric

    pv = portfolio_vol(w, cov)
    assert abs(pv - 0.008656) < 1e-5

    var = parametric_var(w, cov, conf=0.99)
    assert abs(var - 0.020136) < 1e-5

    cv = component_var(w, cov, conf=0.99)
    assert cv.shape == (3,)
    # Sum of components = total VaR (Euler decomposition)
    assert abs(cv.sum() - var) < 1e-12

    print(f"portfolio daily vol = {pv:.6f}")
    print(f"99% parametric VaR  = {var:.6f}")
    print(f"$ VaR99 at $5M      = {5_000_000 * var:.2f}")
    print(f"sum of components   = {cv.sum():.6f}   (= total VaR)")
    print("\n✓ All checks passed.")
