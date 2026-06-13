"""
OPT 3 — Implied Volatility via Newton-Raphson
=============================================

OBJECTIVE
    Given a market call price, solve C_BSM(sigma) = C_mkt for sigma using
    Newton-Raphson with analytical vega.

ESTIMATED TIME
    20 min

TOPICS
    Newton-Raphson on a 1-D nonlinear equation
    Why vega is the right derivative
    Initial guess heuristic: sigma0 = sqrt(2*pi/T) * C/S (Brenner-Subrahmanyam)
    Convergence: |f(sigma)| < tol OR max_iter exceeded

REFERENCE
    Hull, ch. 19; Brenner-Subrahmanyam (1988) initial guess.

EXPECTED OUTPUT  (S=100, K=100, T=1, r=5%, C_mkt=10.45)
    initial guess:       0.250663
    converged iv:        0.199984
    iterations used:     4
    repriced @ iv:       10.450000

GRADING
    All asserts must pass.
"""
import numpy as np
from scipy.stats import norm


def _bsm_call_and_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    return price, vega


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def brenner_subrahmanyam_iv(C: float, S: float, T: float) -> float:
    """Closed-form ATM IV approximation: sigma ≈ sqrt(2*pi/T) * C/S.

    Use as the Newton initial guess.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def implied_vol_newton(C_mkt: float, S: float, K: float, T: float, r: float,
                       tol: float = 1e-8, max_iter: int = 100) -> tuple[float, int]:
    """Newton-Raphson on f(sigma) = bsm_call(sigma) - C_mkt.

    Returns (iv, iters_used). Use the Brenner-Subrahmanyam initial guess.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    C_mkt = 10.45

    sigma0 = brenner_subrahmanyam_iv(C_mkt, S, T)
    assert abs(sigma0 - 0.250663) < 1e-4, f"initial guess off: {sigma0}"

    iv, iters = implied_vol_newton(C_mkt, S, K, T, r)
    assert abs(iv - 0.199984) < 1e-5, f"iv off: {iv}"
    assert 2 <= iters <= 10, f"unreasonable iter count: {iters}"

    # Reprice should match the input to tol
    reprice, _ = _bsm_call_and_vega(S, K, T, r, iv)
    assert abs(reprice - C_mkt) < 1e-7

    print(f"initial guess:       {sigma0:.6f}")
    print(f"converged iv:        {iv:.6f}")
    print(f"iterations used:     {iters}")
    print(f"repriced @ iv:       {reprice:.6f}")
    print("\n✓ All checks passed.")
