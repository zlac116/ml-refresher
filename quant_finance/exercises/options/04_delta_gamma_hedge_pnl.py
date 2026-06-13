"""
OPT 4 — Delta-Gamma-Theta P&L Explain
=====================================

OBJECTIVE
    A trader sells a 1y ATM European call. Spot moves +2.0 overnight and one
    trading day passes. Decompose the change in option value into delta,
    gamma, and theta contributions and compare to the actual revaluation.

ESTIMATED TIME
    20 min

TOPICS
    Taylor expansion of option value:
        dV ≈ delta*dS + 0.5 * gamma * dS^2 + theta * dt
    Residual = actual repriced dV - explain
    Sign conventions: theta uses dt = +1/252 (calendar time forward)

REFERENCE
    Hull, ch. 19 (Greeks); Bouchaud-Potters "Theory of Financial Risk Mgmt", ch. 6.

EXPECTED OUTPUT  (S0=100, K=100, T0=1, r=5%, sigma=20%, dS=+2, dt=1/252)
    C0  =  10.450584
    C1  =  11.735155
    actual dV   =  1.284572
    delta explain =  1.273661
    gamma explain =  0.037524
    theta explain = -0.025452
    sum  explain =  1.285733
    residual     = -1.16e-03

GRADING
    All asserts must pass. Residual must be small (< 0.01).
"""
import numpy as np
from scipy.stats import norm


def _bsm_call_full(S, K, T, r, sigma):
    """Returns (price, delta, gamma, theta)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return price, delta, gamma, theta


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def taylor_explain(delta: float, gamma: float, theta: float,
                   dS: float, dt: float) -> dict:
    """Return a dict with keys 'delta', 'gamma', 'theta', 'total' where
        delta = delta * dS
        gamma = 0.5 * gamma * dS^2
        theta = theta * dt           # dt in years
        total = sum of the three
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def repricing_pnl(S0: float, K: float, T0: float, r: float, sigma: float,
                  dS: float, dt: float) -> float:
    """Actual change in option value:
        C(S0+dS, K, T0-dt, r, sigma) - C(S0, K, T0, r, sigma)
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S0, K, T0, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    dS = 2.0
    dt = 1 / 252

    C0, delta, gamma, theta = _bsm_call_full(S0, K, T0, r, sigma)

    explain = taylor_explain(delta, gamma, theta, dS, dt)
    assert abs(explain["delta"] -  1.273661) < 1e-5
    assert abs(explain["gamma"] -  0.037524) < 1e-5
    assert abs(explain["theta"] - -0.025452) < 1e-5
    assert abs(explain["total"] -  1.285733) < 1e-5

    actual = repricing_pnl(S0, K, T0, r, sigma, dS, dt)
    assert abs(actual - 1.284572) < 1e-5

    residual = actual - explain["total"]
    assert abs(residual) < 1e-2, f"unexplained too large: {residual}"

    C1 = C0 + actual
    print(f"C0  =  {C0:.6f}")
    print(f"C1  =  {C1:.6f}")
    print(f"actual dV   =  {actual:.6f}")
    print(f"delta explain =  {explain['delta']:.6f}")
    print(f"gamma explain =  {explain['gamma']:.6f}")
    print(f"theta explain = {explain['theta']:.6f}")
    print(f"sum  explain =  {explain['total']:.6f}")
    print(f"residual     = {residual:.2e}")
    print("\n✓ All checks passed.")
