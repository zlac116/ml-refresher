"""
PNL 1 — Single-Trade Taylor Explain (4 Greeks)
==============================================

OBJECTIVE
    A trader is long 1 ATM European call. Overnight:
      - spot moves +2.00
      - implied vol moves +1 vol point (+0.01)
      - 1 trading day passes

    Decompose the change in option value into delta, gamma, vega, theta
    contributions, sum to a Taylor estimate, and compare to the actual
    repriced change. Residual must be small.

ESTIMATED TIME
    20 min

TOPICS
    Taylor decomposition:
        dV ≈ delta*dS + 0.5*gamma*dS^2 + vega*dvol + theta*dt
    Each term is the slope (or curvature) at t=0 times the factor change.
    Residual sources: cross terms (vanna, volga), higher-order, model error.

REAL-WORLD NOTE
    Front-office explain reports show each Greek's contribution in $ and
    a "residual" / "unexplained" row. Anything > a few % of |actual PNL|
    triggers an attribution review.

REFERENCE
    Hull, ch. 19; Bouchaud & Potters ch. 6.

EXPECTED OUTPUT  (S0=100, K=100, T=1, r=5%, sigma0=20%, dS=+2, dvol=+0.01, dt=1/252)
    C0     =  10.450584
    C1     =  12.103379
    actual =  1.652796
    delta  =  1.273661
    gamma  =  0.037524
    vega   =  0.375240
    theta  = -0.025452
    sum    =  1.660973
    resid  = -8.18e-03

GRADING
    All asserts must pass. |resid| must be < 0.05.
"""
import numpy as np
from scipy.stats import norm


def _bsm_call_full(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return price, delta, gamma, vega, theta


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def explain_pnl(delta: float, gamma: float, vega: float, theta: float,
                dS: float, dvol: float, dt: float) -> dict:
    """Return a dict with keys 'delta', 'gamma', 'vega', 'theta', 'total':
        delta = delta * dS
        gamma = 0.5  * gamma * dS^2
        vega  = vega * dvol
        theta = theta * dt
        total = sum of the four
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def actual_repricing(S0, K, T0, r, sigma0, dS, dvol, dt) -> float:
    """Actual change in call value:
        V(S0+dS, K, T0-dt, r, sigma0+dvol) - V(S0, K, T0, r, sigma0)
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S0, K, T0, r, sigma0 = 100.0, 100.0, 1.0, 0.05, 0.20
    dS, dvol, dt = 2.0, 0.01, 1 / 252

    C0, d, g, v, t = _bsm_call_full(S0, K, T0, r, sigma0)
    assert abs(C0 - 10.450584) < 1e-5

    explain = explain_pnl(d, g, v, t, dS, dvol, dt)
    assert abs(explain["delta"] -  1.273661) < 1e-5
    assert abs(explain["gamma"] -  0.037524) < 1e-5
    assert abs(explain["vega" ] -  0.375240) < 1e-5
    assert abs(explain["theta"] - -0.025452) < 1e-5
    assert abs(explain["total"] -  1.660973) < 1e-5

    actual = actual_repricing(S0, K, T0, r, sigma0, dS, dvol, dt)
    assert abs(actual - 1.652796) < 1e-5

    residual = actual - explain["total"]
    assert abs(residual) < 0.05

    C1 = C0 + actual
    print(f"C0     =  {C0:.6f}")
    print(f"C1     =  {C1:.6f}")
    print(f"actual =  {actual:.6f}")
    print(f"delta  =  {explain['delta']:.6f}")
    print(f"gamma  =  {explain['gamma']:.6f}")
    print(f"vega   =  {explain['vega']:.6f}")
    print(f"theta  = {explain['theta']:.6f}")
    print(f"sum    =  {explain['total']:.6f}")
    print(f"resid  = {residual:.2e}")
    print("\n✓ All checks passed.")
