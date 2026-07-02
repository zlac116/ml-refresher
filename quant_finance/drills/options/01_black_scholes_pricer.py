"""
OPT 1 — Black-Scholes Pricer + Greeks
=====================================

OBJECTIVE
    Price European call/put under Black-Scholes and return the four common
    Greeks (delta, gamma, vega, theta) — using scipy.stats.norm.

ESTIMATED TIME
    20 min

TOPICS
    BSM formula (no dividends)
    Greeks: dC/dS, d2C/dS2, dC/dsigma, dC/dt (calendar-time convention)
    scipy.stats.norm.cdf / .pdf

REFERENCE
    Hull, Options Futures and Other Derivatives, ch. 15 + 19.

DEFINITIONS USED
    d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    Call  = S*N(d1) - K*exp(-rT)*N(d2)
    Put   = K*exp(-rT)*N(-d2) - S*N(-d1)
    delta_call = N(d1);   delta_put = N(d1) - 1
    gamma      = phi(d1) / (S * sigma * sqrt(T))
    vega       = S * phi(d1) * sqrt(T)
    theta_call = -S*phi(d1)*sigma/(2*sqrt(T)) - r*K*exp(-rT)*N(d2)
    theta_put  = -S*phi(d1)*sigma/(2*sqrt(T)) + r*K*exp(-rT)*N(-d2)

EXPECTED OUTPUT  (S=100, K=100, T=1, r=5%, sigma=20%)
    call price   = 10.450584
    put price    = 5.573526
    delta call   = 0.636831
    delta put    = -0.363169
    gamma        = 0.018762
    vega         = 37.524035
    theta call   = -6.414028
    theta put    = -1.657880

GRADING
    All asserts must pass.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass
class BSMGreeks:
    price: float
    delta: float
    gamma: float
    vega:  float
    theta: float


def _d1_d2(S: float, K: float, T: float, r: float, sigma:float) -> tuple[float, float]:
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return d1, d1 - sigma*np.sqrt(T)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def bsm_call(S: float, K: float, T: float, r: float, sigma: float) -> BSMGreeks:
    """European call under BSM, no dividends. Return BSMGreeks."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    price = S*nd1 - K*np.exp(-r*T)*nd2
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S*norm.pdf(d1)*sigma) / (2*np.sqrt(T)) - r*K*np.exp(-r*T)*nd2
    return BSMGreeks(
        price=price, delta=nd1, gamma=gamma, vega=vega, theta=theta
    )



# ── TASK 2 ─────────────────────────────────────────────────────────────────
def bsm_put(S: float, K: float, T: float, r: float, sigma: float) -> BSMGreeks:
    """European put under BSM, no dividends. Return BSMGreeks."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S*norm.pdf(d1)*sigma) / (2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    return BSMGreeks(
        price=price, delta=norm.cdf(d1) - 1, gamma=gamma, vega=vega, theta=theta
    )


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    c = bsm_call(S, K, T, r, sigma)
    p = bsm_put (S, K, T, r, sigma)

    assert abs(c.price -  10.450584) < 1e-5
    assert abs(p.price -   5.573526) < 1e-5
    assert abs(c.delta -   0.636831) < 1e-5
    assert abs(p.delta -  -0.363169) < 1e-5
    assert abs(c.gamma -   0.018762) < 1e-5
    assert abs(c.vega  -  37.524035) < 1e-4
    assert abs(c.theta -  -6.414028) < 1e-4
    assert abs(p.theta -  -1.657880) < 1e-4

    # Greeks symmetry: gamma_call == gamma_put, vega_call == vega_put
    assert abs(c.gamma - p.gamma) < 1e-12
    assert abs(c.vega  - p.vega ) < 1e-12

    print(f"call price   = {c.price:.6f}")
    print(f"put price    = {p.price:.6f}")
    print(f"delta call   = {c.delta:.6f}")
    print(f"delta put    = {p.delta:.6f}")
    print(f"gamma        = {c.gamma:.6f}")
    print(f"vega         = {c.vega:.6f}")
    print(f"theta call   = {c.theta:.6f}")
    print(f"theta put    = {p.theta:.6f}")
    print("\n✓ Case 1 (T=1) checks passed.")

    # ── Case 2: non-unit T pins down the vega ×/÷ ambiguity ──────────────
    # At T=1, the buggy `S·φ/√T` and canonical `S·φ·√T` coincide (both = S·φ).
    # At T=4 they differ by 4×: canonical vega = 62.45, buggy = 15.61. A
    # single test point can't distinguish two functions that happen to agree
    # there — this second case pins the vega formula down. It also catches
    # the theta `σ/(2√T)` term and the `√T` factor in d₁/d₂ scaling.
    S, K, T, r, sigma = 100.0, 100.0, 4.0, 0.05, 0.20
    c2 = bsm_call(S, K, T, r, sigma)
    p2 = bsm_put (S, K, T, r, sigma)

    assert abs(c2.price -  25.213326) < 1e-5, c2.price
    assert abs(p2.price -   7.086402) < 1e-5, p2.price
    assert abs(c2.delta -   0.758036) < 1e-5, c2.delta
    assert abs(p2.delta -  -0.241964) < 1e-5, p2.delta
    assert abs(c2.gamma -   0.007806) < 1e-5, c2.gamma
    assert abs(c2.vega  -  62.450787) < 1e-4, c2.vega     # flags S·φ/√T bug
    assert abs(c2.theta -  -4.090785) < 1e-4, c2.theta
    assert abs(p2.theta -   0.002869) < 1e-4, p2.theta

    # Symmetry (same for T=1 case)
    assert abs(c2.gamma - p2.gamma) < 1e-12
    assert abs(c2.vega  - p2.vega ) < 1e-12

    print(f"[T=4] call={c2.price:.6f} put={p2.price:.6f} "
          f"vega={c2.vega:.6f} theta_c={c2.theta:.6f}")
    print("\n✓ Case 2 (T=4) checks passed.")
