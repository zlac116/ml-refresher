"""
PNL 4 — Finite-Difference Greeks + Full-Revaluation Attribution
================================================================

OBJECTIVE
    Two numerical-methods techniques used when analytical Greeks aren't
    available (Monte Carlo pricers, trees, exotics):

      1. Compute delta / gamma / vega / theta by BUMP-AND-REVALUE with
         appropriate finite-difference stencils. Validate against analytical.

      2. Run a FULL-REVALUATION attribution: reprice the trade under each
         isolated shock (dS only, dvol only, dt only) and under the FULL
         scenario. Cross-terms = full - sum(isolated).

ESTIMATED TIME
    20 min

TOPICS
    Central difference (2nd-order accurate):
        f'(x)  ≈ (f(x+h) - f(x-h)) / (2h)
        f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
    Choosing h: too small → cancellation noise; too large → truncation bias.
    Forward diff for theta (we can only go one direction in calendar time).
    Full-revaluation attribution captures cross-terms (vanna, volga) that
    a Taylor explain misses.

REAL-WORLD NOTE
    Production: bump sizes scaled by quoting convention (1 bp for rates,
    1 vol point for vols). Common-random-numbers (CRN) used when bumping
    Monte Carlo pricers to keep variance manageable.

REFERENCE
    Glasserman, "Monte Carlo Methods in Financial Engineering", ch. 7.
    Bouchaud-Potters, "Theory of Financial Risk Management", ch. 6.

EXPECTED OUTPUT  (S=100, K=100, T=1, r=5%, sigma=20%)
    delta FD/analyt:  0.636831 / 0.636831
    gamma FD/analyt:  0.018762 / 0.018762
    vega  FD/analyt: 37.524034 / 37.524035
    theta FD/analyt: -6.418199 / -6.414028
    V0:                10.450584
    V isolated dS:      1.310480
    V isolated dvol:    0.375704
    V isolated dt:     -0.025469
    V full shock:       1.652796
    sum isolated:       1.660715
    cross terms:       -0.007919

GRADING
    All asserts must pass. Numerical Greeks must match analytical to 1e-4.
"""
import numpy as np
from scipy.stats import norm


def _bsm_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _analytical_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        "delta": norm.cdf(d1),
        "gamma": norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        "vega":  S * norm.pdf(d1) * np.sqrt(T),
        "theta": -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2),
    }


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def fd_delta(price_fn, S, K, T, r, sigma, h: float = 0.01) -> float:
    """Central difference: (P(S+h) - P(S-h)) / (2h)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def fd_gamma(price_fn, S, K, T, r, sigma, h: float = 0.01) -> float:
    """Second-order central: (P(S+h) - 2*P(S) + P(S-h)) / h^2."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def fd_vega(price_fn, S, K, T, r, sigma, h: float = 1e-4) -> float:
    """Central diff in sigma."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def fd_theta(price_fn, S, K, T, r, sigma, h_days: float = 1.0) -> float:
    """Forward diff in calendar time:
        theta ≈ (P(T - h_days/252) - P(T)) / (h_days / 252)
    (i.e. how much value changes per year as calendar time advances).
    Negative for long calls (time decay).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def full_revaluation_attribution(price_fn, S, K, T, r, sigma,
                                 dS: float, dvol: float, dt: float) -> dict:
    """Run isolated-shock + full-shock revaluations and return a dict:
        V0             baseline price
        dS_only       V(S+dS, sigma, T)      - V0
        dvol_only     V(S,    sigma+dvol, T) - V0
        dt_only       V(S,    sigma, T-dt)   - V0
        full          V(S+dS, sigma+dvol, T-dt) - V0
        sum_isolated  dS_only + dvol_only + dt_only
        cross         full - sum_isolated
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    analyt = _analytical_greeks(S, K, T, r, sigma)

    d_fd = fd_delta(_bsm_call, S, K, T, r, sigma, h=0.01)
    g_fd = fd_gamma(_bsm_call, S, K, T, r, sigma, h=0.01)
    v_fd = fd_vega (_bsm_call, S, K, T, r, sigma, h=1e-4)
    t_fd = fd_theta(_bsm_call, S, K, T, r, sigma, h_days=1.0)

    assert abs(d_fd - analyt["delta"]) < 1e-6
    assert abs(g_fd - analyt["gamma"]) < 1e-4
    assert abs(v_fd - analyt["vega" ]) < 1e-4
    assert abs(t_fd - analyt["theta"]) < 1e-2, f"theta fd: {t_fd}"  # 1-day fwd diff

    attr = full_revaluation_attribution(_bsm_call, S, K, T, r, sigma,
                                        dS=2.0, dvol=0.01, dt=1/252)
    assert abs(attr["V0"]            -  10.450584) < 1e-5
    assert abs(attr["dS_only"]       -   1.310480) < 1e-4
    assert abs(attr["dvol_only"]     -   0.375704) < 1e-4
    assert abs(attr["dt_only"]       -  -0.025469) < 1e-4
    assert abs(attr["full"]          -   1.652796) < 1e-4
    assert abs(attr["sum_isolated"]  -   1.660715) < 1e-4
    assert abs(attr["cross"]         -  -0.007919) < 1e-4
    # full = isolated + cross  (by construction)
    assert abs(attr["full"] - (attr["sum_isolated"] + attr["cross"])) < 1e-12

    print(f"delta FD/analyt: {d_fd:.6f} / {analyt['delta']:.6f}")
    print(f"gamma FD/analyt: {g_fd:.6f} / {analyt['gamma']:.6f}")
    print(f"vega  FD/analyt: {v_fd:.6f} / {analyt['vega']:.6f}")
    print(f"theta FD/analyt: {t_fd:.6f} / {analyt['theta']:.6f}")
    print(f"V0:                {attr['V0']:.6f}")
    print(f"V isolated dS:      {attr['dS_only']:.6f}")
    print(f"V isolated dvol:    {attr['dvol_only']:.6f}")
    print(f"V isolated dt:     {attr['dt_only']:.6f}")
    print(f"V full shock:       {attr['full']:.6f}")
    print(f"sum isolated:       {attr['sum_isolated']:.6f}")
    print(f"cross terms:       {attr['cross']:.6f}")
    print("\n✓ All checks passed.")
