"""
OPT 2 — Put-Call Parity + Arbitrage Detection
=============================================

OBJECTIVE
    Verify European parity C - P = S - K*exp(-rT) (no dividends, no carry),
    and detect arbitrage when a market quoted C, P, S, K pair violates it.

ESTIMATED TIME
    15 min

TOPICS
    Parity identity (lower bound for a European call)
    Arbitrage profit = |C - P - (S - K*e^(-rT))|, financed at r

REFERENCE
    Hull, ch. 11 (parity); Cox-Ross-Rubinstein (1979).

EXPECTED OUTPUT  (S=100, K=100, T=1, r=5%, sigma=20%)
    bsm call:        10.450584
    bsm put:          5.573526
    parity LHS:       4.877058
    parity RHS:       4.877058
    parity gap:       0.00e+00
    mispriced gap:    0.622942
    arbitrage exists: True

GRADING
    All asserts must pass.
"""
import numpy as np
from scipy.stats import norm


def _bsm_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bsm_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def parity_gap(C: float, P: float, S: float, K: float, T: float, r: float) -> float:
    """Return (C - P) - (S - K*exp(-rT)). Zero when parity holds exactly."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def is_arbitrage(C: float, P: float, S: float, K: float, T: float, r: float,
                 tol: float = 1e-4) -> bool:
    """Return True if |parity gap| > tol (an actionable mispricing)."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    c_bsm = _bsm_call(S, K, T, r, sigma)
    p_bsm = _bsm_put (S, K, T, r, sigma)

    gap_ok = parity_gap(c_bsm, p_bsm, S, K, T, r)
    assert abs(gap_ok) < 1e-10, f"BSM should obey parity exactly: gap={gap_ok}"
    assert not is_arbitrage(c_bsm, p_bsm, S, K, T, r)

    # Now construct a mispricing: market call is too high by 0.623
    c_mkt = c_bsm + 0.622942
    p_mkt = p_bsm
    gap_bad = parity_gap(c_mkt, p_mkt, S, K, T, r)
    assert abs(gap_bad - 0.622942) < 1e-6
    assert is_arbitrage(c_mkt, p_mkt, S, K, T, r)

    print(f"bsm call:        {c_bsm:.6f}")
    print(f"bsm put:         {p_bsm:.6f}")
    print(f"parity LHS:      {c_bsm - p_bsm:.6f}")
    print(f"parity RHS:      {S - K * np.exp(-r * T):.6f}")
    print(f"parity gap:      {abs(gap_ok):.2e}")
    print(f"mispriced gap:   {gap_bad:.6f}")
    print(f"arbitrage exists: {is_arbitrage(c_mkt, p_mkt, S, K, T, r)}")
    print("\n✓ All checks passed.")
