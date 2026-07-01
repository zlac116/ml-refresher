"""
FI 10 — Annuity Factor + Par Swap Rate Identity
==================================================

OBJECTIVE
    Two of the most-used building blocks in fixed income, both falling out of
    the par swap NPV identity:

      1. ANNUITY FACTOR:  A(T_n) = sum_{i=1..n} δ_i * D(0, T_i)
         The PV of $1 of "rate" paid every period over the swap's tenor.
         Appears in: swap pricing, swaption Black-76, swap DV01.

      2. PAR SWAP RATE FROM A DISCOUNT CURVE:
                       1 - D(0, T_n)
            c(T_n) = ──────────────────
                          A(T_n)
         Verify: the bootstrapped curve from FI 8 should reprice the input
         par rates exactly (as a self-consistency check).

ESTIMATED TIME
    15 min

TOPICS
    Par swap NPV identity:  c * A(T_n) + D(0, T_n) - 1 = 0
    Annuity = "DV01-like" measure of how much $1/yr of coupon is worth in PV
    Identity inversion: c = (1 - D)/A   (par rate of fresh swap)

REFERENCE
    Hull, ch. 7; Andersen-Piterbarg vol I, ch. 5 + 16.

EXPECTED OUTPUT  (5y annual swap curve, deltas=1)
    D bootstrap = [0.95238095, 0.90349448, 0.85625698, 0.80981485, 0.76425867]

    A(5y)          = 4.28620593
    par rate (5y)  = 0.05500000   (matches input par rate of 5.5%)

GRADING
    All asserts must pass; the implied par rate MUST equal the input par rate.
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def annuity_factor(deltas: np.ndarray, discount_factors: np.ndarray) -> float:
    """Annuity = sum_i delta_i * D(0, T_i).

    Both inputs are vectors of length n. Returns a scalar (the PV of $1/period
    of coupon over the swap's tenor).
    """
    return np.sum(deltas * discount_factors)


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def par_swap_rate(discount_factors: np.ndarray, deltas: np.ndarray) -> float:
    """Par swap rate from a discount curve:

        c = (1 - D(0, T_n)) / A(T_n)

    where A = sum_i delta_i * D(0, T_i).
    """
    annuity = annuity_factor(deltas, discount_factors)
    pv_float = (1 - discount_factors[-1])
    return pv_float / annuity


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def verify_bootstrap_consistency(par_rates: np.ndarray, deltas: np.ndarray,
                                 discount_factors: np.ndarray,
                                 atol: float = 1e-10) -> bool:
    """For each tenor n, the implied par rate from the curve  c_implied =
    (1 - D[n-1]) / A_n  should equal the input par rate par_rates[n-1].

    Returns True if all implied par rates match the input to atol.
    """
    n = len(par_rates)
    error = np.empty(n)
    for i in range(n):
        error[i] = np.abs(par_rates[i] - par_swap_rate(discount_factors[:i+1], deltas[:i+1]))
    
    return all(error < atol)


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Reuse the bootstrapped 5y curve from FI 9 (par rates linearly interp)
    deltas    = np.ones(5)
    par_rates = np.array([0.050, 0.052, 0.053, 0.054, 0.055])
    # Bootstrap recursion (same as FI 8)
    D = np.empty(5)
    s = 0.0
    for k, c in enumerate(par_rates):
        D[k] = (1 - c * s) / (1 + c)
        s += D[k]
    expected_D = np.array([0.95238095, 0.90349448, 0.85625698, 0.80981485, 0.76425867])
    assert np.allclose(D, expected_D, atol=1e-6)

    A_5y = annuity_factor(deltas, D)
    assert abs(A_5y - 4.28620593) < 1e-6, A_5y

    c_5y = par_swap_rate(D, deltas)
    assert abs(c_5y - 0.055) < 1e-8     # MUST equal the input 5y par rate
    
    # Self-consistency check at every tenor
    ok = verify_bootstrap_consistency(par_rates, deltas, D)
    assert ok is True

    # Also: shorter sub-curves give the right par rates
    c_2y = par_swap_rate(D[:2], deltas[:2])
    assert abs(c_2y - 0.052) < 1e-10
    c_3y = par_swap_rate(D[:3], deltas[:3])
    assert abs(c_3y - 0.053) < 1e-10

    print(f"D bootstrap = {D}")
    print()
    print(f"A(5y)          = {A_5y:.8f}")
    print(f"par rate (5y)  = {c_5y:.8f}   (matches input par rate of 5.5%)")
    print()
    print(f"par rate (2y)  = {c_2y:.8f}")
    print(f"par rate (3y)  = {c_3y:.8f}")
    print("\n✓ All checks passed.")
