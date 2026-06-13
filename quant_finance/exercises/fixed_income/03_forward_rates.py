"""
FI 3 — Forward Rates from a Discount Curve
==========================================

OBJECTIVE
    Given continuously-compounded zero rates at observation tenors, compute:
      1. Discount factors D(0, T) = exp(-r(T) * T).
      2. The continuously-compounded forward rate between any two tenors:
            f(T1, T2) = (r2*T2 - r1*T1) / (T2 - T1)
            equivalently  f(T1, T2) = -ln(D(T2)/D(T1)) / (T2-T1)

ESTIMATED TIME
    15 min

TOPICS
    Continuous compounding identity
    Forward = forward-starting zero between two tenors
    No-arbitrage equivalence between the two formulas

REFERENCE
    Hull, ch. 4; Andersen-Piterbarg vol I, ch. 4.4.

EXPECTED OUTPUT  (zeros at 1y, 2y, 3y, 5y, 10y = 3%, 3.3%, 3.5%, 3.7%, 4%)
    D(0,1)   = 0.970446
    D(0,5)   = 0.831104
    f(1,2)   = 0.036000
    f(2,3)   = 0.039000
    f(5,10)  = 0.043000

GRADING
    All asserts must pass and the two forward-rate formulas must agree.
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def discount_factors_from_zeros(zero_T: np.ndarray, zero_r: np.ndarray) -> np.ndarray:
    """D(0, T) = exp(-r(T) * T). Vectorise across tenors."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def forward_rate_zero_form(r1: float, T1: float, r2: float, T2: float) -> float:
    """f(T1, T2) = (r2 * T2 - r1 * T1) / (T2 - T1)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def forward_rate_discount_form(D1: float, T1: float, D2: float, T2: float) -> float:
    """f(T1, T2) = -ln(D2/D1) / (T2 - T1).  Must match zero-form for any curve."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    zero_T = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    zero_r = np.array([0.030, 0.033, 0.035, 0.037, 0.040])

    D = discount_factors_from_zeros(zero_T, zero_r)
    assert D.shape == (5,)
    assert abs(D[0] - 0.970446) < 1e-5
    assert abs(D[3] - 0.831104) < 1e-5

    f12  = forward_rate_zero_form(zero_r[0], zero_T[0], zero_r[1], zero_T[1])
    f23  = forward_rate_zero_form(zero_r[1], zero_T[1], zero_r[2], zero_T[2])
    f510 = forward_rate_zero_form(zero_r[3], zero_T[3], zero_r[4], zero_T[4])
    assert abs(f12  - 0.036) < 1e-9, f12
    assert abs(f23  - 0.039) < 1e-9
    assert abs(f510 - 0.043) < 1e-9

    # Two formulas must agree on the same inputs
    f12_d = forward_rate_discount_form(D[0], zero_T[0], D[1], zero_T[1])
    assert abs(f12_d - f12) < 1e-12, (f12_d, f12)

    print(f"D(0,1)   = {D[0]:.6f}")
    print(f"D(0,5)   = {D[3]:.6f}")
    print(f"f(1,2)   = {f12:.6f}")
    print(f"f(2,3)   = {f23:.6f}")
    print(f"f(5,10)  = {f510:.6f}")
    print("\n✓ All checks passed.")
