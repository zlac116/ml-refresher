"""
FI 11 — Interest Rate Swap Pricing: Fixed + Floating Legs + DV01
==================================================================

OBJECTIVE
    Price a 5y annual-pay IRS where the trader RECEIVES a contracted fixed
    rate of 5.00% and PAYS floating (single-curve world).

      1. PV the fixed leg:      PV_fix   = N * c * A(T_n)
      2. PV the floating leg:   PV_float = N * (1 - D(0, T_n))
         (this is the "telescoping" identity in single-curve world)
      3. PV of the swap:        PV_swap  = PV_fix - PV_float
                                (receive-fixed perspective)
      4. DV01 of the swap:      bump all zero rates +1 bp, reprice

ESTIMATED TIME
    20 min

TOPICS
    Fixed leg = N · c · A    where A = annuity factor (from FI 10)
    Floating leg = N · (1 - D(0, T_n))    — derivation in 03_curve_building.md
    Receive-fixed swap is a long-duration position; loses when rates rise

REFERENCE
    Hull, ch. 7; Andersen-Piterbarg vol I, ch. 5.

EXPECTED OUTPUT  (N=$100M, c=5%, 5y annual swap, same curve as FI 10)
    Annuity A(5y)               = 4.28620593
    PV fixed leg                = $21,431,029.67
    PV float leg                = $23,574,132.64
    PV swap (rcv-fix)           = -$2,143,102.97   (we'd PAY $2.14M today)
    Par rate from curve         = 0.05500000      (5.50%, > our 5.00%)
    DV01 of swap (rcv-fix)      = $44,396.62      (loss per +1 bp)

REAL-WORLD NOTE
    A 5y receive-fixed swap of $100M has DV01 ≈ N · A · 0.0001 ≈ $42.9k
    using just the annuity. The actual DV01 includes the small float-leg
    sensitivity, so we compute via reprice.
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def pv_fixed_leg(notional: float, fixed_rate: float, deltas: np.ndarray,
                 discount_factors: np.ndarray) -> float:
    """PV of fixed leg = N * c * sum(delta_i * D(T_i))
                      = N * c * A(T_n).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def pv_floating_leg(notional: float, discount_factors: np.ndarray) -> float:
    """PV of floating leg in single-curve world:

        PV_float = N * (1 - D(0, T_n))

    where D(0, T_n) is the discount factor at the FINAL payment date.
    (Telescoping identity — see 03_curve_building.md for derivation.)
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def pv_swap_receive_fixed(notional: float, fixed_rate: float,
                          deltas: np.ndarray, discount_factors: np.ndarray) -> float:
    """PV from the receive-fixed perspective = PV_fix - PV_float."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def swap_dv01(notional: float, fixed_rate: float,
              deltas: np.ndarray, zero_rates: np.ndarray,
              tenors: np.ndarray, bump: float = 0.0001) -> float:
    """Compute DV01 of a receive-fixed swap by repricing under a parallel
    +1 bp shift of the zero rates.

        DV01 = PV_swap(zeros) - PV_swap(zeros + bump)

    Positive DV01 = receive-fixed swap LOSES when rates rise.

    Use: D = exp(-z * T) to convert zero rates to discount factors.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Same curve as FI 10
    deltas    = np.ones(5)
    par_rates = np.array([0.050, 0.052, 0.053, 0.054, 0.055])
    D = np.empty(5); s = 0.0
    for k, c in enumerate(par_rates):
        D[k] = (1 - c * s) / (1 + c); s += D[k]

    N        = 100_000_000.0
    c_swap   = 0.05               # we receive fixed 5%
    tenors   = np.arange(1, 6, dtype=float)
    zeros    = -np.log(D) / tenors

    A = float(np.sum(deltas * D))
    assert abs(A - 4.28620593) < 1e-6

    PV_fix   = pv_fixed_leg   (N, c_swap, deltas, D)
    PV_float = pv_floating_leg(N, D)
    PV_swap  = pv_swap_receive_fixed(N, c_swap, deltas, D)

    assert abs(PV_fix   - 21_431_029.67) < 1.0,  PV_fix
    assert abs(PV_float - 23_574_132.64) < 1.0,  PV_float
    assert abs(PV_swap  - -2_143_102.97) < 1.0,  PV_swap
    assert abs(PV_fix - PV_float - PV_swap) < 1e-6  # additive identity

    # Par rate from curve should equal the curve's input 5y par rate (5.5%)
    par_from_curve = (1 - D[-1]) / A
    assert abs(par_from_curve - 0.055) < 1e-8

    # DV01: bump zeros +1 bp, reprice
    DV01 = swap_dv01(N, c_swap, deltas, zeros, tenors, bump=0.0001)
    assert abs(DV01 - 44_396.62) < 1.0, DV01
    # Sanity: receive-fixed → positive DV01 (loses on rate rises)
    assert DV01 > 0
    # Should be roughly annuity * N * bump
    annuity_approx = N * A * 0.0001
    assert 0.7 * annuity_approx < DV01 < 1.3 * annuity_approx

    print(f"Annuity A(5y)               = {A:.8f}")
    print(f"PV fixed leg                = ${PV_fix:,.2f}")
    print(f"PV float leg                = ${PV_float:,.2f}")
    print(f"PV swap (rcv-fix)           = ${PV_swap:,.2f}   (we'd PAY ${abs(PV_swap)/1e6:.2f}M today)")
    print(f"Par rate from curve         = {par_from_curve:.8f}      ({par_from_curve*100:.2f}%, > our {c_swap*100:.2f}%)")
    print(f"DV01 of swap (rcv-fix)      = ${DV01:,.2f}      (loss per +1 bp)")
    print("\n✓ All checks passed.")
