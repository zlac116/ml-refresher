"""
FI 12 — Forward-Starting Swap (2y x 3y)
=========================================

OBJECTIVE
    Price a swap that STARTS in 2 years and runs for 3 years (years 3, 4, 5
    of the curve). Then back out its par forward swap rate from the curve.

      1. Float leg PV   = N * (D(T_start) - D(T_end))
                        = N * (D(2y) - D(5y))
      2. Forward annuity = sum of D over the forward window only
                         = D(3) + D(4) + D(5)         (with annual deltas=1)
      3. Par forward swap rate:  c_fwd = float_leg / annuity_fwd

ESTIMATED TIME
    20 min

TOPICS
    Forward swap = swap whose first payment is at a future date, not at t+δ
    Float leg PV generalises:  N * (D(T_start) - D(T_end))
    Annuity is now over the FORWARD window only (excludes years 1, 2)
    Forward swap rates are the standard underlying of SWAPTIONS

REFERENCE
    Hull, ch. 7 + 31; Brigo-Mercurio "Interest Rate Models" ch. 6.

EXPECTED OUTPUT  (N=$100M, same curve as FI 10, forward window 2y -> 5y)
    D(2y)                = 0.90349448
    D(5y)                = 0.76425867
    A forward (3y over 3-5) = 2.43033050
    PV float leg         = $13,923,580.40
    Par forward swap rate (2y x 3y) = 0.05729089   (5.729%, > 5y spot of 5.5%)
    PV fixed at par      = $13,923,580.40   (equals PV float — net zero by def)

REAL-WORLD NOTE
    The forward swap rate (5.73%) is ABOVE the spot 5y rate (5.50%) because
    the curve is upward-sloping. Locking in 5y now starting 2y forward costs
    you more rate than starting today.
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def forward_annuity(forward_deltas: np.ndarray,
                    forward_discount_factors: np.ndarray) -> float:
    """Annuity over the FORWARD window only.

    A_fwd = sum_i delta_i * D(0, T_i)   where the sum is over only the
    forward-window payment dates.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def forward_float_leg_pv(notional: float, D_start: float, D_end: float) -> float:
    """Generalised floating-leg identity:

        PV_float = N * (D(0, T_start) - D(0, T_end))

    Reduces to N*(1 - D(T_end)) for a spot-starting swap (where T_start = 0,
    D_start = 1).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def par_forward_swap_rate(D_start: float, D_end: float,
                          forward_deltas: np.ndarray,
                          forward_discount_factors: np.ndarray) -> float:
    """Par forward swap rate:

                  D(0, T_start) - D(0, T_end)
        c_fwd = ─────────────────────────────
                       A_fwd(T_n)

    Sets PV of forward swap = 0.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Same curve as FI 10 / 11
    par_rates = np.array([0.050, 0.052, 0.053, 0.054, 0.055])
    D = np.empty(5); s = 0.0
    for k, c in enumerate(par_rates):
        D[k] = (1 - c * s) / (1 + c); s += D[k]
    # D = [D(1y), D(2y), D(3y), D(4y), D(5y)]

    # Forward window: starts at T_start = 2y, ends at T_end = 5y
    # Payment dates inside the window: years 3, 4, 5
    D_start = D[1]                     # D(2y)
    D_end   = D[4]                     # D(5y)
    fwd_D   = D[2:5]                   # D(3y), D(4y), D(5y)
    fwd_deltas = np.ones(3)

    N = 100_000_000.0

    A_fwd = forward_annuity(fwd_deltas, fwd_D)
    assert abs(A_fwd - 2.43033050) < 1e-6, A_fwd

    PV_float = forward_float_leg_pv(N, D_start, D_end)
    assert abs(PV_float - 13_923_580.40) < 1.0, PV_float

    c_fwd = par_forward_swap_rate(D_start, D_end, fwd_deltas, fwd_D)
    assert abs(c_fwd - 0.05729089) < 1e-6, c_fwd
    # Forward swap rate must be ABOVE spot 5y par (5.5%) on upward-sloping curve
    assert c_fwd > 0.055

    # Sanity: PV of fixed leg at par_fwd = PV_float by construction
    PV_fix_at_par = N * c_fwd * A_fwd
    assert abs(PV_fix_at_par - PV_float) < 1e-3

    print(f"D(2y)                = {D_start:.8f}")
    print(f"D(5y)                = {D_end:.8f}")
    print(f"A forward (3y over 3-5) = {A_fwd:.8f}")
    print(f"PV float leg         = ${PV_float:,.2f}")
    print(f"Par forward swap rate (2y x 3y) = {c_fwd:.8f}   ({c_fwd*100:.3f}%, > 5y spot of 5.5%)")
    print(f"PV fixed at par      = ${PV_fix_at_par:,.2f}   (equals PV float — net zero by def)")
    print("\n✓ All checks passed.")
