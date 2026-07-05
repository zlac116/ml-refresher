"""
FI 9 — Multi-Instrument Curve Bootstrap + Interpolation
=========================================================

OBJECTIVE
    Build a discount curve from a REALISTIC instrument set:
      - Short end:  3M and 6M ACT/360 deposits → D(3M), D(6M)
      - Long end:   1y, 2y, 5y par swap rates → D(1y), D(2y), D(5y) via bootstrap

    Then evaluate the discount factor at an OFF-PILLAR point (T = 3.5y) using
    TWO interpolation methods and compare:
      - log-linear in D       — most common for production rate curves
      - linear in zero rate   — common pedagogical choice; differs from log-linear

ESTIMATED TIME
    25 min

TOPICS
    Mixed front-end (simple-comp deposits) + back-end (compound swaps)
    D(deposit) = 1 / (1 + r * d/360)
    Par-swap bootstrap (same recursion as FI 8, with annual deltas = 1)
    Linear par-rate interpolation onto missing tenors
    Two flavours of curve interpolation:
        log-lin in D:    ln(D(T)) interpolated linearly between pillars
        lin in zero:     z(T) interpolated linearly, then D(T) = exp(-z*T)

REFERENCE
    Andersen-Piterbarg vol I, ch. 5; Hagan-West (2006) for monotone-convex
    interpolation (production standard).

EXPECTED OUTPUT
    D(3M)          = 0.98875294
    D(6M)          = 0.97666583
    D(1y)          = 0.95238095
    D(2y)          = 0.90349448
    D(3y)          = 0.85625698 (par rate interp at year 3 = 5.30%)
    D(5y)          = 0.76425867

    Interp at T=3.5y:
      log-linear in D = 0.83096540
      linear in zero  = 0.83285414
      diff            = 1.89e-03
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def deposit_discount_factor(rate: float, days: int, basis: int = 360) -> float:
    """Simple-compounded discount factor:  D = 1 / (1 + r * days/basis).

    Used for money-market deposits.
    """
    return 1 / (1 + rate * days/basis)


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def bootstrap_swap_DFs(par_rates: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Sequential par-swap bootstrap (same recursion as FI 8).

    D[k] = (1 - c * sum_{i<k} delta_i * D[i]) / (1 + c * delta[k])
    """
    D = np.empty_like(par_rates)
    weighted_sum = 0.0
    for i, (c, d) in enumerate(zip(par_rates, deltas)):
        D[i] = (1 - c * weighted_sum) / (1 + c * d)
        weighted_sum += d * D[i]
    return D


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def interp_par_rates(input_tenors: np.ndarray, input_rates: np.ndarray,
                     all_years: np.ndarray) -> np.ndarray:
    """Linearly interpolate par rates onto every integer year in all_years."""
    return np.interp(all_years, input_tenors, input_rates)


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def interp_loglinear_in_D(pillars_T: np.ndarray, pillars_D: np.ndarray,
                          T_eval: float) -> float:
    """Log-linear in D:  ln(D(T)) = ln(D1) + (T-T1)/(T2-T1) * (ln(D2) - ln(D1))

    For T_eval between pillars T1 and T2. Returns D(T_eval).
    """
    # TODO: implement (hint: find bracketing pillars; np.interp can be used on
    # log(D) directly if you pass arrays)
    T1, T2 = pillars_T[0], pillars_T[1]
    D1, D2 = pillars_D[0], pillars_D[1]
    log_D_at_T = np.log(D1) + (T_eval - T1) / (T2 - T1) * (np.log(D2/D1))
    return float(np.exp(log_D_at_T))


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def interp_linear_in_zero(pillars_T: np.ndarray, pillars_D: np.ndarray,
                          T_eval: float) -> float:
    """Linear in zero rate:  compute z_i = -ln(D_i)/T_i, linearly interpolate
    z(T_eval), then back out D(T_eval) = exp(-z*T_eval).
    """
    z1 = -np.log(pillars_D[0]) / pillars_T[0]
    z2 = -np.log(pillars_D[1]) / pillars_T[1]
    z_at_T = z1 + (T_eval - pillars_T[0]) / (pillars_T[1] - pillars_T[0]) * (z2 - z1)
    return float(np.exp(-z_at_T * T_eval))

# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Front end — deposits
    D_3m = deposit_discount_factor(0.045, 91)
    D_6m = deposit_discount_factor(0.047, 183)
    assert abs(D_3m - 0.98875294) < 1e-6
    assert abs(D_6m - 0.97666583) < 1e-6

    # Long end — swaps at 1y, 2y, 5y with interpolation at 3y, 4y
    swap_tenors = np.array([1, 2, 5], dtype=float)
    swap_pars   = np.array([0.050, 0.052, 0.055])
    all_years   = np.array([1, 2, 3, 4, 5], dtype=float)

    pars_interp = interp_par_rates(swap_tenors, swap_pars, all_years)
    assert pars_interp.shape == (5,)
    assert abs(pars_interp[2] - 0.0530) < 1e-6        # 3y interp = 5.30%

    deltas = np.ones(5)
    D_swap = bootstrap_swap_DFs(pars_interp, deltas)
    assert D_swap.shape == (5,)
    assert abs(D_swap[0] - 0.95238095) < 1e-6
    assert abs(D_swap[1] - 0.90349448) < 1e-6
    assert abs(D_swap[2] - 0.85625698) < 1e-6
    assert abs(D_swap[4] - 0.76425867) < 1e-6
    assert (np.diff(D_swap) < 0).all()                  # monotone decreasing

    # Interpolation at T = 3.5y between the 2y and 5y pillars
    pillars_T = np.array([2.0, 5.0])
    pillars_D = np.array([D_swap[1], D_swap[4]])

    D_loglin = interp_loglinear_in_D(pillars_T, pillars_D, T_eval=3.5)
    D_linz   = interp_linear_in_zero(pillars_T, pillars_D, T_eval=3.5)

    assert abs(D_loglin - 0.83096540) < 1e-5
    assert abs(D_linz   - 0.83285414) < 1e-5

    # The two methods disagree by ~2e-3 — a real production-relevant difference
    diff = abs(D_loglin - D_linz)
    assert diff > 1e-4 and diff < 1e-2, f"diff out of expected range: {diff}"

    print(f"D(3M)          = {D_3m:.8f}")
    print(f"D(6M)          = {D_6m:.8f}")
    print(f"D(1y)          = {D_swap[0]:.8f}")
    print(f"D(2y)          = {D_swap[1]:.8f}")
    print(f"D(3y)          = {D_swap[2]:.8f} (par rate interp at year 3 = {pars_interp[2]:.4f})")
    print(f"D(5y)          = {D_swap[4]:.8f}")
    print()
    print(f"Interp at T=3.5y:")
    print(f"  log-linear in D = {D_loglin:.8f}")
    print(f"  linear in zero  = {D_linz:.8f}")
    print(f"  diff            = {diff:.2e}")
    print("\n✓ All checks passed.")
