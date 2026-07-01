"""
FI 1 — Yield Curve Bootstrap from Par Swap Rates
================================================

OBJECTIVE
    Given annual-pay par swap rates at 1y, 2y, 3y, 5y, 10y, bootstrap the
    discount-factor curve D(0,T) and the continuously-compounded zero rate
    at every integer year 1..10.

ESTIMATED TIME
    20 min

TOPICS
    Par swap NPV identity (general form with per-period year fractions delta_i):

        c * sum_{i=1..n} delta_i * D(0, T_i)  +  D(0, T_n)  =  1

    Isolating the unknown D(0, T_n):

        D(0, T_n)  =  (1 - c * sum_{i<n} delta_i * D(0, T_i))  /  (1 + c * delta_n)

    Sequential bootstrap: D(0, T_n) uses ONLY previously-bootstrapped D values.
    The deltas come from the payment schedule + day-count convention; this
    exercise uses annual-pay so delta_i = 1.0 for every i.

    Linearly interpolate par rates at missing tenors (4y, 6y, 7y, 8y, 9y).

REFERENCE
    Hull, ch. 7 (swaps); Andersen-Piterbarg vol I, ch. 5.

EXPECTED OUTPUT
    D(0,1)         = 0.970874
    D(0,5)         = 0.829062
    D(0,10)        = 0.673202
    zero(5y, cont) = 0.037492

GRADING
    All asserts must pass.
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def interp_par_rates(tenors: np.ndarray, par_rates: np.ndarray,
                     all_years: np.ndarray) -> np.ndarray:
    """Linearly interpolate par rates onto every integer year in `all_years`.

    Use numpy.interp.
    """
    return np.interp(all_years, tenors, par_rates)


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def bootstrap_discount_factors(par_rates: np.ndarray,
                               deltas: np.ndarray) -> np.ndarray:
    """Sequentially bootstrap D(0, T_k) for k = 1, 2, ..., len(par_rates).

    `par_rates[k]` is the par swap rate for tenor T_k.
    `deltas[k]`    is the year fraction (day-count) of accrual period k.

    General par-swap identity:
        c * sum_{i=1..n} delta_i * D(0, T_i)  +  D(0, T_n)  =  1
        => D(0, T_n) = (1 - c * sum_{i<n} delta_i * D(0, T_i)) / (1 + c * delta_n)

    Best-practice scaffold (canonical):
        - Pre-allocate D = np.empty(n) (not a list)
        - Carry a SCALAR `weighted_prior_sum = sum_{i<k} delta_i * D[i]`
          and accumulate it incrementally — O(n) total, not O(n^2).
        - Use only previously-set D[:k] when computing D[k].
    """
    n = len(par_rates)
    D = np.empty(n)
    weighted_prior_sum = 0.0
    for k, (c, dk) in enumerate(zip(par_rates, deltas)):
        D[k] = (1 - c * weighted_prior_sum) / (1 + c * dk)
        weighted_prior_sum += dk * D[k]
    return D


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def continuous_zero_rates(D: np.ndarray, years: np.ndarray) -> np.ndarray:
    """z(T) = -ln(D(0,T)) / T. Returns one zero per year."""
    return -np.log(D) / years


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tenors    = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    par_rates = np.array([0.030, 0.034, 0.036, 0.038, 0.040])
    all_years = np.arange(1, 11)

    pis = interp_par_rates(tenors, par_rates, all_years)
    assert pis.shape == (10,)
    assert abs(pis[3] - 0.0370) < 1e-6, "year 4 par should interp to 3.70%"

    # Annual-pay convention: every accrual period is 1.0 years.
    # Swap to semi-annual by passing np.full(10, 0.5) and adjusting tenors.
    deltas = np.ones(10)
    D = bootstrap_discount_factors(pis, deltas)
    assert D.shape == (10,)
    assert abs(D[0] - 0.970874) < 1e-5
    assert abs(D[4] - 0.829062) < 1e-5
    assert abs(D[9] - 0.673202) < 1e-5
    # Discount factors must be monotone decreasing
    assert (np.diff(D) < 0).all()

    z = continuous_zero_rates(D, all_years.astype(float))
    assert abs(z[4] - 0.037492) < 1e-5

    print(f"D(0,1)         = {D[0]:.6f}")
    print(f"D(0,5)         = {D[4]:.6f}")
    print(f"D(0,10)        = {D[9]:.6f}")
    print(f"zero(5y, cont) = {z[4]:.6f}")
    print("\n✓ All checks passed.")
