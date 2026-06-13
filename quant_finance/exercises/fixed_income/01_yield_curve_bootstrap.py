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
    Par swap NPV identity:  sum_{k=1..T} c*tau_k*D(0,k) + D(0,T) = 1
    Sequential bootstrap: D(0,T) solved from prior D values.
    Linearly interpolate par rates at missing tenors (4y, 6y, 7y, 8y, 9y).

REFERENCE
    Hull, ch. 6 (swaps); Andersen-Piterbarg vol I, ch. 4.

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
def bootstrap_discount_factors(par_at_year: np.ndarray) -> np.ndarray:
    """Sequentially bootstrap D(0, T) for T = 1, 2, ..., len(par_at_year).

    For year T with par rate c:
        sum_{k=1..T-1} c * D(0,k)  +  (1 + c) * D(0,T)  =  1
        =>  D(0,T) = (1 - c * sum_{k=1..T-1} D(0,k)) / (1 + c)
    """
    df = []
    for t, r in enumerate(par_at_year):
        df.append(1 / (1 + r * (t+1)))
    breakpoint()
    return np.asarray(df)

# ── TASK 3 ─────────────────────────────────────────────────────────────────
def continuous_zero_rates(D: np.ndarray, years: np.ndarray) -> np.ndarray:
    """z(T) = -ln(D(0,T)) / T. Returns one zero per year."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tenors    = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    par_rates = np.array([0.030, 0.034, 0.036, 0.038, 0.040])
    all_years = np.arange(1, 11)

    pis = interp_par_rates(tenors, par_rates, all_years)
    assert pis.shape == (10,)
    assert abs(pis[3] - 0.0370) < 1e-6, "year 4 par should interp to 3.70%"

    D = bootstrap_discount_factors(pis)
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
