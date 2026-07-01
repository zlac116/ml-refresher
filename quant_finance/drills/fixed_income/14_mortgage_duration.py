"""
FI 14 — Mortgage Duration: Macaulay on an Amortising Schedule
================================================================

OBJECTIVE
    Apply the Macaulay duration formula to a fixed-rate amortising mortgage.
    Verify the textbook claim from 06_loan_amortisation.md: an amortising
    loan has roughly HALF the duration of a bullet bond of the same maturity.

ESTIMATED TIME
    20 min

TOPICS
    Macaulay duration formula is unchanged:  D_mac = sum(t_i * PV_i) / sum(PV_i)
    For an amortising loan, every cashflow is the level payment PMT
    Discount each cf at the contractual per-period rate (= YTM of the loan)
    Compare to a bullet bond of equal maturity to confirm the "half" rule

REFERENCE
    Hull, ch. 4; Bodie-Kane-Marcus, ch. 14.

REAL-WORLD NOTE
    This is the deterministic (no-prepayment) duration. Real mortgage books
    layer a prepayment model (CPR/PSA) on top, which cuts duration further
    to ~5-7 years for a typical 30y fixed product. See 06_loan_amortisation.md
    section on prepayment risk.

EXPECTED OUTPUT  (5y mortgage, $100k, 6% nominal monthly compounding)
    Monthly payment           = $1,933.28
    Mac D (amortising 5y)     = 2.417198 yrs    ← about half maturity
    Mac D (5y bullet 6% bond) = 4.332016 yrs    ← close to maturity
    Ratio amort / bullet      = 0.5580           (the "half" rule confirmed)
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def monthly_payment(principal: float, annual_rate_nominal: float, years: int) -> float:
    """Level monthly payment from the annuity formula."""
    r = annual_rate_nominal / 12 # monthly rate
    n = years * 12 # total months
    pmt = principal * r * ((1 + r)**n) / ((1 + r)**n - 1) # monthly payment
    return pmt


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def mortgage_mac_duration(principal: float, annual_rate_nominal: float,
                          years: int) -> float:
    """Macaulay duration of an amortising mortgage.

      times[i]   = i / 12   (year fraction at end of month i, i = 1..n)
      cashflows  = level payment PMT, same every month
      PV_i       = PMT / (1 + r_per)^i        where r_per = annual_rate_nominal/12
      D_mac      = sum(times[i] * PV_i) / sum(PV_i)

    Returns Macaulay duration in YEARS.
    """
    n = int(years * 12)
    r_per = annual_rate_nominal / 12
    periods = np.arange(1, n + 1)
    times = periods / 12
    pmt = monthly_payment(principal, annual_rate_nominal, years)
    pv = pmt / (1 + r_per)**periods
    return np.sum(pv * times) / np.sum(pv)


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def bullet_bond_mac_duration(face: float, coupon: float, ytm: float,
                             T: float, freq: int = 12) -> float:
    """Standard Macaulay duration for a bullet (non-amortising) coupon bond,
    discounted at YTM. Used as the comparison anchor.
    """
    # TODO: implement
    n = int(T * freq)
    periods = np.arange(1, n + 1)
    times = periods / freq
    cf = np.full(n, coupon * face / freq)
    cf[-1] += face
    pv = cf / (1 + ytm / freq)**periods
    return np.sum(pv * times) / np.sum(pv)


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    P, r_nom, yrs = 100_000.0, 0.06, 5

    pmt = monthly_payment(P, r_nom, yrs)
    assert abs(pmt - 1933.280153) < 1e-3

    D_amort = mortgage_mac_duration(P, r_nom, yrs)
    assert abs(D_amort - 2.417198) < 1e-4

    D_bullet = bullet_bond_mac_duration(100, 0.06, 0.06, 5, freq=12)
    assert abs(D_bullet - 4.332016) < 1e-4

    ratio = D_amort / D_bullet
    assert abs(ratio - 0.5580) < 1e-3
    # The "half rule" — amort is roughly half the bullet's duration
    assert 0.45 < ratio < 0.65

    # Sanity: amort duration must be POSITIVE and < legal maturity
    assert 0 < D_amort < yrs
    # Sanity: bullet duration must be close to (but less than) maturity for coupon bond
    assert 0.8 * yrs < D_bullet < yrs

    print(f"Monthly payment           = ${pmt:,.2f}")
    print(f"Mac D (amortising 5y)     = {D_amort:.6f} yrs    ← about half maturity")
    print(f"Mac D (5y bullet 6% bond) = {D_bullet:.6f} yrs    ← close to maturity")
    print(f"Ratio amort / bullet      = {ratio:.4f}           (the 'half' rule confirmed)")
    print("\n✓ All checks passed.")
