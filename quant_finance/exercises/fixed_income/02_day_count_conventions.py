"""
FI 2 — Day-Count Conventions (30/360, ACT/360, ACT/365, ACT/ACT)
================================================================

OBJECTIVE
    Compute the simple interest on the same notional ($1M), the same rate (5%),
    and the same calendar period (91 days) under the four major day-count
    conventions. Demonstrate that a 6% ACT/360 rate over a 365-day year pays
    MORE cash than a 6% ACT/365 rate.

ESTIMATED TIME
    15 min

TOPICS
    Simple interest:    interest = N * r * days_count / basis_denominator
    30/360 day-count:   each month = 30 days; each year = 360 days
                        (used by US corporate bonds, some swaps)
    ACT/360:            actual days / 360 (USD/EUR/CHF money market, SOFR)
    ACT/365:            actual days / 365 (GBP money market, SONIA, JPY since 2021)
    ACT/ACT:            actual days / actual days in year (US Treasury bonds)

REFERENCE
    ISDA day-count definitions; OpenGamma "Interest Rate Instruments and
    Market Conventions Guide".

EXPECTED OUTPUT  (N=$1M, r=5%, 91 actual days, 3 30-day months)
    30/360       interest = 12500.000000   (uses 90 conventional days)
    ACT/360      interest = 12638.888889
    ACT/365      interest = 12465.753425
    ACT/ACT      interest = 12465.753425   (non-leap year, same as ACT/365)

    1y year fraction 30/360  = 1.00000000
    1y year fraction ACT/360 = 1.01388889   (365/360 = pays 1.39% more)
    1y year fraction ACT/365 = 1.00000000
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def interest_30_360(notional: float, rate: float, n_30day_months: int) -> float:
    """30/360: assumes each month is 30 days, year is 360 days.

    interest = N * r * (n_30day_months * 30) / 360
    """
    return notional * rate * (n_30day_months * 30) / 360


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def interest_act_360(notional: float, rate: float, days_actual: int) -> float:
    """ACT/360: N * r * days_actual / 360.   (USD/EUR money market)"""
    return notional * rate * days_actual / 360


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def interest_act_365(notional: float, rate: float, days_actual: int) -> float:
    """ACT/365: N * r * days_actual / 365.   (GBP money market)"""
    # TODO: implement
    return notional * rate * days_actual / 365


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def year_fraction(days_actual: int, basis: str) -> float:
    """Return the year-fraction (days / basis_denominator).

    basis must be one of: '30/360', 'ACT/360', 'ACT/365', 'ACT/ACT'.

    For 30/360 and ACT/ACT (non-leap), pass days_actual = 360 / 365 respectively
    (the function should NOT do calendar lookup — it just divides by the
    basis denominator).
    """
    # TODO: implement (hint: dict lookup or if/elif on basis string)
    basis_map = {
        "30/360": 360,
        "ACT/360": 360,
        "ACT/365": 365,
        "ACT/ACT": 365
    }

    return days_actual / basis_map[basis]


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N, r = 1_000_000.0, 0.05
    days_actual = 91

    i_30_360  = interest_30_360 (N, r, n_30day_months=3)      # uses 90 conv days
    i_act_360 = interest_act_360(N, r, days_actual=days_actual)
    i_act_365 = interest_act_365(N, r, days_actual=days_actual)

    assert abs(i_30_360  - 12500.000000) < 1e-4
    assert abs(i_act_360 - 12638.888889) < 1e-4
    assert abs(i_act_365 - 12465.753425) < 1e-4
    # ACT/360 pays the MOST cash because 360 is the smallest denominator
    assert i_act_360 > i_30_360 > i_act_365

    # Year fractions for 1 year (assume 365-day non-leap year)
    yf_30  = year_fraction(360, "30/360")
    yf_360 = year_fraction(365, "ACT/360")
    yf_365 = year_fraction(365, "ACT/365")
    yf_act = year_fraction(365, "ACT/ACT")

    assert abs(yf_30  - 1.00000000) < 1e-8
    assert abs(yf_360 - 1.01388889) < 1e-8     # 365/360 = pays 1.39% more cash
    assert abs(yf_365 - 1.00000000) < 1e-8
    assert abs(yf_act - 1.00000000) < 1e-8

    print(f"30/360       interest = {i_30_360:.6f}   (uses 90 conventional days)")
    print(f"ACT/360      interest = {i_act_360:.6f}")
    print(f"ACT/365      interest = {i_act_365:.6f}")
    print(f"ACT/ACT      interest = {i_act_365:.6f}   (non-leap year, same as ACT/365)")
    print()
    print(f"1y year fraction 30/360  = {yf_30:.8f}")
    print(f"1y year fraction ACT/360 = {yf_360:.8f}   (365/360 = pays 1.39% more)")
    print(f"1y year fraction ACT/365 = {yf_365:.8f}")
    print("\n✓ All checks passed.")
