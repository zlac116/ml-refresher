"""
FI 6 — Money-Market Deposit + FRA Pricing (ACT/360)
===================================================

OBJECTIVE
    Work in money-market conventions (simple interest, ACT/360).
      1. Price a 91-day USD deposit at 5% on $1M notional.
      2. Compute the no-arbitrage 3x6 FRA rate given 3m and 6m simple cash
         rates of 4.5% and 4.8% respectively.

ESTIMATED TIME
    15 min

TOPICS
    Simple interest:  I = N * r * (days / 360)            (USD, GBP, EUR cash)
    ACT/360 vs ACT/365 day-count conventions
    FRA / forward simple rate via no-arb:
        1 + r6 * d6/360  =  (1 + r3 * d3/360) * (1 + f * (d6-d3)/360)

REFERENCE
    Choudhry, "Bond and Money Markets" ch. 14; ISDA market conventions.

REAL-WORLD NOTE
    Most USD/EUR/GBP money-market deposits and SOFR settle ACT/360.
    GBP money-market historically uses ACT/365 — check the convention
    per currency before applying these formulas.

EXPECTED OUTPUT
    3m simple interest   = 12638.888889
    future value         = 1012638.888889
    FRA 3x6 rate         = 0.05039416  (~5.039 %)

GRADING
    All asserts must pass.
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def deposit_interest_act360(notional: float, rate: float, days: int) -> float:
    """Simple interest on an ACT/360 deposit."""
    return notional * (1 + rate * days / 360 - 1)


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def fra_rate(r_short: float, days_short: int,
             r_long:  float, days_long:  int) -> float:
    """No-arb forward simple rate between two ACT/360 deposit tenors:
        1 + r_long  * d_long /360
      = (1 + r_short * d_short/360) * (1 + f * (d_long - d_short)/360)

    Returns f (annualised simple rate, ACT/360).
    """
    t_short = days_short / 360
    t_long  = days_long  / 360
    delta = t_long - t_short
    return (r_long * t_long - r_short * t_short) / (delta * (1 + r_short * t_short))


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    interest = deposit_interest_act360(notional=1_000_000.0, rate=0.05, days=91)
    assert abs(interest - 12638.888889) < 1e-4

    fv = 1_000_000.0 + interest
    assert abs(fv - 1_012_638.888889) < 1e-4

    fra = fra_rate(r_short=0.045, days_short=91,
                   r_long =0.048, days_long =183)
    assert abs(fra - 0.05039416) < 1e-7, f"FRA off: {fra}"

    # Sanity: implied forward must be > short rate when curve slopes upward
    assert fra > 0.045

    print(f"3m simple interest   = {interest:.6f}")
    print(f"future value         = {fv:.6f}")
    print(f"FRA 3x6 rate         = {fra:.8f}  (~{fra * 100:.3f} %)")
    print("\n✓ All checks passed.")
