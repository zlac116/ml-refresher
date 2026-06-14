"""
FI 5 — Loan Amortisation Schedule (Fixed-Rate Mortgage)
=======================================================

OBJECTIVE
    For a $200,000 30-year mortgage at 6% annual rate (monthly compounding):
      1. Compute the level monthly payment.
      2. Build the full amortisation schedule (balance, interest, principal).
      3. Verify the schedule closes (balance after 360 months = 0).
      4. Confirm the IRR of the cashflows equals the contractual monthly rate.

ESTIMATED TIME
    20 min

TOPICS
    Annuity payment formula:  P * r * (1+r)^n / ((1+r)^n - 1)
    Per-month: interest_i = balance_{i-1} * r;  principal_i = pmt - interest_i
    Effective annual rate = (1 + r_monthly)^12 - 1

REFERENCE
    Bodie-Kane-Marcus, "Investments", ch. 14; standard mortgage math.

REAL-WORLD NOTE
    US 30-year fixed mortgages quote the NOMINAL annual rate but compound
    monthly — the effective annual rate is higher than the quoted nominal.

EXPECTED OUTPUT  (P=200_000, r=6% nominal annual, n=360)
    monthly pmt       = 1199.101050
    balance after 60m = 186108.713646
    final balance     = ~0
    interest yr 1     = 11933.1892
    principal yr 1    =  2456.0234
    effective annual  = 6.168 %
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def monthly_payment(principal: float, annual_rate_nominal: float, years: int) -> float:
    """Level monthly payment for a fully-amortising loan.

    Use the annuity formula with r = annual_rate_nominal / 12 and n = years*12.
    """
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def amortisation_schedule(principal: float, annual_rate_nominal: float,
                          years: int) -> dict[str, np.ndarray]:
    """Return a dict with arrays:
      balance:    length n+1, balance[0] = principal, balance[n] ~ 0
      interest:   length n, interest paid each month
      principal:  length n, principal paid each month

    Loop is fine — the schedule is recursive.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def effective_annual_rate(annual_rate_nominal: float, freq: int = 12) -> float:
    """(1 + r_nominal/freq)^freq - 1."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    P, rnom, yrs = 200_000.0, 0.06, 30
    n = yrs * 12

    pmt = monthly_payment(P, rnom, yrs)
    assert abs(pmt - 1199.101050) < 1e-4

    sch = amortisation_schedule(P, rnom, yrs)
    assert set(sch) == {"balance", "interest", "principal"}
    assert len(sch["balance"]) == n + 1
    assert len(sch["interest"]) == n == len(sch["principal"])
    assert sch["balance"][0] == P
    assert abs(sch["balance"][60] - 186108.713646) < 1e-3
    assert abs(sch["balance"][-1]) < 1e-6, "schedule must close"
    # Interest + principal = payment, every month
    pay_check = sch["interest"] + sch["principal"]
    assert np.allclose(pay_check, pmt, atol=1e-8)
    assert abs(sch["interest"][:12].sum()  - 11933.1892) < 1e-3
    assert abs(sch["principal"][:12].sum() -  2456.0234) < 1e-3

    ear = effective_annual_rate(rnom, 12)
    assert abs(ear - 0.061678) < 1e-5

    print(f"monthly pmt       = {pmt:.6f}")
    print(f"balance after 60m = {sch['balance'][60]:.6f}")
    print(f"final balance     = {sch['balance'][-1]:.2e}")
    print(f"interest yr 1     = {sch['interest'][:12].sum():.4f}")
    print(f"principal yr 1    = {sch['principal'][:12].sum():.4f}")
    print(f"effective annual  = {ear * 100:.3f} %")
    print("\n✓ All checks passed.")
