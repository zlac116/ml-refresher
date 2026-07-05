"""
FI 6 — FRA Settlement: Discount-at-Fixing Payment
=================================================

OBJECTIVE
    The 3x6 FRA you priced in exercise 05 has a contracted rate F = 5.04%.
    At fixing date the 3M reference rate is observed at L. Compute the actual
    cash payment that changes hands at T_1 (the start of the forward period).

ESTIMATED TIME
    15 min

TOPICS
    Economic interest differential at T_2:    N * (L - F) * δ
    But FRA settles at T_1 (not T_2), so we discount BACK at the observed L:

                            N * (L - F) * δ
        Payment_at_T_1  =  ─────────────────
                              1 + L * δ

    Sign convention: the BUYER (receive floating, pay fixed) receives positive
    payment when L > F (rates rose above the contracted rate); pays when L < F.

REFERENCE
    ISDA FRA definitions; Choudhry "Bond and Money Markets" ch. 14.

REAL-WORLD NOTE
    The discount factor 1/(1 + L*δ) is what makes FRAs "fair" vs cash-settled
    futures. Futures pay (L-F)*δ undiscounted (with daily MtM); FRAs pay the
    discounted amount at T_1, hence no convexity adjustment for FRAs.

EXPECTED OUTPUT  (N=$10M, F=5.04%, L=5.50%, δ=92/360)
    Interest diff at T_2   = 11755.555556
    Discount factor        = 0.98613926
    Cash payment at T_1    = 11592.614913   (positive: buyer receives)

    If L=4.5% (below F):   = -13643.104301  (negative: buyer pays)
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def fra_settlement_payment(notional: float, contracted_rate: float,
                           fixing_rate: float, days_period: int,
                           basis: int = 360) -> float:
    """Cash payment at T_1 (start of the forward period):

        payment = N * (L - F) * δ  /  (1 + L * δ)

    where δ = days_period / basis.
    Positive payment → buyer receives.
    """
    delta = days_period / basis
    DFt = 1 / (1 + fixing_rate * delta)
    return notional * (fixing_rate - contracted_rate) * delta * DFt


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def fra_interest_differential_at_T2(notional: float, contracted_rate: float,
                                    fixing_rate: float, days_period: int,
                                    basis: int = 360) -> float:
    """The UNDISCOUNTED interest differential at T_2:

        diff = N * (L - F) * δ

    This is the economic exposure; the FRA discounts it back to T_1.
    """
    delta = days_period / basis
    return notional * (fixing_rate - contracted_rate) * delta


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N      = 10_000_000.0
    F      = 0.0504
    L_up   = 0.0550      # fixing above contracted (buyer wins)
    L_down = 0.0450      # fixing below contracted (buyer loses)
    days   = 92

    diff_T2 = fra_interest_differential_at_T2(N, F, L_up, days)
    assert abs(diff_T2 - 11755.555556) < 1e-4

    payment_up = fra_settlement_payment(N, F, L_up, days)
    assert abs(payment_up - 11592.614913) < 1e-4
    # Payment must be LESS than the undiscounted T_2 amount (we discounted)
    assert payment_up < diff_T2

    payment_down = fra_settlement_payment(N, F, L_down, days)
    assert payment_down < 0   # buyer pays when fixing falls below contract
    assert abs(payment_down - -13643.104301) < 1e-4

    # Discount factor must equal 1 / (1 + L*δ) at the fixing
    df_implied = payment_up / diff_T2
    assert abs(df_implied - 0.98613926) < 1e-6

    print(f"Interest diff at T_2   = {diff_T2:.6f}")
    print(f"Discount factor        = {df_implied:.8f}")
    print(f"Cash payment at T_1    = {payment_up:.6f}   (positive: buyer receives)")
    print(f"\nIf L=4.5% (below F):   = {payment_down:.6f}  (negative: buyer pays)")
    print("\n✓ All checks passed.")
