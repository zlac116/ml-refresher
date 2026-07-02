"""
FUT 2 — Oil Futures: Contango, Backwardation, Roll Yield, Calendar Spreads
==========================================================================

OBJECTIVE
    Work with WTI-style monthly futures.
      1. Detect contango vs backwardation from front + second-month prices.
      2. Compute roll yield per roll and annualised (12 monthly rolls).
      3. P&L on a long calendar spread (long M1 / short M2).
      4. Back out implied (storage - convenience yield) from F = S*exp((r+u-y)T).

ESTIMATED TIME
    20 min

TOPICS
    Contango: F(T2) > F(T1) for T2 > T1  (storage cost > convenience yield)
    Backwardation: F(T2) < F(T1)         (supply tight, holders prefer spot)
    Roll yield ≈ (F_front - F_next) / F_front per roll
    F_T = S_0 * exp((r + u - y) * T)  where u = storage, y = convenience yield

REAL-WORLD NOTE
    CME WTI Light Sweet Crude (CL): contract = 1000 barrels;
        $0.01/barrel tick = $10/contract.
    During April 2020, May WTI settled at -$37.63 — futures CAN go negative
    when storage hits Cushing capacity. The exponential formula breaks down
    in those regimes (negative spot, storage above capacity).

REFERENCE
    Hull, ch. 5; Geman, "Commodities and Commodity Derivatives" ch. 3.

EXPECTED OUTPUT  (M1=75.00, M2=76.20)
    in contango:         True
    roll per period:     -0.016000
    annualised roll:     -0.192000   (about -19.2% / year for a passive long)
    cal-spread $ PnL:    7000.00     (10 spreads, +0.70 net, mult 1000)
    implied (u-y):       0.003692    (S=74, F=75, r=5%, T=0.25y)

GRADING
    All asserts must pass.
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def is_contango(front: float, second: float) -> bool:
    """True if F(second_month) > F(front_month)."""
    return second > front


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def roll_yield(front: float, second: float, rolls_per_year: int = 12) -> tuple[float, float]:
    """Return (per_roll, annualised) roll yield.

    per_roll       = (front - second) / front
    annualised     = per_roll * rolls_per_year
    Contango → both NEGATIVE for a passive long.
    """
    per_roll = (front - second) / front
    ann = per_roll * rolls_per_year
    return per_roll, ann


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def calendar_spread_pnl(basis_initial: float, basis_final: float,
                        n_spreads: int, multiplier: float) -> float:
    """Long the calendar spread = long M1, short M2.
    Spread value at any time = F(M1) - F(M2) = basis.
    PnL per spread per unit = basis_final - basis_initial.
    Total $ PnL = pnl_per_spread * n_spreads * multiplier.
    """
    return n_spreads * multiplier * (basis_final - basis_initial)


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def implied_storage_minus_convenience(S: float, F: float, r: float, T: float) -> float:
    """Back out (u - y) from F = S * exp((r + u - y) * T):
        (u - y) = ln(F/S) / T - r
    """
    return (np.log(F / S) / T) - r


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    front, second = 75.00, 76.20
    assert is_contango(front, second) is True
    assert is_contango(80.0, 78.0) is False

    per, ann = roll_yield(front, second, rolls_per_year=12)
    assert abs(per - -0.016)  < 1e-6
    assert abs(ann - -0.192)  < 1e-5

    pnl = calendar_spread_pnl(basis_initial=-1.20, basis_final=-0.50,
                              n_spreads=10, multiplier=1000.0)
    assert abs(pnl - 7000.0) < 1e-6

    uy = implied_storage_minus_convenience(S=74.0, F=75.0, r=0.05, T=0.25)
    assert abs(uy - 0.003692) < 1e-5

    print(f"in contango:         {is_contango(front, second)}")
    print(f"roll per period:     {per:.6f}")
    print(f"annualised roll:     {ann:.6f}   (about -{abs(ann)*100:.1f}% / year for a passive long)")
    print(f"cal-spread $ PnL:    {pnl:.2f}     (10 spreads, +0.70 net, mult 1000)")
    print(f"implied (u-y):       {uy:.6f}    (S=74, F=75, r=5%, T=0.25y)")
    print("\n✓ All checks passed.")
