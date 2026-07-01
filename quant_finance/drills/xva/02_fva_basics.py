"""
XVA 2 — FVA: Funding Value Adjustment Basics
============================================

OBJECTIVE
    Approximate the Funding Value Adjustment (FVA) on an uncollateralised
    one-way trade given:
      1. An expected positive exposure (EPE) profile.
      2. Discount factors.
      3. A flat funding spread over risk-free (the bank's CDS spread is a
         common proxy).

ESTIMATED TIME
    15 min

TOPICS
    Crude FVA approximation (discrete-time integration):
        FVA ≈ funding_spread * sum_i EPE_i * D_i * Δt_i
    This funds the uncollateralised exposure at the bank's spread.

REAL-WORLD NOTE
    The "FVA debate" (Hull-White 2012 vs Burgard-Kjaer 2011) sparked
    industry-wide disagreement about whether FVA exists as an economic
    cost or is double-counting CVA/DVA. Today it's standard front-office
    P&L.  FBA (Funding Benefit Adjustment) is the negative counterpart
    for liabilities. MVA covers initial-margin funding under cleared/SIMM.

REFERENCE
    Hull-White, "The FVA Debate" Risk Magazine (2012/2014).
    Burgard-Kjaer (2011), "PDE Representations of Derivatives with Bilateral
    Counterparty Risk and Funding Costs".

EXPECTED OUTPUT
    avg EPE:               4070.00
    FVA:                   151.171
    FVA / avg EPE:         3.71%

GRADING
    All asserts must pass.
"""
import numpy as np


# Same EPE profile as XVA 1 — 6-month buckets out to 5y
TIMES   = np.arange(0.5, 5.5, 0.5)
EPE     = np.array([2000, 4500, 5800, 6200, 5800, 5200, 4500, 3500, 2200, 1000])
R_FREE  = 0.03      # risk-free (continuous)
FUNDING_SPREAD = 0.0080   # 80 bps over risk-free
DT = 0.5            # bucket length in years


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def discount_factors(times: np.ndarray, r: float) -> np.ndarray:
    """D(0, t) = exp(-r * t)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def fva(epe: np.ndarray, discounts: np.ndarray, funding_spread: float, dt: float) -> float:
    """FVA ≈ funding_spread * sum(EPE * D * dt)."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    D = discount_factors(TIMES, R_FREE)
    f = fva(EPE, D, FUNDING_SPREAD, DT)
    assert abs(f - 151.171) < 1e-2, f

    # Doubling the funding spread should roughly double the FVA (it's linear in spread)
    f2 = fva(EPE, D, FUNDING_SPREAD * 2, DT)
    assert abs(f2 - 2 * f) < 1e-6

    # FVA is positive when EPE is positive
    assert f > 0

    print(f"avg EPE:               {EPE.mean():.2f}")
    print(f"FVA:                   {f:.3f}")
    print(f"FVA / avg EPE:         {f / EPE.mean() * 100:.2f}%")
    print("\n✓ All checks passed.")
