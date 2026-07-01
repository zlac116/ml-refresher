"""
XVA 1 — CVA from an Exposure Profile + Survival Probabilities
==============================================================

OBJECTIVE
    Compute Credit Value Adjustment (CVA) for an uncollateralised derivative
    given:
      1. An Expected Positive Exposure (EPE) time profile.
      2. Risk-free discount factors.
      3. Counterparty survival probabilities from a flat 200 bp hazard rate.
    Then back out the hazard rate implied by a 5y 200 bp CDS spread.

ESTIMATED TIME
    20 min

TOPICS
    CVA = (1 - R) * sum_i EE_i * DF_i * PD_i
      where R = recovery, EE_i = exposure at bucket i, DF_i = discount factor,
      PD_i = P(default in (t_{i-1}, t_i]) = S(t_{i-1}) - S(t_i)
    Hazard-rate model: S(t) = exp(-λ * t).
    Credit triangle (CDS approx):  spread ≈ (1 - R) * λ   →   λ ≈ s / (1-R).

REAL-WORLD NOTE
    Bilateral CVA also subtracts DVA (own-default benefit). Modern desks
    compute CVA on a Monte Carlo simulated EPE profile per netting set,
    then aggregate. Hazard rates come from the CDS curve (or proxy).
    Basel: CCR + CVA capital are separate add-ons.

REFERENCE
    Gregory, "The xVA Challenge" 4th ed.; Brigo et al, "Counterparty Credit Risk".

EXPECTED OUTPUT
    sum PD over 5y:        0.095163
    EPE peak:              6200
    D(0,5y):               0.860708
    S(5y):                 0.904837
    CVA:                   217.053
    hazard rate from CDS:  0.033333

GRADING
    All asserts must pass.
"""
import numpy as np


# Buckets every 6 months out to 5y
TIMES = np.arange(0.5, 5.5, 0.5)
# Mock EPE profile (peak, then runs off — typical for amortising swaps)
EPE   = np.array([2000, 4500, 5800, 6200, 5800, 5200, 4500, 3500, 2200, 1000])
R_FREE  = 0.03    # continuous risk-free rate
RECOVERY = 0.40   # 40% recovery; LGD = 60%
HAZARD   = 0.02   # 200 bps flat hazard


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def discount_factors(times: np.ndarray, r: float) -> np.ndarray:
    """D(0, t) = exp(-r * t)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def survival_probabilities(times: np.ndarray, hazard: float) -> np.ndarray:
    """S(t) = exp(-hazard * t)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def default_probabilities(survival: np.ndarray) -> np.ndarray:
    """PD_i = S(t_{i-1}) - S(t_i), with S(t_0) = 1 (origin)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def cva(epe: np.ndarray, discounts: np.ndarray, default_probs: np.ndarray,
        recovery: float) -> float:
    """CVA = (1 - recovery) * sum(EPE * D * PD)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def hazard_from_cds_spread(spread: float, recovery: float) -> float:
    """Credit triangle approximation: lambda = spread / (1 - recovery)."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    D = discount_factors(TIMES, R_FREE)
    assert abs(D[-1] - 0.860708) < 1e-5

    S = survival_probabilities(TIMES, HAZARD)
    assert abs(S[-1] - 0.904837) < 1e-5

    PD = default_probabilities(S)
    assert abs(PD.sum() - 0.095163) < 1e-5
    assert (PD > 0).all()

    val = cva(EPE, D, PD, RECOVERY)
    assert abs(val - 217.053) < 1e-2

    hz = hazard_from_cds_spread(spread=0.0200, recovery=RECOVERY)
    assert abs(hz - 0.033333) < 1e-5

    print(f"sum PD over 5y:        {PD.sum():.6f}")
    print(f"EPE peak:              {EPE.max()}")
    print(f"D(0,5y):               {D[-1]:.6f}")
    print(f"S(5y):                 {S[-1]:.6f}")
    print(f"CVA:                   {val:.3f}")
    print(f"hazard rate from CDS:  {hz:.6f}")
    print("\n✓ All checks passed.")
