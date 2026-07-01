"""
DRILL 8 — Historical-Simulation VaR Engine (a miniature of Just's stress engine)
===============================================================================

OBJECTIVE
    Build the core of a historical-simulation VaR engine, then extend it to a
    BOND priced off a rate curve + credit spread:
      1. full-reval bond pricer        (rates + credit spread)
      2. DV01 / CS01 sensitivities     (1bp bumps)
      3. P&L vector by FULL REVALUATION under each historical scenario
      4. 99.5% historical VaR          (sort -> percentile index)

RUN
    uv run python hist_var_drill.py      (pure stdlib; numpy equivalents noted)
    (stuck? hist_var_drill_SOLUTIONS.py)

METHOD (historical simulation VaR)
    For each historical day: shock today's risk factors by THAT day's move,
    revalue the portfolio, record P&L = V(shocked) - V(base). Sort the P&Ls;
    the 99.5% VaR is the loss at the (1-0.995)*N-th worst scenario.
        VaR(99.5%) = -sorted_pnl[ floor(0.005 * N) ]
    99.5% / 1-yr is the Solvency II SCR market-risk calibration.

GIVEN (do not change)
    A zero curve + zero_rate(); a deterministic 250-day history of risk-factor
    daily changes (bp) for rates and credit spread; a 5y 4% annual coupon bond.

EXPECTED OUTPUT
    base price:       92.0706
    DV01 (per 1bp):    0.042462      CS01 (per 1bp): 0.042462   (equal here — see note)
    P&L min / max:    -0.8011 / 0.8504
    99.5% VaR:         0.7363
    99.0% VaR:         0.7033

GRADING
    All asserts must pass.
"""

import numpy as np
# ── GIVEN: market + helpers (do not change) ─────────────────────────────────
CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]

def interp(x, xs, ys):
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i-1]) / (xs[i] - xs[i-1]); return ys[i-1] + w*(ys[i]-ys[i-1])

def zero_rate(t, rate_bump_bp=0.0):
    """Interpolated zero rate at t + an optional parallel rate bump (bp)."""
    return interp(t, CURVE_T, CURVE_Z) + rate_bump_bp/1e4

# Deterministic 250-day history of daily risk-factor changes, in bp (no RNG)
N = 250
d_rate_bp   = [8*np.sin(i*0.3) + ((i % 11) - 5) for i in range(N)]
d_spread_bp = [5*np.cos(i*0.2) + ((i % 7) - 3) for i in range(N)]

# Bond: 5-year, 4% annual coupon, notional 100; current credit spread 150bp
TIMES = [1, 2, 3, 4, 5]
CFS   = [4, 4, 4, 4, 104]
BASE_SPREAD_BP = 150.0


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1 — Full-reval bond pricer            (rates + credit spread)
#   FORMULA:  P = Σ CFₜ · exp( −(z(t) + s)·t )
#     z(t) = zero_rate(t, rate_bump_bp) ;  s = spread_bp/1e4  (bp -> decimal)
#   EXPECTED: bond_price(...) = 92.0706   (rate_bump=0, spread=150bp)
# ═══════════════════════════════════════════════════════════════════════════
def bond_price(times, cfs, rate_bump_bp=0.0, spread_bp=BASE_SPREAD_BP):
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2 — DV01 & CS01                        (1bp bumps)
#   DV01 = P(rate) − P(rate + 1bp)        (sensitivity to the rate curve)
#   CS01 = P(spread) − P(spread + 1bp)    (sensitivity to the credit spread)
#   EXPECTED: both ≈ 0.042462
#   NOTE: they're EQUAL here because rate and spread both add to the discount
#         exponent; they'd DIFFER for a floating-rate note or a CDS.
# ═══════════════════════════════════════════════════════════════════════════
def dv01(times, cfs, spread_bp=BASE_SPREAD_BP):
    raise NotImplementedError

def cs01(times, cfs, spread_bp=BASE_SPREAD_BP):
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3 — P&L vector by FULL REVALUATION under each historical scenario
#   For each day i:  shock rates by d_rate_bp[i] and spread by d_spread_bp[i];
#   P&L_i = bond_price(shocked) − bond_price(base).
#   Return a list of N P&Ls.
#   EXPECTED: min ≈ −0.8011 , max ≈ 0.8504
# ═══════════════════════════════════════════════════════════════════════════
def pnl_vector(times, cfs, base_spread_bp, d_rate_bp, d_spread_bp):
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# TASK 4 — Historical VaR at a confidence level
#   Sort P&Ls ascending; idx = floor((1−conf)·N); VaR = −sorted[idx]
#   (loss reported as a POSITIVE number). numpy: -np.percentile(pnls,(1-conf)*100)
#   EXPECTED: VaR(99.5%) = 0.7363 ,  VaR(99.0%) = 0.7033
# ═══════════════════════════════════════════════════════════════════════════
def historical_var(pnls, conf=0.995):
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# GRADING
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    assert abs(d_rate_bp[0] - (-5.0)) < 1e-9 and abs(d_spread_bp[0] - 2.0) < 1e-9  # given check

    base = bond_price(TIMES, CFS)
    assert abs(base - 92.0706468) < 1e-5, f"price off: {base}"

    assert abs(dv01(TIMES, CFS) - 0.0424617) < 1e-6, dv01(TIMES, CFS)
    assert abs(cs01(TIMES, CFS) - 0.0424617) < 1e-6, cs01(TIMES, CFS)

    pnls = pnl_vector(TIMES, CFS, BASE_SPREAD_BP, d_rate_bp, d_spread_bp)
    assert len(pnls) == N
    assert abs(min(pnls) - (-0.8011423)) < 1e-5, min(pnls)
    assert abs(max(pnls) -   0.8504003)  < 1e-5, max(pnls)

    v995 = historical_var(pnls, 0.995)
    v99  = historical_var(pnls, 0.99)
    assert abs(v995 - 0.7362917) < 1e-5, f"VaR99.5 off: {v995}"
    assert abs(v99  - 0.7032813) < 1e-5, f"VaR99 off: {v99}"

    print(f"base price:   {base:.4f}")
    print(f"DV01 / CS01:  {dv01(TIMES,CFS):.6f} / {cs01(TIMES,CFS):.6f}")
    print(f"P&L min/max:  {min(pnls):.4f} / {max(pnls):.4f}")
    print(f"99.5% VaR:    {v995:.4f}")
    print(f"99.0% VaR:    {v99:.4f}")
    print("\n✓ All checks passed.")
