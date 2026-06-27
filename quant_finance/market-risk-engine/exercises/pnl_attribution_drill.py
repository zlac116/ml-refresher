"""
DRILL 12 — P&L Attribution  ("explain the P&L")
===============================================

A daily valuation-desk task: take the full-reval P&L of a move and explain it
by RISK FACTOR — first-order (delta) + second-order (gamma) — and report the
UNEXPLAINED residual. That residual is what the FRTB P&L-Attribution (PLA) test
watches: if too much P&L is unexplained, your internal model fails.

Builds on your capstone engine (imported). You implement:
  TASK 1  gammas(base)            second-order sensitivities (central difference)
  TASK 2  attribute(base, scen)   decompose full P&L -> per-factor + residual

KEY IDEA
  full_pnl = V(shocked) − V(base)
  first_order_f  = sensitivity_f · shock_f                 (per factor)
  second_order   = Σ ½ · gamma_f · shock_f²
  unexplained    = full_pnl − first_order_total − second_order
  -> adding gamma SHRINKS the residual; what's left is cross-gamma + higher-order.

RUN
    uv run python pnl_attribution_drill.py     (stuck? pnl_attribution_drill_SOLUTIONS.py)

EXPECTED (scenario: +15bp rate, +10bp spread, +2 vol pts, 0% fx)
    first-order by factor:  rate -2.9377 | spread -1.2261 | vol +1.1095 | fx 0.0
    full P&L -3.0239 | first-order total -3.0543 | second-order +0.0228
    unexplained: first-only +0.0304  ->  with gamma +0.0076   (~4x smaller)

GRADING: all asserts must pass.
"""
import numpy as np
from dataclasses import replace
from extension_drill2_SOLUTIONS import (
    Market, Scenario, Portfolio, Bond, InterestRateSwap, CreditDefaultSwap,
    TotalReturnSwap, ForeignBond, Swaption, Repo, SecLending, Loan, FXSpot, FXOption)

# ── GIVEN: the capstone portfolio ───────────────────────────────────────────
PF = Portfolio([
    Bond((1,2,3,4,5),(4,4,4,4,104),150.0),
    InterestRateSwap((1,2,3,4,5),1.0,0.042,100.0),
    CreditDefaultSwap((1,2,3,4,5),1.0,100.0,120.0,100.0),
    TotalReturnSwap(Bond((1,2,3,4,5),(4,4,4,4,104),150.0),90.0),
    ForeignBond((1,2,3,4,5),(3,3,3,3,103),100.0,1.25),
    Swaption(2.0,(3,4,5),1.0,0.042,0.20,100.0),
    Repo(100.0,0.043,0.5), SecLending(100.0,20.0,(1,2,3),1.0), Loan(100.0,0.05,5,200.0),
    FXSpot(80.0,1.25), FXOption(1.0,1.30,0.12,1.25,100.0)])


class PnLAttributor:
    def __init__(self, portfolio): self.pf = portfolio

    # GIVEN: first-order sensitivities (per 1bp rate/spread, per 1.0 vol/fx)
    def sensitivities(self, base):
        v0 = self.pf.value(base)
        return (self.pf.value(replace(base, rate_bump_bp=base.rate_bump_bp+1)) - v0,
                self.pf.value(replace(base, spread_bump_bp=base.spread_bump_bp+1)) - v0,
                (self.pf.value(replace(base, vol_shock=base.vol_shock+0.01)) - v0)/0.01,
                (self.pf.value(replace(base, fx_shock=base.fx_shock+0.01)) - v0)/0.01)

    # ═══════════════════════════════════════════════════════════════════════
    # TASK 1 — second-order sensitivities (gamma) by CENTRAL DIFFERENCE
    #   g = [V(+h) − 2·V(base) + V(−h)] / h²       (h = bump size)
    #   rate/spread: bump ±1bp -> divide by 1² (i.e. per bp²)
    #   vol/fx:      bump ±0.01 -> divide by 0.01² (per 1.0²)
    #   return (g_rate, g_spread, g_vol, g_fx)
    #   EXPECTED ≈ (1.855e-4, 4.287e-5, -1.06362, 412.9212)
    # ═══════════════════════════════════════════════════════════════════════
    def gammas(self, base):
        raise NotImplementedError

    # ═══════════════════════════════════════════════════════════════════════
    # TASK 2 — attribute the full P&L of a scenario
    #   full   = pf.value(scenario.apply(base)) − pf.value(base)
    #   first_*  = sensitivity_* · shock_*   (rate·d_rate_bp, spread·d_spread_bp,
    #                                         vol·d_vol, fx·fx_ret)
    #   second = Σ ½ · gamma_* · shock_*²
    #   Return a dict:
    #     {"rate":.., "spread":.., "vol":.., "fx":..,          # first-order by factor
    #      "first_order":.., "second_order":.., "full":..,
    #      "unexplained":.. }                                   # = full − first − second
    # ═══════════════════════════════════════════════════════════════════════
    def attribute(self, base, scenario):
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# GRADING
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    attr = PnLAttributor(PF)
    base = Market()
    gr, gs, gv, gf = attr.gammas(base)
    assert abs(gr - 1.855e-4) < 1e-6 and abs(gs - 4.287e-5) < 1e-7
    assert abs(gv - -1.0636217) < 1e-5 and abs(gf - 412.9211975) < 1e-4

    scen = Scenario(15.0, 10.0, 0.02, 0.0)            # +15bp, +10bp, +2 vol pts, 0% fx
    a = attr.attribute(base, scen)
    assert abs(a["rate"]   - -2.9376858) < 1e-5
    assert abs(a["spread"] - -1.2260855) < 1e-5
    assert abs(a["vol"]    -  1.1094981) < 1e-5
    assert abs(a["fx"]     -  0.0)        < 1e-9
    assert abs(a["full"]         - -3.0238731) < 1e-5
    assert abs(a["first_order"]  - -3.0542732) < 1e-5
    assert abs(a["second_order"] -  0.0228052) < 1e-6
    assert abs(a["unexplained"]  -  0.0075949) < 1e-6
    # gamma must shrink the residual vs first-order-only
    assert abs(a["unexplained"]) < abs(a["full"] - a["first_order"])

    print("first-order by factor:")
    for k in ("rate","spread","vol","fx"): print(f"   {k:7s} {a[k]:+.4f}")
    print(f"full P&L      {a['full']:+.4f}")
    print(f"first-order   {a['first_order']:+.4f}")
    print(f"second-order  {a['second_order']:+.4f}")
    print(f"unexplained:  first-only {a['full']-a['first_order']:+.4f}  ->  with gamma {a['unexplained']:+.4f}")
    print("\n✓ All checks passed.")
