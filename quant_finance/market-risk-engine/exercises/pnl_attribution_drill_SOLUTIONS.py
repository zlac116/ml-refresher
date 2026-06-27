"""
DRILL 12 — P&L Attribution — SOLUTION KEY
Run: uv run python pnl_attribution_drill_SOLUTIONS.py
"""
import numpy as np
from dataclasses import replace
from extension_drill2_SOLUTIONS import (
    Market, Scenario, Portfolio, Bond, InterestRateSwap, CreditDefaultSwap,
    TotalReturnSwap, ForeignBond, Swaption, Repo, SecLending, Loan, FXSpot, FXOption)

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

    def sensitivities(self, base):
        v0 = self.pf.value(base)
        return (self.pf.value(replace(base, rate_bump_bp=base.rate_bump_bp+1)) - v0,
                self.pf.value(replace(base, spread_bump_bp=base.spread_bump_bp+1)) - v0,
                (self.pf.value(replace(base, vol_shock=base.vol_shock+0.01)) - v0)/0.01,
                (self.pf.value(replace(base, fx_shock=base.fx_shock+0.01)) - v0)/0.01)

    def gammas(self, base):
        v0 = self.pf.value(base)
        gr = self.pf.value(replace(base, rate_bump_bp=base.rate_bump_bp+1)) - 2*v0 + \
             self.pf.value(replace(base, rate_bump_bp=base.rate_bump_bp-1))            # /1²
        gs = self.pf.value(replace(base, spread_bump_bp=base.spread_bump_bp+1)) - 2*v0 + \
             self.pf.value(replace(base, spread_bump_bp=base.spread_bump_bp-1))        # /1²
        gv = (self.pf.value(replace(base, vol_shock=base.vol_shock+0.01)) - 2*v0 +
              self.pf.value(replace(base, vol_shock=base.vol_shock-0.01))) / 0.01**2
        gf = (self.pf.value(replace(base, fx_shock=base.fx_shock+0.01)) - 2*v0 +
              self.pf.value(replace(base, fx_shock=base.fx_shock-0.01))) / 0.01**2
        return gr, gs, gv, gf

    def attribute(self, base, scenario):
        sr, ss, sv, sf = self.sensitivities(base)
        gr, gs, gv, gf = self.gammas(base)
        dr, dsp, dv, dfx = scenario.d_rate_bp, scenario.d_spread_bp, scenario.d_vol, scenario.fx_ret
        rate, spread, vol, fx = sr*dr, ss*dsp, sv*dv, sf*dfx
        first = rate + spread + vol + fx
        second = 0.5*(gr*dr**2 + gs*dsp**2 + gv*dv**2 + gf*dfx**2)
        full = self.pf.value(scenario.apply(base)) - self.pf.value(base)
        return {"rate": rate, "spread": spread, "vol": vol, "fx": fx,
                "first_order": first, "second_order": second, "full": full,
                "unexplained": full - first - second}


if __name__ == "__main__":
    attr = PnLAttributor(PF); base = Market()
    gr, gs, gv, gf = attr.gammas(base)
    assert abs(gr - 1.855e-4) < 1e-6 and abs(gs - 4.287e-5) < 1e-7
    assert abs(gv - -1.0636217) < 1e-5 and abs(gf - 412.9211975) < 1e-4
    a = attr.attribute(base, Scenario(15.0, 10.0, 0.02, 0.0))
    assert abs(a["rate"] - -2.9376858) < 1e-5 and abs(a["spread"] - -1.2260855) < 1e-5
    assert abs(a["vol"] - 1.1094981) < 1e-5 and abs(a["fx"]) < 1e-9
    assert abs(a["full"] - -3.0238731) < 1e-5 and abs(a["first_order"] - -3.0542732) < 1e-5
    assert abs(a["second_order"] - 0.0228052) < 1e-6 and abs(a["unexplained"] - 0.0075949) < 1e-6
    print("✓ Solution key — all checks passed.")
