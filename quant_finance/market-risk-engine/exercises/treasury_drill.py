"""
DRILL 14 — Treasury & inflation products + basis swaps + the annuity liability
==============================================================================

Treasury/ALM instruments and FIVE risk factors: rate, INFLATION (new),
fx, TENOR BASIS (new), XCCY BASIS (new) — plus the ANNUITY LIABILITY, so you
stress ASSETS and LIABILITIES together. You implement every revalue().

  TASK 1  FXForward         (fx + rate)
  TASK 2  FRA               (rate; forward rate)
  TASK 3  InflationSwap     (inflation; zero-coupon)
  TASK 4  Linker            (inflation-linked bond: rate + inflation)
  TASK 5  AnnuityLiability  (rate + inflation; NEGATIVE = liability)
  TASK 6  TenorBasisSwap    (rate + tenor-basis: e.g. 3M-vs-6M spread)
  TASK 7  XCCYBasisSwap     (rate + fx + cross-currency-basis spread)

KEY IDEAS
  • New factor pattern: instrument carries the base level, Market carries the SHOCK
    (infl_shock_bp, tenor_basis_bp, xccy_basis_bp) — same as spread_bump earlier.
  • IE01 = ΔV per 1bp breakeven inflation; tenor-/xccy-01 = ΔV per 1bp of that basis.
  • A basis swap's value ≈ the PV of the basis-spread leg (basis · notional · Σ τ·DF);
    the xccy one is also fx-sensitive (it's in a foreign currency).
  • The annuity liability is NEGATIVE: rates down or inflation up → liability bigger.

RUN
    uv run python treasury_drill.py        (stuck? treasury_drill_SOLUTIONS.py)

EXPECTED (base)
    fxForward -2.8680 | fra 0.0859 | inflSwap 0.9088 | linker 101.1778 | annuity -46.7900
    tenorBasis 0.6607 | xccyBasis -1.1012 | portfolio 52.0740
    sens: DV01 -0.014310 | IE01 +0.068749 | fx 118.3984 | tenor01 +0.044050 | xccy01 +0.055062

GRADING: all asserts must pass.
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]
def interp(x, xs, ys): return float(np.interp(x, xs, ys))

@dataclass(frozen=True)
class Market:
    rate_bump_bp: float = 0.0     # nominal curve, bp
    infl_shock_bp: float = 0.0    # breakeven inflation, bp
    fx_shock: float = 0.0         # relative fx return
    tenor_basis_bp: float = 0.0   # tenor-basis spread, bp
    xccy_basis_bp: float = 0.0    # cross-currency-basis spread, bp
    def df(self, t: float) -> float:
        return np.exp(-(interp(t, CURVE_T, CURVE_Z) + self.rate_bump_bp / 1e4) * t)
    def fx_factor(self) -> float:
        return 1.0 + self.fx_shock

@dataclass(frozen=True)
class Scenario:
    d_rate_bp: float; d_infl_bp: float; fx_ret: float; d_tenor_bp: float; d_xccy_bp: float
    def apply(self, b: Market) -> Market:
        return Market(b.rate_bump_bp + self.d_rate_bp, b.infl_shock_bp + self.d_infl_bp,
                      b.fx_shock + self.fx_ret, b.tenor_basis_bp + self.d_tenor_bp,
                      b.xccy_basis_bp + self.d_xccy_bp)

class Instrument(ABC):
    @abstractmethod
    def revalue(self, market: Market) -> float: ...


# TASK 1 — FX Forward:  foreign_notional·(fx0·fx_factor() − strike)·DF(T)                -> -2.8680
@dataclass(frozen=True)
class FXForward(Instrument):
    foreign_notional: float; fx0: float; strike: float; T: float
    def revalue(self, market): raise NotImplementedError

# TASK 2 — FRA:  fwd=(DF(T1)/DF(T2)−1)/tau ; notional·(fwd−strike)·tau·DF(T2)            -> 0.0859
@dataclass(frozen=True)
class FRA(Instrument):
    notional: float; T1: float; T2: float; tau: float; strike: float
    def revalue(self, market): raise NotImplementedError

# TASK 3 — Inflation Swap (ZC receiver): b=breakeven+infl_shock_bp/1e4 ;
#   notional·DF(T)·((1+b)^T − (1+strike)^T)                                              -> 0.9088
@dataclass(frozen=True)
class InflationSwap(Instrument):
    notional: float; T: float; breakeven: float; strike: float
    def revalue(self, market): raise NotImplementedError

# TASK 4 — Linker:  Σ real_cfₜ·(1+b)^t·DF(t)                                              -> 101.1778
@dataclass(frozen=True)
class Linker(Instrument):
    times: tuple[float, ...]; real_cfs: tuple[float, ...]; breakeven: float
    def revalue(self, market): raise NotImplementedError

# TASK 5 — Annuity Liability:  − Σ payment·(1+b)^t·DF(t),  t=1..years                    -> -46.7900
@dataclass(frozen=True)
class AnnuityLiability(Instrument):
    payment: float; years: int; breakeven: float
    def revalue(self, market): raise NotImplementedError

# TASK 6 — Tenor Basis Swap (receive the basis spread):
#   b = (basis_bp + market.tenor_basis_bp)/1e4 ;  value = b·notional·Σ tau·DF(t)         -> 0.6607
@dataclass(frozen=True)
class TenorBasisSwap(Instrument):
    notional: float; times: tuple[float, ...]; tau: float; basis_bp: float
    def revalue(self, market): raise NotImplementedError

# TASK 7 — XCCY Basis Swap (foreign-ccy basis leg; fx-sensitive):
#   b = (basis_bp + market.xccy_basis_bp)/1e4 ;  fx = fx0·fx_factor()
#   value = foreign_notional·fx·b·Σ tau·DF(t)   (principal exchange simplified out)      -> -1.1012
@dataclass(frozen=True)
class XCCYBasisSwap(Instrument):
    foreign_notional: float; fx0: float; times: tuple[float, ...]; tau: float; basis_bp: float
    def revalue(self, market): raise NotImplementedError


class Portfolio:
    def __init__(self, instruments): self._instruments = list(instruments)
    def value(self, market: Market) -> float:
        return float(np.sum([i.revalue(market) for i in self._instruments]))


# ═══════════════════════════════════════════════════════════════════════════
# GRADING
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fxf  = FXForward(100.0, 1.25, 1.28, 1.0)
    fra  = FRA(100.0, 1.0, 2.0, 1.0, 0.043)
    infl = InflationSwap(100.0, 5.0, 0.03, 0.028)
    lk   = Linker((1,2,3,4,5), (1.5,1.5,1.5,1.5,101.5), 0.03)
    ann  = AnnuityLiability(5.0, 10, 0.03)
    tbs  = TenorBasisSwap(100.0, (1,2,3,4,5), 1.0, 15.0)
    xccy = XCCYBasisSwap(100.0, 1.25, (1,2,3,4,5), 1.0, -20.0)
    base = Market()

    checks = {"fxForward":(fxf,-2.8679924),"fra":(fra,0.0858887),"inflSwap":(infl,0.9087836),
              "linker":(lk,101.1778224),"annuity":(ann,-46.7899884),"tenorBasis":(tbs,0.6607472),
              "xccyBasis":(xccy,-1.1012453)}
    for name,(inst,exp) in checks.items():
        assert abs(inst.revalue(base) - exp) < 1e-5, f"{name}: {inst.revalue(base)}"

    pf = Portfolio([fxf, fra, infl, lk, ann, tbs, xccy]); V0 = pf.value(base)
    assert abs(V0 - 52.0740157) < 1e-5

    dv01   = pf.value(replace(base, rate_bump_bp=1)) - V0
    ie01   = pf.value(replace(base, infl_shock_bp=1)) - V0
    fxs    = (pf.value(replace(base, fx_shock=0.01)) - V0) / 0.01
    tenor01= pf.value(replace(base, tenor_basis_bp=1)) - V0
    xccy01 = pf.value(replace(base, xccy_basis_bp=1)) - V0
    assert abs(dv01 - -0.0143105) < 1e-6 and abs(ie01 - 0.0687494) < 1e-6
    assert abs(fxs - 118.3984399) < 1e-4 and abs(tenor01 - 0.0440498) < 1e-6 and abs(xccy01 - 0.0550623) < 1e-6

    for name,(inst,_) in checks.items(): print(f"  {name:11s} {inst.revalue(base):.4f}")
    print(f"portfolio: {V0:.4f}")
    print(f"sens: DV01 {dv01:.6f} | IE01 {ie01:.6f} | fx {fxs:.4f} | tenor01 {tenor01:.6f} | xccy01 {xccy01:.6f}")
    print("\n✓ All checks passed.")
