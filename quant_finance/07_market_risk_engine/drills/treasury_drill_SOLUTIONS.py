"""
DRILL 14 — Treasury & inflation products + basis swaps — SOLUTION KEY
Run: uv run python treasury_drill_SOLUTIONS.py
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]
def interp(x, xs, ys): return float(np.interp(x, xs, ys))

@dataclass(frozen=True)
class Market:
    rate_bump_bp: float = 0.0; infl_shock_bp: float = 0.0; fx_shock: float = 0.0
    tenor_basis_bp: float = 0.0; xccy_basis_bp: float = 0.0
    def df(self, t): return np.exp(-(interp(t, CURVE_T, CURVE_Z) + self.rate_bump_bp/1e4) * t)
    def fx_factor(self): return 1.0 + self.fx_shock

@dataclass(frozen=True)
class Scenario:
    d_rate_bp: float; d_infl_bp: float; fx_ret: float; d_tenor_bp: float; d_xccy_bp: float
    def apply(self, b):
        return Market(b.rate_bump_bp+self.d_rate_bp, b.infl_shock_bp+self.d_infl_bp, b.fx_shock+self.fx_ret,
                      b.tenor_basis_bp+self.d_tenor_bp, b.xccy_basis_bp+self.d_xccy_bp)

class Instrument(ABC):
    @abstractmethod
    def revalue(self, market: Market) -> float: ...

@dataclass(frozen=True)
class FXForward(Instrument):
    foreign_notional: float; fx0: float; strike: float; T: float
    def revalue(self, m):
        return float(self.foreign_notional * (self.fx0*m.fx_factor() - self.strike) * m.df(self.T))

@dataclass(frozen=True)
class FRA(Instrument):
    notional: float; T1: float; T2: float; tau: float; strike: float
    def revalue(self, m):
        fwd = (m.df(self.T1)/m.df(self.T2) - 1) / self.tau
        return float(self.notional * (fwd - self.strike) * self.tau * m.df(self.T2))

@dataclass(frozen=True)
class InflationSwap(Instrument):
    notional: float; T: float; breakeven: float; strike: float
    def revalue(self, m):
        b = self.breakeven + m.infl_shock_bp/1e4
        return float(self.notional * m.df(self.T) * ((1+b)**self.T - (1+self.strike)**self.T))

@dataclass(frozen=True)
class Linker(Instrument):
    times: tuple[float, ...]; real_cfs: tuple[float, ...]; breakeven: float
    def revalue(self, m):
        b = self.breakeven + m.infl_shock_bp/1e4
        return float(sum(cf*(1+b)**t*m.df(t) for t, cf in zip(self.times, self.real_cfs)))

@dataclass(frozen=True)
class AnnuityLiability(Instrument):
    payment: float; years: int; breakeven: float
    def revalue(self, m):
        b = self.breakeven + m.infl_shock_bp/1e4
        return float(-sum(self.payment*(1+b)**t*m.df(t) for t in range(1, self.years+1)))

@dataclass(frozen=True)
class TenorBasisSwap(Instrument):
    notional: float; times: tuple[float, ...]; tau: float; basis_bp: float
    def revalue(self, m):
        b = (self.basis_bp + m.tenor_basis_bp) / 1e4
        return float(b * self.notional * sum(self.tau*m.df(t) for t in self.times))

@dataclass(frozen=True)
class XCCYBasisSwap(Instrument):
    foreign_notional: float; fx0: float; times: tuple[float, ...]; tau: float; basis_bp: float
    def revalue(self, m):
        b = (self.basis_bp + m.xccy_basis_bp) / 1e4
        fx = self.fx0 * m.fx_factor()
        return float(self.foreign_notional * fx * b * sum(self.tau*m.df(t) for t in self.times))

class Portfolio:
    def __init__(self, instruments): self._instruments = list(instruments)
    def value(self, m): return float(np.sum([i.revalue(m) for i in self._instruments]))


if __name__ == "__main__":
    fxf = FXForward(100.0,1.25,1.28,1.0); fra = FRA(100.0,1.0,2.0,1.0,0.043)
    infl = InflationSwap(100.0,5.0,0.03,0.028); lk = Linker((1,2,3,4,5),(1.5,1.5,1.5,1.5,101.5),0.03)
    ann = AnnuityLiability(5.0,10,0.03); tbs = TenorBasisSwap(100.0,(1,2,3,4,5),1.0,15.0)
    xccy = XCCYBasisSwap(100.0,1.25,(1,2,3,4,5),1.0,-20.0); base = Market()
    assert abs(fxf.revalue(base) - -2.8679924) < 1e-5
    assert abs(fra.revalue(base) - 0.0858887) < 1e-5
    assert abs(infl.revalue(base) - 0.9087836) < 1e-5
    assert abs(lk.revalue(base) - 101.1778224) < 1e-5
    assert abs(ann.revalue(base) - -46.7899884) < 1e-5
    assert abs(tbs.revalue(base) - 0.6607472) < 1e-5
    assert abs(xccy.revalue(base) - -1.1012453) < 1e-5
    pf = Portfolio([fxf, fra, infl, lk, ann, tbs, xccy]); V0 = pf.value(base)
    assert abs(V0 - 52.0740157) < 1e-5
    assert abs((pf.value(replace(base, rate_bump_bp=1)) - V0) - -0.0143105) < 1e-6
    assert abs((pf.value(replace(base, infl_shock_bp=1)) - V0) - 0.0687494) < 1e-6
    assert abs((pf.value(replace(base, fx_shock=0.01)) - V0)/0.01 - 118.3984399) < 1e-4
    assert abs((pf.value(replace(base, tenor_basis_bp=1)) - V0) - 0.0440498) < 1e-6
    assert abs((pf.value(replace(base, xccy_basis_bp=1)) - V0) - 0.0550623) < 1e-6
    print("✓ Solution key — all checks passed.")
