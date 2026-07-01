"""
DRILL 10 — Extend the engine (IRS · CDS · TRS · FX · ES)  — SOLUTION KEY
Run: uv run python extension_drill_SOLUTIONS.py
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]

def interp(x, xs, ys):
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i-1]) / (xs[i] - xs[i-1]); return ys[i-1] + w*(ys[i]-ys[i-1])


@dataclass(frozen=True)
class Market:
    rate_bump_bp: float = 0.0
    spread_bump_bp: float = 0.0
    fx_shock: float = 0.0
    def df(self, t: float, spread_bp: float = 0.0) -> float:
        z = interp(t, CURVE_T, CURVE_Z) + self.rate_bump_bp / 1e4
        return np.exp(-(z + spread_bp / 1e4) * t)
    def fx_factor(self) -> float:
        return 1.0 + self.fx_shock


@dataclass(frozen=True)
class Scenario:
    d_rate_bp: float
    d_spread_bp: float
    fx_ret: float
    def apply(self, base: Market) -> Market:
        return Market(base.rate_bump_bp + self.d_rate_bp,
                      base.spread_bump_bp + self.d_spread_bp,
                      base.fx_shock + self.fx_ret)


class Instrument(ABC):
    @abstractmethod
    def revalue(self, market: Market) -> float: ...


@dataclass(frozen=True)
class Bond(Instrument):
    times: tuple[float, ...]
    cashflows: tuple[float, ...]
    spread_bp: float
    def revalue(self, market: Market) -> float:
        return float(np.sum([cf * market.df(t, self.spread_bp + market.spread_bump_bp)
                             for t, cf in zip(self.times, self.cashflows)]))


@dataclass(frozen=True)
class InterestRateSwap(Instrument):
    times: tuple[float, ...]
    tau: float
    fixed_rate: float
    notional: float
    receive_fixed: bool = True
    def revalue(self, market: Market) -> float:
        dfs = [market.df(t, 0.0) for t in self.times]
        fixed_pv = self.fixed_rate * self.tau * sum(dfs)
        float_pv = 1.0 - dfs[-1]
        val = (fixed_pv - float_pv) * self.notional
        return float(val if self.receive_fixed else -val)


@dataclass(frozen=True)
class CreditDefaultSwap(Instrument):
    times: tuple[float, ...]
    tau: float
    contractual_spread_bp: float
    spread_bp: float
    notional: float
    recovery: float = 0.4
    protection_buyer: bool = True
    def revalue(self, market: Market) -> float:
        lam = ((self.spread_bp + market.spread_bump_bp) / 1e4) / (1 - self.recovery)
        prev_Q = 1.0
        premium = protection = 0.0
        for t in self.times:
            Q = np.exp(-lam * t)
            df = market.df(t, 0.0)
            premium += self.tau * df * Q
            protection += (prev_Q - Q) * df
            prev_Q = Q
        premium *= self.contractual_spread_bp / 1e4
        protection *= (1 - self.recovery)
        val = (protection - premium) * self.notional
        return float(val if self.protection_buyer else -val)


@dataclass(frozen=True)
class TotalReturnSwap(Instrument):
    reference: Instrument
    funding_notional: float
    receive_total_return: bool = True
    def revalue(self, market: Market) -> float:
        val = self.reference.revalue(market) - self.funding_notional
        return float(val if self.receive_total_return else -val)


@dataclass(frozen=True)
class ForeignBond(Instrument):
    times: tuple[float, ...]
    cashflows: tuple[float, ...]
    spread_bp: float
    fx0: float
    def revalue(self, market: Market) -> float:
        local = np.sum([cf * market.df(t, self.spread_bp + market.spread_bump_bp)
                        for t, cf in zip(self.times, self.cashflows)])
        return float(local * self.fx0 * market.fx_factor())


class Portfolio:
    def __init__(self, instruments: list[Instrument]):
        self._instruments = list(instruments)
    def value(self, market: Market) -> float:
        return float(np.sum([i.revalue(market) for i in self._instruments]))


class HistoricalVaREngine:
    def __init__(self, portfolio: Portfolio, scenarios: list[Scenario]):
        self.portfolio = portfolio
        self.scenarios = scenarios
    def pnl_vector(self, base: Market) -> list[float]:
        v0 = self.portfolio.value(base)
        return [self.portfolio.value(s.apply(base)) - v0 for s in self.scenarios]
    def var(self, base: Market, conf: float = 0.995) -> float:
        p = sorted(self.pnl_vector(base))
        return -p[int((1 - conf) * len(p))]
    def expected_shortfall(self, base: Market, conf: float = 0.995) -> float:
        p = sorted(self.pnl_vector(base))
        k = int((1 - conf) * len(p)) + 1
        return -float(np.mean(p[:k]))


if __name__ == "__main__":
    bond = Bond((1,2,3,4,5),(4,4,4,4,104),150.0)
    irs  = InterestRateSwap((1,2,3,4,5),1.0,0.042,100.0)
    cds  = CreditDefaultSwap((1,2,3,4,5),1.0,100.0,120.0,100.0)
    trs  = TotalReturnSwap(Bond((1,2,3,4,5),(4,4,4,4,104),150.0),90.0)
    fbond = ForeignBond((1,2,3,4,5),(3,3,3,3,103),100.0,1.25)
    base = Market()
    assert abs(bond.revalue(base)  -  92.0706468) < 1e-5
    assert abs(irs.revalue(base)   -  -0.4406536) < 1e-5
    assert abs(cds.revalue(base)   -   0.8816087) < 1e-5
    assert abs(trs.revalue(base)   -   2.0706468) < 1e-5
    assert abs(fbond.revalue(base) - 112.4267417) < 1e-5
    pf = Portfolio([bond, irs, cds, trs, fbond])
    assert abs(pf.value(base) - 207.0089904) < 1e-5
    N = 250
    scen = [Scenario(8*np.sin(i*0.3)+((i%11)-5), 5*np.cos(i*0.2)+((i%7)-3),
                     0.01*np.sin(i*0.5)+0.002*((i%5)-2)) for i in range(N)]
    eng = HistoricalVaREngine(pf, scen)
    pnls = eng.pnl_vector(base)
    assert abs(min(pnls)-(-2.8956889)) < 1e-5 and abs(max(pnls)-4.1878866) < 1e-5
    assert abs(eng.var(base,0.995) - 2.7015819) < 1e-5
    assert abs(eng.expected_shortfall(base,0.995) - 2.7986354) < 1e-5
    assert abs(eng.var(base,0.99) - 2.6660891) < 1e-5
    print("✓ Solution key — all checks passed.")
