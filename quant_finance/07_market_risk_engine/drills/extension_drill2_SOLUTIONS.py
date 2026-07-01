"""
DRILL 11 (CAPSTONE) — all 11 products + sensitivity VaR  — SOLUTION KEY
Run: uv run python extension_drill2_SOLUTIONS.py
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from scipy.special import erf

CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]
def interp(x, xs, ys): return float(np.interp(x, xs, ys))
def N(x): return 0.5 * (1 + erf(x / np.sqrt(2)))

@dataclass(frozen=True)
class Market:
    rate_bump_bp: float = 0.0; spread_bump_bp: float = 0.0; vol_shock: float = 0.0; fx_shock: float = 0.0
    def df(self, t, spread_bp=0.0):
        return np.exp(-(interp(t, CURVE_T, CURVE_Z) + self.rate_bump_bp/1e4 + spread_bp/1e4) * t)
    def fx_factor(self): return 1.0 + self.fx_shock

@dataclass(frozen=True)
class Scenario:
    d_rate_bp: float; d_spread_bp: float; d_vol: float; fx_ret: float
    def apply(self, b):
        return Market(b.rate_bump_bp+self.d_rate_bp, b.spread_bump_bp+self.d_spread_bp,
                      b.vol_shock+self.d_vol, b.fx_shock+self.fx_ret)

class Instrument(ABC):
    @abstractmethod
    def revalue(self, market: Market) -> float: ...

@dataclass(frozen=True)
class Bond(Instrument):
    times: tuple[float, ...]; cashflows: tuple[float, ...]; spread_bp: float
    def revalue(self, m):
        return float(np.sum([cf*m.df(t, self.spread_bp+m.spread_bump_bp) for t, cf in zip(self.times, self.cashflows)]))

@dataclass(frozen=True)
class InterestRateSwap(Instrument):
    times: tuple[float, ...]; tau: float; fixed_rate: float; notional: float; receive_fixed: bool = True
    def revalue(self, m):
        dfs = [m.df(t) for t in self.times]
        val = (self.fixed_rate*self.tau*sum(dfs) - (1 - dfs[-1])) * self.notional
        return float(val if self.receive_fixed else -val)

@dataclass(frozen=True)
class CreditDefaultSwap(Instrument):
    times: tuple[float, ...]; tau: float; contractual_spread_bp: float; spread_bp: float
    notional: float; recovery: float = 0.4; protection_buyer: bool = True
    def revalue(self, m):
        lam = ((self.spread_bp + m.spread_bump_bp)/1e4) / (1 - self.recovery)
        prev_Q = 1.0; premium = protection = 0.0
        for t in self.times:
            Q = np.exp(-lam*t); df = m.df(t)
            premium += self.tau*df*Q; protection += (prev_Q - Q)*df; prev_Q = Q
        val = (protection*(1 - self.recovery) - premium*self.contractual_spread_bp/1e4) * self.notional
        return float(val if self.protection_buyer else -val)

@dataclass(frozen=True)
class TotalReturnSwap(Instrument):
    reference: Instrument; funding_notional: float; receive_total_return: bool = True
    def revalue(self, m):
        val = self.reference.revalue(m) - self.funding_notional
        return float(val if self.receive_total_return else -val)

@dataclass(frozen=True)
class ForeignBond(Instrument):
    times: tuple[float, ...]; cashflows: tuple[float, ...]; spread_bp: float; fx0: float
    def revalue(self, m):
        local = np.sum([cf*m.df(t, self.spread_bp+m.spread_bump_bp) for t, cf in zip(self.times, self.cashflows)])
        return float(local * self.fx0 * m.fx_factor())

@dataclass(frozen=True)
class Swaption(Instrument):
    expiry: float; pay_times: tuple[float, ...]; tau: float; strike: float; vol: float; notional: float; payer: bool = True
    def revalue(self, m):
        ann = sum(self.tau*m.df(t) for t in self.pay_times)
        S = (m.df(self.expiry) - m.df(self.pay_times[-1])) / ann
        v = self.vol + m.vol_shock; T = self.expiry
        d1 = (np.log(S/self.strike) + 0.5*v*v*T) / (v*np.sqrt(T)); d2 = d1 - v*np.sqrt(T)
        unit = (S*N(d1) - self.strike*N(d2)) if self.payer else (self.strike*N(-d2) - S*N(-d1))
        return float(ann * unit * self.notional)

@dataclass(frozen=True)
class Repo(Instrument):
    cash: float; repo_rate: float; maturity: float; is_reverse: bool = True
    def revalue(self, m):
        val = self.cash*(1 + self.repo_rate*self.maturity)*m.df(self.maturity) - self.cash
        return float(val if self.is_reverse else -val)

@dataclass(frozen=True)
class SecLending(Instrument):
    notional: float; fee_rate_bp: float; times: tuple[float, ...]; tau: float
    def revalue(self, m):
        return float((self.fee_rate_bp/1e4) * self.notional * sum(self.tau*m.df(t) for t in self.times))

@dataclass(frozen=True)
class Loan(Instrument):
    principal: float; coupon: float; n: int; spread_bp: float
    def revalue(self, m):
        amort = self.principal/self.n; out = self.principal; pv = 0.0
        for i in range(1, self.n+1):
            cf = amort + self.coupon*out; out -= amort
            pv += cf * m.df(i, self.spread_bp + m.spread_bump_bp)
        return float(pv)

@dataclass(frozen=True)
class FXSpot(Instrument):
    foreign_amount: float; fx0: float
    def revalue(self, m):
        return float(self.foreign_amount * self.fx0 * m.fx_factor())

@dataclass(frozen=True)
class FXOption(Instrument):
    expiry: float; strike: float; vol: float; fx0: float; notional: float; call: bool = True
    def revalue(self, m):
        S = self.fx0 * m.fx_factor(); T = self.expiry; v = self.vol + m.vol_shock; dfT = m.df(T)
        F = S / dfT
        d1 = (np.log(F/self.strike) + 0.5*v*v*T) / (v*np.sqrt(T)); d2 = d1 - v*np.sqrt(T)
        unit = dfT*(F*N(d1) - self.strike*N(d2)) if self.call else dfT*(self.strike*N(-d2) - F*N(-d1))
        return float(unit * self.notional)


class Portfolio:
    def __init__(self, instruments): self._instruments = list(instruments)
    def value(self, m): return float(np.sum([i.revalue(m) for i in self._instruments]))

class HistoricalVaREngine:
    def __init__(self, portfolio, scenarios): self.portfolio = portfolio; self.scenarios = scenarios
    def pnl_vector(self, base):
        v0 = self.portfolio.value(base)
        return [self.portfolio.value(s.apply(base)) - v0 for s in self.scenarios]
    def var(self, base, conf=0.995):
        p = sorted(self.pnl_vector(base)); return -p[int((1 - conf) * len(p))]
    def sensitivities(self, base):
        v0 = self.portfolio.value(base)
        return (self.portfolio.value(replace(base, rate_bump_bp=base.rate_bump_bp+1)) - v0,
                self.portfolio.value(replace(base, spread_bump_bp=base.spread_bump_bp+1)) - v0,
                (self.portfolio.value(replace(base, vol_shock=base.vol_shock+0.01)) - v0)/0.01,
                (self.portfolio.value(replace(base, fx_shock=base.fx_shock+0.01)) - v0)/0.01)
    def estimate_pnl(self, base):
        sr, ss, sv, sf = self.sensitivities(base)
        return [sr*s.d_rate_bp + ss*s.d_spread_bp + sv*s.d_vol + sf*s.fx_ret for s in self.scenarios]
    def var_sensitivity(self, base, conf=0.995):
        p = sorted(self.estimate_pnl(base)); return -p[int((1 - conf) * len(p))]


if __name__ == "__main__":
    bond = Bond((1,2,3,4,5),(4,4,4,4,104),150.0)
    pf = Portfolio([bond, InterestRateSwap((1,2,3,4,5),1.0,0.042,100.0),
                    CreditDefaultSwap((1,2,3,4,5),1.0,100.0,120.0,100.0),
                    TotalReturnSwap(Bond((1,2,3,4,5),(4,4,4,4,104),150.0),90.0),
                    ForeignBond((1,2,3,4,5),(3,3,3,3,103),100.0,1.25),
                    Swaption(2.0,(3,4,5),1.0,0.042,0.20,100.0),
                    Repo(100.0,0.043,0.5), SecLending(100.0,20.0,(1,2,3),1.0), Loan(100.0,0.05,5,200.0),
                    FXSpot(80.0,1.25), FXOption(1.0,1.30,0.12,1.25,100.0)])
    base = Market()
    assert abs(pf.value(base) - 411.0148137) < 1e-5
    scen = [Scenario(8*np.sin(i*0.3)+((i%11)-5), 5*np.cos(i*0.2)+((i%7)-3),
                     0.005*np.sin(i*0.4)+0.001*((i%3)-1), 0.01*np.sin(i*0.5)+0.002*((i%5)-2)) for i in range(250)]
    eng = HistoricalVaREngine(pf, scen)
    sr, ss, sv, sf = eng.sensitivities(base)
    assert abs(sr - -0.1958457) < 1e-6 and abs(ss - -0.1226085) < 1e-6
    assert abs(sv - 55.4749062) < 1e-4 and abs(sf - 282.3615132) < 1e-4
    assert abs(eng.var(base,0.995) - 4.6781031) < 1e-5
    assert abs(eng.var_sensitivity(base,0.995) - 4.7449419) < 1e-5
    print("✓ Solution key — all checks passed.")
