"""
DRILL 11 (CAPSTONE) — define revalue() for ALL 11 products + sensitivity VaR
===========================================================================

Every product from DRILLs 8–10 PLUS the new ones, all in one engine. FOUR risk
factors: rate, spread, VOL (new), FX. You implement revalue() for each, then a
SENSITIVITY-BASED VaR beside full revaluation.

  TASK 1  Bond                 (rate + spread)            [from earlier drills]
  TASK 2  InterestRateSwap     (rate)
  TASK 3  CreditDefaultSwap    (spread -> hazard/survival)
  TASK 4  TotalReturnSwap      (composition: wraps a Bond)
  TASK 5  ForeignBond          (rate + spread + FX)
  TASK 6  Swaption             (rate + VOL; Black-76)      [new]
  TASK 7  Repo                 (rate)
  TASK 8  SecLending           (rate; fee income)
  TASK 9  Loan                 (rate + spread; amortising)
  TASK 10 FXSpot               (FX)
  TASK 11 FXOption             (FX + VOL + rate; Black-Scholes)
  TASK 12 Engine.sensitivities + estimate_pnl  (first-order Greeks)

WHY: cement the pattern — a new product is a class with revalue(); a new factor
is a Market field + the instruments using it. FULL REVAL (exact, convex) vs
SENSITIVITY (fast, first-order) — compare the two VaRs; options make them diverge.

RUN
    uv run python extension_drill2.py        (stuck? extension_drill2_SOLUTIONS.py)

EXPECTED (base values)
    bond 92.0706 | irs -0.4407 | cds 0.8816 | trs 2.0706 | foreignBond 112.4267
    swaption 1.1297 | repo -0.1227 | secLending 0.5501 | loan 96.1183 | fxSpot 100.0 | fxOption 6.3304
    portfolio 411.0148
    sensitivities: rate/1bp -0.195846 | spread/1bp -0.122609 | vol/1.0 55.4749 | fx/1.0 282.3615
    full 99.5% VaR 4.678103   vs   sensitivity 99.5% VaR 4.744942

GRADING: all asserts must pass.
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from scipy.special import erf

CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]
def interp(x, xs, ys): return float(np.interp(x, xs, ys))
def N(x): return 0.5 * (1 + erf(x / np.sqrt(2)))          # standard normal CDF

@dataclass(frozen=True)
class Market:
    rate_bump_bp: float = 0.0      # absolute bp
    spread_bump_bp: float = 0.0    # absolute bp
    vol_shock: float = 0.0         # absolute vol (0.01 = +1 vol pt)
    fx_shock: float = 0.0          # RELATIVE fx return
    def df(self, t: float, spread_bp: float = 0.0) -> float:
        z = interp(t, CURVE_T, CURVE_Z) + self.rate_bump_bp / 1e4
        return np.exp(-(z + spread_bp / 1e4) * t)
    def fx_factor(self) -> float:
        return 1.0 + self.fx_shock

@dataclass(frozen=True)
class Scenario:
    d_rate_bp: float
    d_spread_bp: float
    d_vol: float
    fx_ret: float
    def apply(self, base: Market) -> Market:
        return Market(base.rate_bump_bp + self.d_rate_bp, base.spread_bump_bp + self.d_spread_bp,
                      base.vol_shock + self.d_vol, base.fx_shock + self.fx_ret)

class Instrument(ABC):
    @abstractmethod
    def revalue(self, market: Market) -> float: ...


# TASK 1 — Bond:  Σ CFₜ · DF(t, spread_bp + market.spread_bump_bp)            -> 92.0706
@dataclass(frozen=True)
class Bond(Instrument):
    times: tuple[float, ...]; cashflows: tuple[float, ...]; spread_bp: float
    def revalue(self, market: Market) -> float:
        dfs = [market.df(t, self.spread_bp + market.spread_bump_bp) for t in self.times]
        value = np.sum(np.array(self.cashflows) * np.array(dfs))
        return value

# TASK 2 — InterestRateSwap (receiver): (fixed_rate·tau·ΣDF − (1−DF_last))·notional   -> -0.4407
@dataclass(frozen=True)
class InterestRateSwap(Instrument):
    times: tuple[float, ...]; tau: float; fixed_rate: float; notional: float; receive_fixed: bool = True
    def revalue(self, market: Market) -> float:
        dfs = np.array([market.df(t) for t in self.times])
        float_pv = 1 - dfs[-1]
        fixed_pv = self.fixed_rate * np.sum(self.tau * dfs)
        value = self.notional * (fixed_pv - float_pv)
        return value if self.receive_fixed else -value

# TASK 3 — CreditDefaultSwap (buyer): λ=(spread+bump)/1e4/(1−R); Q=e^{−λt};
#   premium = c·Σ tau·DF·Q ; protection = (1−R)·Σ[Q_{i-1}−Q_i]·DF ; (prot−prem)·N   -> 0.8816
@dataclass(frozen=True)
class CreditDefaultSwap(Instrument):
    times: tuple[float, ...]; tau: float; contractual_spread_bp: float; spread_bp: float
    notional: float; recovery: float = 0.4; protection_buyer: bool = True
    def revalue(self, market: Market) -> float:
        market_spread = (self.spread_bp + market.spread_bump_bp) / 1e4
        hazard = market_spread / (1 - self.recovery)
        Q = lambda t: np.exp(-hazard * t)
        risky_annuity = np.sum([market.df(t) * Q(t) * self.tau for t in self.times])
        premium_leg = (self.contractual_spread_bp / 1e4) * risky_annuity
        protection_leg = 0.0
        Q_prev = 1.0
        for t in self.times:
            Qt = Q(t)
            protection_leg += market.df(t) * (Q_prev - Qt)
            Q_prev = Qt
        protection_leg *= (1 - self.recovery)
        value = self.notional * (protection_leg - premium_leg)
        return value if self.protection_buyer else -value


# TASK 4 — TotalReturnSwap: reference.revalue(market) − funding_notional               -> 2.0706
@dataclass(frozen=True)
class TotalReturnSwap(Instrument):
    reference: Instrument; funding_notional: float; receive_total_return: bool = True
    def revalue(self, market: Market) -> float:
        value = self.reference.revalue(market) - self.funding_notional
        return value if self.receive_total_return else -value

# TASK 5 — ForeignBond: (Σ CFₜ·DF(t, spread+bump)) · fx0 · market.fx_factor()          -> 112.4267
@dataclass(frozen=True)
class ForeignBond(Instrument):
    times: tuple[float, ...]; cashflows: tuple[float, ...]; spread_bp: float; fx0: float
    def revalue(self, market: Market) -> float:
        dfs = np.array([market.df(t, self.spread_bp + market.spread_bump_bp) for t in self.times])
        disc_cfs = dfs * self.cashflows * self.fx0 * market.fx_factor()
        return np.sum(disc_cfs)

# TASK 6 — Swaption (Black-76, VOL factor):
#   A=Σtau·DF(pay); S=(DF(expiry)−DF(pay_last))/A; v=vol+vol_shock; T=expiry
#   d1=[ln(S/K)+½v²T]/(v√T); d2=d1−v√T
#   payer=A·[S·N(d1)−K·N(d2)] ; receiver=A·[K·N(−d2)−S·N(−d1)] ; ×notional             -> 1.1297
@dataclass(frozen=True)
class Swaption(Instrument):
    expiry: float; pay_times: tuple[float, ...]; tau: float; strike: float; vol: float; notional: float; payer: bool = True
    def revalue(self, market: Market) -> float:
        dfs = np.array([market.df(t) for t in self.pay_times])
        annuity = np.sum(self.tau * dfs)
        S = (market.df(self.expiry) - dfs[-1]) / annuity
        K = self.strike
        sigma = self.vol + market.vol_shock
        T = self.expiry
        d1 = (np.log(S/K) + 0.5 * (sigma**2) * T ) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        unit_val = annuity * (S*N(d1) - K*N(d2) if self.payer else K*N(-d2) - S*N(-d1))
        return unit_val * self.notional


# TASK 7 — Repo (reverse/lender): cash·(1+repo_rate·maturity)·DF(maturity) − cash      -> -0.1227
@dataclass(frozen=True)
class Repo(Instrument):
    cash: float; repo_rate: float; maturity: float; is_reverse: bool = True
    def revalue(self, market: Market) -> float:
        repo_leg = self.cash * (1 + self.repo_rate * self.maturity) * market.df(self.maturity)
        value = repo_leg - self.cash if self.is_reverse else self.cash - repo_leg
        return value

# TASK 8 — SecLending: (fee_rate_bp/1e4)·notional·Σ tau·DF(tᵢ)                          -> 0.5501
@dataclass(frozen=True)
class SecLending(Instrument):
    notional: float; fee_rate_bp: float; times: tuple[float, ...]; tau: float
    def revalue(self, market: Market) -> float:
        ann = sum([market.df(t) * self.tau for t in self.times])
        return (self.fee_rate_bp / 1e4) * self.notional * ann

# TASK 9 — Loan (amortising): amort=principal/n; cfᵢ=amort+coupon·out; out−=amort;
#   value = Σ cfᵢ·DF(i, spread_bp + market.spread_bump_bp)                              -> 96.1183
@dataclass(frozen=True)
class Loan(Instrument):
    principal: float; coupon: float; n: int; spread_bp: float
    def revalue(self, market: Market) -> float:
        amort = self.principal / self.n
        out = self.principal
        pv = 0.0
        for t in range(1, self.n+1):
            cf = amort + self.coupon * out
            out -= amort
            pv += cf * market.df(t, self.spread_bp + market.spread_bump_bp)
        return float(pv)

# TASK 10 — FXSpot: foreign_amount · fx0 · market.fx_factor()                            -> 100.0
@dataclass(frozen=True)
class FXSpot(Instrument):
    foreign_amount: float; fx0: float
    def revalue(self, market: Market) -> float:
        return self.foreign_amount * self.fx0 * market.fx_factor()

# TASK 11 — FXOption (Black-Scholes on FX): S=fx0·fx_factor(); v=vol+vol_shock; T=expiry;
#   dfT=DF(T); F=S/dfT; d1=[ln(F/K)+½v²T]/(v√T); d2=d1−v√T
#   call=dfT·[F·N(d1)−K·N(d2)] ; put=dfT·[K·N(−d2)−F·N(−d1)] ; ×notional                 -> 6.3304
@dataclass(frozen=True)
class FXOption(Instrument):
    expiry: float; strike: float; vol: float; fx0: float; notional: float; call: bool = True
    def revalue(self, market: Market) -> float:
        S = self.fx0 * market.fx_factor()
        sigma = self.vol + market.vol_shock
        T = self.expiry
        dfT = market.df(T)
        F = S/dfT
        d1 = (np.log(F/self.strike) + 0.5 * (sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        unit_value = dfT*(F*N(d1) - self.strike*N(d2)) if self.call else dfT*(self.strike*N(-d2) - F*N(-d1))
        return unit_value * self.notional


# ── GIVEN: Portfolio + Engine (full reval given; you add the SENSITIVITY path)
class Portfolio:
    def __init__(self, instruments): self._instruments = list(instruments)
    def value(self, market: Market) -> float:
        return float(np.sum([i.revalue(market) for i in self._instruments]))

class HistoricalVaREngine:
    def __init__(self, portfolio, scenarios):
        self.portfolio = portfolio; self.scenarios = scenarios
    def pnl_vector(self, base):
        v0 = self.portfolio.value(base)
        return np.array([self.portfolio.value(s.apply(base)) - v0 for s in self.scenarios])

    def var(self, base, conf=0.995):
        p = sorted(self.pnl_vector(base))
        k = int(len(p)*(1-conf))
        return -p[k]

    # TASK 12 — sensitivity path (finite-difference bumps of the whole portfolio):
    #   s_rate   = V(rate +1bp) − V(base)             # replace(base, rate_bump_bp=base.rate_bump_bp+1)
    #   s_spread = V(spread +1bp) − V(base)
    #   s_vol    = [V(vol +0.01) − V(base)] / 0.01
    #   s_fx     = [V(fx +0.01)  − V(base)] / 0.01
    #   estimate_pnl: ΔV ≈ s_rate·d_rate_bp + s_spread·d_spread_bp + s_vol·d_vol + s_fx·fx_ret
    def sensitivities(self, base):
        v0 = self.portfolio.value(base)
        return (
            self.portfolio.value(replace(base, rate_bump_bp=base.rate_bump_bp + 1.0)) - v0,
            self.portfolio.value(replace(base, spread_bump_bp=base.spread_bump_bp + 1.0)) - v0,
            (self.portfolio.value(replace(base, vol_shock=base.vol_shock + 0.01)) - v0)*100,
            (self.portfolio.value(replace(base, fx_shock=base.fx_shock + 0.01)) - v0)*100,
        )
    
    def estimate_pnl(self, base):
        sr, ss, sv, sf = self.sensitivities(base)
        return np.array([s.d_rate_bp*sr + s.d_spread_bp*ss + s.d_vol*sv + s.fx_ret*sf for s in self.scenarios])

    def var_sensitivity(self, base, conf=0.995):
        p = sorted(self.estimate_pnl(base))
        return -p[int(len(p) * (1 - conf))]


if __name__ == "__main__":
    bond = Bond((1,2,3,4,5),(4,4,4,4,104),150.0)
    irs  = InterestRateSwap((1,2,3,4,5),1.0,0.042,100.0)
    cds  = CreditDefaultSwap((1,2,3,4,5),1.0,100.0,120.0,100.0)    
    trs  = TotalReturnSwap(Bond((1,2,3,4,5),(4,4,4,4,104),150.0),90.0)
    fbond = ForeignBond((1,2,3,4,5),(3,3,3,3,103),100.0,1.25)
    swp  = Swaption(2.0,(3,4,5),1.0,0.042,0.20,100.0)
    repo = Repo(100.0,0.043,0.5)
    secl = SecLending(100.0,20.0,(1,2,3),1.0)
    loan = Loan(100.0,0.05,5,200.0)
    fxs  = FXSpot(80.0,1.25)
    fxo  = FXOption(1.0,1.30,0.12,1.25,100.0)
    base = Market()

    checks = {"bond":(bond,92.0706468),"irs":(irs,-0.4406536),"cds":(cds,0.8816087),
              "trs":(trs,2.0706468),"foreignBond":(fbond,112.4267417),"swaption":(swp,1.1297436),
              "repo":(repo,-0.1227111),"secLending":(secl,0.5501465),"loan":(loan,96.1182606),
              "fxSpot":(fxs,100.0),"fxOption":(fxo,6.3303837)}
    for name,(inst,exp) in checks.items():
        assert abs(inst.revalue(base) - exp) < 1e-5, f"{name}: {inst.revalue(base)}"

    pf = Portfolio([bond,irs,cds,trs,fbond,swp,repo,secl,loan,fxs,fxo])
    assert abs(pf.value(base) - 411.0148137) < 1e-5
    
    Nsc = 250
    scen = [Scenario(8*np.sin(i*0.3)+((i%11)-5), 5*np.cos(i*0.2)+((i%7)-3),
                     0.005*np.sin(i*0.4)+0.001*((i%3)-1), 0.01*np.sin(i*0.5)+0.002*((i%5)-2)) for i in range(Nsc)]
    eng = HistoricalVaREngine(pf, scen)

    sr, ss, sv, sf = eng.sensitivities(base)
    assert abs(sr - -0.1958457) < 1e-6 and abs(ss - -0.1226085) < 1e-6
    assert abs(sv - 55.4749062) < 1e-4 and abs(sf - 282.3615132) < 1e-4

    pnls = eng.pnl_vector(base)
    assert abs(min(pnls) - (-4.9229221)) < 1e-5 and abs(max(pnls) - 6.8355020) < 1e-5
    full = eng.var(base, 0.995); sens = eng.var_sensitivity(base, 0.995)
    assert abs(full - 4.6781031) < 1e-5, full
    assert abs(sens - 4.7449419) < 1e-5, sens

    print("base values:")
    for name,(inst,_) in checks.items():
        print(f"  {name:12s} {inst.revalue(base):.4f}")
    print(f"portfolio: {pf.value(base):.4f}")
    print(f"sensitivities: rate/1bp {sr:.6f} | spread/1bp {ss:.6f} | vol/1.0 {sv:.4f} | fx/1.0 {sf:.4f}")
    print(f"full 99.5% VaR {full:.6f}  vs  sensitivity 99.5% VaR {sens:.6f}  (Δ {full-sens:+.6f})")
    print("\n✓ All checks passed.")
