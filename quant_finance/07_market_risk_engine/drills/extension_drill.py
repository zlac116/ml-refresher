"""
DRILL 10 — Extend the VaR engine: IRS · CDS · TRS · a new FX risk factor · ES
============================================================================

OBJECTIVE
    Prove the design scales. WITHOUT touching Market/Scenario/Portfolio/Engine's
    core loop, add three instruments and a new risk factor + risk measure:
      TASK 1  InterestRateSwap.revalue      (rates only; OIS discounting)
      TASK 2  CreditDefaultSwap.revalue     (credit spread -> hazard/survival)
      TASK 3  TotalReturnSwap.revalue       (COMPOSITION: wraps a Bond)
      TASK 4  ForeignBond.revalue           (NEW risk factor: FX, a RELATIVE shock)
      TASK 5  Engine.expected_shortfall     (NEW measure: mean of the tail)

WHAT THIS SHOWS
    • A new instrument = a new class implementing revalue(); Portfolio/Engine
      never change (open/closed).
    • A new RISK FACTOR (FX) = a field on Market + the instruments that use it.
      Note FX is a RELATIVE shock (×(1+fx)), unlike rates/spreads (absolute bp).
    • Instruments react DIFFERENTLY to the same factor: a Bond LOSES when spreads
      widen; a protection-buyer CDS GAINS -> partial hedge. Watch the P&L.
    • A new measure (ES) = one method on Engine.

RUN
    uv run python extension_drill.py        (stuck? extension_drill_SOLUTIONS.py)

EXPECTED OUTPUT (base values)
    bond 92.0706 | irs -0.4407 | cds 0.8816 | trs 2.0706 | foreignBond 112.4267
    portfolio 207.0090
    P&L min/max  -2.8957 / 4.1879
    99.5% VaR 2.7016 | 99.5% ES 2.7986 | 99.0% VaR 2.6661

GRADING
    All asserts must pass.
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ── GIVEN: curve + helpers (do not change) ──────────────────────────────────
CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]

def interp(x, xs, ys):
    # if x <= xs[0]:  return ys[0]
    # if x >= xs[-1]: return ys[-1]
    # for i in range(1, len(xs)):
    #     if x <= xs[i]:
    #         w = (x - xs[i-1]) / (xs[i] - xs[i-1]); return ys[i-1] + w*(ys[i]-ys[i-1])
    return np.interp(x, xs, ys)


# ── GIVEN: Market now carries a THIRD risk factor — FX (relative) ───────────
@dataclass(frozen=True)
class Market:
    rate_bump_bp: float = 0.0      # absolute bp shift to the curve
    spread_bump_bp: float = 0.0    # absolute bp shift to credit spread
    fx_shock: float = 0.0          # RELATIVE FX return (e.g. 0.02 = +2%)
    def df(self, t: float, spread_bp: float = 0.0) -> float:
        z = interp(t, CURVE_T, CURVE_Z) + self.rate_bump_bp / 1e4
        return np.exp(-(z + spread_bp / 1e4) * t)
    def fx_factor(self) -> float:
        return 1.0 + self.fx_shock   # multiply a foreign value by this


# ── GIVEN: Scenario now shocks rates+spread (absolute) AND fx (relative) ─────
@dataclass(frozen=True)
class Scenario:
    d_rate_bp: float
    d_spread_bp: float
    fx_ret: float
    def apply(self, base: Market) -> Market:
        return Market(base.rate_bump_bp + self.d_rate_bp,
                      base.spread_bump_bp + self.d_spread_bp,
                      base.fx_shock + self.fx_ret)


# ── GIVEN: the interface + a WORKING Bond to copy the pattern from ──────────
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


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1 — Interest Rate Swap (receive-fixed value; payer = opposite sign)
#   Discount on the risk-free curve: market.df(t, 0).
#   fixed_pv = fixed_rate · tau · Σ DF(tᵢ)
#   float_pv = 1 − DF(t_last)                      (per unit notional, starts today)
#   value    = (fixed_pv − float_pv) · notional    (negate if not receive_fixed)
#   EXPECTED: irs.revalue(base) = -0.4407
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class InterestRateSwap(Instrument):
    times: tuple[float, ...]
    tau: float
    fixed_rate: float
    notional: float
    receive_fixed: bool = True
    def revalue(self, market: Market) -> float:
        dfs = np.array([market.df(t) for t in self.times])
        fixed_pv = self.fixed_rate * np.sum(self.tau * dfs)
        float_pv = 1 - dfs[-1]
        value = (fixed_pv - float_pv) * self.notional
        return value if self.receive_fixed else -value


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2 — Credit Default Swap (protection-buyer value)
#   market_spread = (spread_bp + market.spread_bump_bp)/1e4
#   hazard  λ = market_spread / (1 − recovery)
#   survival Q(t) = exp(−λ·t)        (loop; keep previous Q for the default leg)
#   premium    = contractual_spread/1e4 · Σ tau · DF(tᵢ) · Q(tᵢ)     (risky annuity)
#   protection = (1 − recovery)       · Σ [Q(tᵢ₋₁) − Q(tᵢ)] · DF(tᵢ)
#   value(buyer) = (protection − premium) · notional   (negate if seller)
#   NOTE: a CDS protection-buyer GAINS when spreads widen — opposite to a bond.
#   EXPECTED: cds.revalue(base) = 0.8816
# ═══════════════════════════════════════════════════════════════════════════
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
        market_spread = (self.spread_bp + market.spread_bump_bp) / 1e4
        hazard = market_spread / (1 - self.recovery)
        Q = lambda t: np.exp(-hazard * t)
        risky_annuity = np.sum([self.tau * market.df(t) * Q(t) for t in self.times])
        premium = (self.contractual_spread_bp / 1e4) * risky_annuity
        protection = 0.0
        prev_Q = 1.0
        for t in self.times:
            Qt = Q(t)
            protection += market.df(t) * (prev_Q - Qt)
            prev_Q = Qt

        protection = (1 - self.recovery) * protection
        value = (protection - premium) * self.notional
        return value if self.protection_buyer else -value


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3 — Total Return Swap (COMPOSITION: it HOLDS a reference instrument)
#   value(receiver) = reference.revalue(market) − funding_notional
#   (funding leg approximated as par/rate-insensitive). Negate if payer.
#   -> the TRS inherits the reference's rate + spread risk for free.
#   EXPECTED: trs.revalue(base) = 2.0706
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TotalReturnSwap(Instrument):
    reference: Instrument
    funding_notional: float
    receive_total_return: bool = True
    def revalue(self, market: Market) -> float:
        ref_pv = self.reference.revalue(market)
        value = ref_pv - self.funding_notional 
        return value if self.receive_total_return else -value


# ═══════════════════════════════════════════════════════════════════════════
# TASK 4 — Foreign-currency Bond  (consumes the NEW FX risk factor)
#   local = Σ CFₜ · market.df(t, spread_bp + market.spread_bump_bp)
#   base-ccy value = local · fx0 · market.fx_factor()     # ×(1 + fx_shock)
#   -> responds to rates, spread AND fx (a relative shock).
#   EXPECTED: foreignBond.revalue(base) = 112.4267
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ForeignBond(Instrument):
    times: tuple[float, ...]
    cashflows: tuple[float, ...]
    spread_bp: float
    fx0: float                      # base-ccy per 1 unit foreign, at t0
    def revalue(self, market: Market) -> float:
        dfs = [market.df(t, self.spread_bp + market.spread_bump_bp) for t in self.times]
        local_pv = np.sum(self.cashflows * np.array(dfs))
        value = local_pv * self.fx0 * market.fx_factor()
        return value


# ── GIVEN: Portfolio + Engine (var works; you add ES) ───────────────────────
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
        return -p[int((1 - conf) * len(p))]  # -np.percentile(p, (1 - conf)*100)

    # ═══════════════════════════════════════════════════════════════════════
    # TASK 5 — Expected Shortfall (average loss in the tail)
    #   sort P&Ls ascending; k = floor((1−conf)·N) + 1;  ES = −mean(sorted[:k])
    #   EXPECTED: ES(99.5%) = 2.7986
    # ═══════════════════════════════════════════════════════════════════════
    def expected_shortfall(self, base: Market, conf: float = 0.995) -> float:
        pnls_sorted = sorted(self.pnl_vector(base))
        k = int(len(pnls_sorted)*(1-conf))
        return -np.mean(pnls_sorted[:k+1])


# ═══════════════════════════════════════════════════════════════════════════
# GRADING
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    bond = Bond((1, 2, 3, 4, 5), (4, 4, 4, 4, 104), 150.0)
    irs  = InterestRateSwap((1, 2, 3, 4, 5), 1.0, 0.042, 100.0)
    cds  = CreditDefaultSwap((1, 2, 3, 4, 5), 1.0, 100.0, 120.0, 100.0)
    trs  = TotalReturnSwap(Bond((1, 2, 3, 4, 5), (4, 4, 4, 4, 104), 150.0), 90.0)
    fbond = ForeignBond((1, 2, 3, 4, 5), (3, 3, 3, 3, 103), 100.0, 1.25)
    base = Market()
    test = fbond.revalue(base)

    # per-instrument base values (debug one at a time)
    assert abs(bond.revalue(base)  -  92.0706468) < 1e-5, bond.revalue(base)
    assert abs(irs.revalue(base)   -  -0.4406536) < 1e-5, irs.revalue(base)
    assert abs(cds.revalue(base)   -   0.8816087) < 1e-5, cds.revalue(base)
    assert abs(trs.revalue(base)   -   2.0706468) < 1e-5, trs.revalue(base)
    assert abs(fbond.revalue(base) - 112.4267417) < 1e-5, fbond.revalue(base)

    portfolio = Portfolio([bond, irs, cds, trs, fbond])
    assert abs(portfolio.value(base) - 207.0089904) < 1e-5

    N = 250
    scenarios = [Scenario(8*np.sin(i*0.3) + ((i % 11)-5),
                          5*np.cos(i*0.2) + ((i % 7)-3),
                          0.01*np.sin(i*0.5) + 0.002*((i % 5)-2)) for i in range(N)]
    engine = HistoricalVaREngine(portfolio, scenarios)
    pnls = engine.pnl_vector(base)
    assert abs(min(pnls) - (-2.8956889)) < 1e-5 and abs(max(pnls) - 4.1878866) < 1e-5
    assert abs(engine.var(base, 0.995) - 2.7015819) < 1e-5
    assert abs(engine.expected_shortfall(base, 0.995) - 2.7986354) < 1e-5
    assert abs(engine.var(base, 0.99)  - 2.6660891) < 1e-5
    assert engine.expected_shortfall(base, 0.995) >= engine.var(base, 0.995)  # ES ≥ VaR always

    print(f"bond {bond.revalue(base):.4f} | irs {irs.revalue(base):.4f} | "
          f"cds {cds.revalue(base):.4f} | trs {trs.revalue(base):.4f} | "
          f"foreignBond {fbond.revalue(base):.4f}")
    print(f"portfolio value:  {portfolio.value(base):.4f}")
    print(f"P&L min / max:    {min(pnls):.4f} / {max(pnls):.4f}")
    print(f"99.5% VaR: {engine.var(base,0.995):.4f} | "
          f"99.5% ES: {engine.expected_shortfall(base,0.995):.4f} | "
          f"99.0% VaR: {engine.var(base,0.99):.4f}")
    print("\n✓ All checks passed.")
