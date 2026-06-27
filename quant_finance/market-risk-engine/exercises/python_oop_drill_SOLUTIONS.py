"""
DRILL 9 — OOP refactor of the VaR engine  — SOLUTION KEY
Run: uv run python python_oop_drill_SOLUTIONS.py
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
    def df(self, t: float, spread_bp: float) -> float:
        z = interp(t, CURVE_T, CURVE_Z) + self.rate_bump_bp / 1e4
        return np.exp(-(z + spread_bp / 1e4) * t)


@dataclass(frozen=True)
class Scenario:
    d_rate_bp: float
    d_spread_bp: float
    def apply(self, base: Market) -> Market:
        return Market(base.rate_bump_bp + self.d_rate_bp,
                      base.spread_bump_bp + self.d_spread_bp)


class Instrument(ABC):
    @abstractmethod
    def revalue(self, market: Market) -> float: ...


@dataclass(frozen=True)
class Bond(Instrument):
    times: tuple[float, ...]
    cashflows: tuple[float, ...]
    spread_bp: float
    def revalue(self, market: Market) -> float:
        s = self.spread_bp + market.spread_bump_bp
        return sum(cf * market.df(t, s) for t, cf in zip(self.times, self.cashflows))


class Portfolio:
    def __init__(self, instruments: list[Instrument]):
        self._instruments = list(instruments)
    def value(self, market: Market) -> float:
        return sum(inst.revalue(market) for inst in self._instruments)


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


if __name__ == "__main__":
    from dataclasses import FrozenInstanceError
    N = 250
    scenarios = [Scenario(8*np.sin(i*0.3) + ((i % 11)-5),
                          5*np.cos(i*0.2) + ((i % 7)-3)) for i in range(N)]
    engine = HistoricalVaREngine(Portfolio([Bond((1,2,3,4,5),(4,4,4,4,104),150.0)]), scenarios)
    base = Market()
    assert abs(engine.portfolio.value(base) - 92.0706468) < 1e-5
    pnls = engine.pnl_vector(base)
    assert abs(min(pnls)-(-0.8011423)) < 1e-5 and abs(max(pnls)-0.8504003) < 1e-5
    assert abs(engine.var(base, 0.995) - 0.7362917) < 1e-5
    assert abs(engine.var(base, 0.99)  - 0.7032813) < 1e-5
    assert issubclass(Bond, Instrument)
    try: Instrument(); raise AssertionError
    except TypeError: pass
    try: base.rate_bump_bp = 5; raise AssertionError
    except FrozenInstanceError: pass
    print("✓ Solution key — all checks passed.")
