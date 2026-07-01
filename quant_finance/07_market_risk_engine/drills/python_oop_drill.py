"""
DRILL 9 — Refactor the VaR engine into clean OOP  (classes · best practices)
============================================================================

OBJECTIVE
    Re-express DRILL 8's functions as an idiomatic, extensible class design —
    the exact "data structures + classes + best practices, applied to our
    stress engine" answer. You implement the METHOD BODIES; the scaffolding
    (ABCs, dataclasses, type hints) is given so you absorb the design.

WHY THIS DESIGN (say these in interview)
    • Instrument is an ABC  -> every instrument MUST implement revalue();
      adding a Bond/CDS is a new class, the engine never changes (open/closed).
    • Market & Scenario are FROZEN dataclasses -> immutable value objects:
      auto __init__/__repr__/__eq__, and a scenario can't mutate the base market.
    • Scenario.apply() is PURE -> returns a NEW market, no side effects -> testable.
    • Portfolio/Engine use COMPOSITION (hold instruments) not inheritance.
    • Type hints throughout -> mypy-checkable, self-documenting.

RUN
    uv run python python_oop_drill.py        (stuck? python_oop_drill_SOLUTIONS.py)

EXPECTED OUTPUT
    portfolio value:   92.0706
    P&L min / max:    -0.8011 / 0.8504
    99.5% VaR:         0.7363    99.0% VaR: 0.7033
    + design asserts: ABC not instantiable, Market frozen, apply() pure

GRADING
    All asserts must pass.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ── GIVEN: curve + interpolator (do not change) ─────────────────────────────
CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]

def interp(x, xs, ys):
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i-1]) / (xs[i] - xs[i-1]); return ys[i-1] + w*(ys[i]-ys[i-1])


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1 — Market: an immutable value object that discounts cashflows
#   exp(-(z(t) + spread)·t), where z(t) = interp(...) + rate_bump_bp/1e4.
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Market:
    rate_bump_bp: float = 0.0
    spread_bump_bp: float = 0.0

    def df(self, t: float, spread_bp: float) -> float:
        """Discount factor: exp(-(zero_rate + spread)·t). spread_bp is bp."""
        z = interp(t, CURVE_T, CURVE_Z) + self.rate_bump_bp / 1e4
        return np.exp(-(z + spread_bp / 1e4) * t)


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2 — Scenario: a daily risk-factor move; apply() returns a NEW Market
#   (pure function — must not mutate `base`).
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Scenario:
    d_rate_bp: float
    d_spread_bp: float

    def apply(self, base: Market) -> Market:
        """Return a new Market with this scenario's bumps ADDED to base's."""
        return Market(base.rate_bump_bp + self.d_rate_bp,
                      base.spread_bump_bp + self.d_spread_bp)


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3 — Instrument interface (ABC) + a Bond that implements it
#   Bond.revalue uses its own spread + the market's spread bump.
# ═══════════════════════════════════════════════════════════════════════════
class Instrument(ABC):
    @abstractmethod
    def revalue(self, market: Market) -> float:
        """Price this instrument under the given market."""
        ...

@dataclass(frozen=True)
class Bond(Instrument):
    times: tuple[float, ...]
    cashflows: tuple[float, ...]
    spread_bp: float

    def revalue(self, market: Market) -> float:
        """Σ CFₜ · market.df(t, self.spread_bp + market.spread_bump_bp)."""
        dfs = np.array([market.df(t, self.spread_bp + market.spread_bump_bp) for t in self.times])
        return np.sum(self.cashflows * dfs)


# ═══════════════════════════════════════════════════════════════════════════
# TASK 4 — Portfolio (composition) and the VaR engine
# ═══════════════════════════════════════════════════════════════════════════
class Portfolio:
    def __init__(self, instruments: list[Instrument]):
        self._instruments = list(instruments)

    def value(self, market: Market) -> float:
        """Sum of revalue() over all instruments."""
        return np.sum([i.revalue(market) for i in self._instruments])


class HistoricalVaREngine:
    def __init__(self, portfolio: Portfolio, scenarios: list[Scenario]):
        self.portfolio = portfolio
        self.scenarios = scenarios

    def pnl_vector(self, base: Market) -> list[float]:
        """For each scenario: portfolio.value(scenario.apply(base)) − value(base)."""
        v0 = self.portfolio.value(base)
        return [self.portfolio.value(s.apply(base)) - v0 for s in self.scenarios]

    def var(self, base: Market, conf: float = 0.995) -> float:
        """Sort the P&L vector; return −sorted[ floor((1−conf)·N) ]."""
        pnl_sorted = sorted(self.pnl_vector(base))
        n = len(pnl_sorted)
        idx = int((1 - conf) * n)
        return -pnl_sorted[idx]
        


# ═══════════════════════════════════════════════════════════════════════════
# GRADING
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from dataclasses import FrozenInstanceError

    N = 250
    scenarios = [Scenario(8*np.sin(i*0.3) + ((i % 11)-5),
                          5*np.cos(i*0.2) + ((i % 7)-3)) for i in range(N)]
    bond = Bond((1, 2, 3, 4, 5), (4, 4, 4, 4, 104), 150.0)
    engine = HistoricalVaREngine(Portfolio([bond]), scenarios)
    base = Market()

    # --- numerical (must match DRILL 8) ---
    val = engine.portfolio.value(base)
    assert abs(val - 92.0706468) < 1e-5, f"value off: {val}"
    pnls = engine.pnl_vector(base)
    assert abs(min(pnls) - (-0.8011423)) < 1e-5 and abs(max(pnls) - 0.8504003) < 1e-5
    assert abs(engine.var(base, 0.995) - 0.7362917) < 1e-5
    assert abs(engine.var(base, 0.99)  - 0.7032813) < 1e-5

    # --- design / best-practice ---
    assert issubclass(Bond, Instrument)
    try: Instrument(); raise AssertionError("ABC should not be instantiable")
    except TypeError: pass
    try: base.rate_bump_bp = 5; raise AssertionError("Market should be frozen")
    except FrozenInstanceError: pass
    assert base.rate_bump_bp == 0.0 and scenarios[0].apply(base).rate_bump_bp != 0.0  # apply is pure

    print(f"portfolio value:  {val:.4f}")
    print(f"P&L min / max:    {min(pnls):.4f} / {max(pnls):.4f}")
    print(f"99.5% VaR:        {engine.var(base, 0.995):.4f}")
    print(f"99.0% VaR:        {engine.var(base, 0.99):.4f}")
    print("design asserts:   ABC ✓  frozen ✓  pure apply ✓")
    print("\n✓ All checks passed.")
