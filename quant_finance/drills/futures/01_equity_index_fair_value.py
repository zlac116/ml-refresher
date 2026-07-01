"""
FUT 1 — Equity Index Futures: Fair Value + Cash-and-Carry Arb
=============================================================

OBJECTIVE
    Equity index futures (e.g. ES, NQ) — compute the no-arbitrage fair price
    given spot, risk-free rate and dividend yield, then size a cash-and-carry
    arbitrage when the listed contract trades rich.

ESTIMATED TIME
    15 min

TOPICS
    Fair forward (no-arb):  F = S * exp((r - q) * T)
    Cost-of-carry intuition: borrow cash, buy spot, short the future
    Per-share arb = F_market - F_fair (when positive: sell future, buy stock)

REAL-WORLD NOTE
    CME E-mini S&P 500 (ES): multiplier = $50/index point.
    Micro E-mini (MES):      multiplier = $5/index point.
    Daily mark-to-market means real arb requires interest financing on
    the variation margin — ignored in this exercise.

REFERENCE
    Hull, ch. 5 (futures pricing); CME ES contract specs.

EXPECTED OUTPUT  (S=4500 idx, r=5%, q=2%, T=0.25y, F_mkt=4540)
    fair future:         4533.876880
    basis (F_mkt-Fair):  6.123120
    arb per index point: 6.123120
    arb total ($):       30615.60   (100 contracts, $50 multiplier)

GRADING
    All asserts must pass.
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def fair_forward(S: float, r: float, q: float, T: float) -> float:
    """No-arb forward: S * exp((r - q) * T).  r and q are continuous-compounded."""
    return S*np.exp((r - q)*T)


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def cash_and_carry_pnl(F_market: float, S: float, r: float, q: float, T: float,
                       n_contracts: int, multiplier: float) -> float:
    """Expected risk-free profit of selling F_market and buying spot:
        per_share = F_market - fair_forward(S, r, q, T)
        total     = per_share * n_contracts * multiplier
    Returns the total $ profit.  Can be negative (future is cheap).
    """
    # TODO: implement
    per_share = F_market - fair_forward(S, r, q, T)
    return per_share * n_contracts * multiplier


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S, r, q, T = 4500.0, 0.05, 0.02, 0.25
    F_fair = fair_forward(S, r, q, T)
    assert abs(F_fair - 4533.876880) < 1e-4

    # No-dividend special case: F = S * e^(rT)
    F_no_div = fair_forward(S, r, q=0.0, T=T)
    assert abs(F_no_div - S * np.exp(r * T)) < 1e-12

    # Arb
    arb = cash_and_carry_pnl(F_market=4540.0, S=S, r=r, q=q, T=T,
                             n_contracts=100, multiplier=50.0)
    assert abs(arb - 30615.60) < 1e-1, arb

    # Reverse: market is CHEAP → arb is negative (we'd buy the future, short stock)
    arb_cheap = cash_and_carry_pnl(F_market=4520.0, S=S, r=r, q=q, T=T,
                                   n_contracts=100, multiplier=50.0)
    assert arb_cheap < 0

    print(f"fair future:         {F_fair:.6f}")
    print(f"basis (F_mkt-Fair):  {4540.0 - F_fair:.6f}")
    print(f"arb per index point: {4540.0 - F_fair:.6f}")
    print(f"arb total ($):       {arb:.2f}   (100 contracts, $50 multiplier)")
    print("\n✓ All checks passed.")
