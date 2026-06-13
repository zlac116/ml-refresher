"""
PNL 3 — Day-over-Day Attribution: Market vs New Trade
=====================================================

OBJECTIVE
    A trader holds 100 calls at end of Day 0. On Day 1:
      - spot moves +1
      - 1 day passes
      - the trader SHORTS 50 puts (newly added position)

    Decompose total Day-1 P&L into:
      1. OLD-book market move P&L (reprice existing positions only)
      2. NEW-trade P&L = (close-of-day mark - traded price) * qty

ESTIMATED TIME
    20 min

TOPICS
    Production attribution columns: "market move", "new trades", "expiry",
    "fee/commission". The discipline is that the sum must = realised P&L.
    A "favourable execution" puts the trade on the book ABOVE (sell) /
    BELOW (buy) the prevailing mid; that's the new-trade P&L.

REAL-WORLD NOTE
    Real systems use start-of-day Greeks for explain attribution; new
    trades on Day 1 contribute zero to OLD-book explain — by definition,
    those positions don't exist at SOD.

REFERENCE
    BIS RBC30.42 (FRTB attribution); generic trading-desk P&L books.

EXPECTED OUTPUT  (S0=100, K=100, T0=1, r=5%, sigma=20%)
    C0           = 10.4506
    C1           = 11.0710
    old book pnl = 62.0415
    new fair P   =  5.2128, sold @ 5.2628 (favourable +0.05)
    new trade pnl = 2.5000
    total pnl    = 64.5415

GRADING
    All asserts must pass.
"""
import numpy as np
from scipy.stats import norm


def _bsm_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bsm_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def old_book_market_pnl(qty: int, S0: float, S1: float, K: float, T0: float,
                        r: float, sigma: float, dt: float) -> float:
    """P&L from repricing an EXISTING call position:
        qty * (C(S1, K, T0-dt, r, sigma) - C(S0, K, T0, r, sigma))
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def new_trade_pnl(qty: int, fair_price_eod: float, traded_price: float) -> float:
    """P&L from a NEW trade marked at end of day:
        qty * (fair_price_eod - traded_price)
    Convention: qty < 0 for short. If qty=-50 and you sold ABOVE fair, your
    P&L is positive: (-50) * (fair - traded) = (-50) * (fair - above_fair) > 0.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S0, S1 = 100.0, 101.0
    K, T0, r, sigma = 100.0, 1.0, 0.05, 0.20
    dt = 1 / 252

    C0 = _bsm_call(S0, K, T0,      r, sigma)
    C1 = _bsm_call(S1, K, T0 - dt, r, sigma)
    assert abs(C0 - 10.4506) < 1e-3
    assert abs(C1 - 11.0710) < 1e-3

    old_pnl = old_book_market_pnl(100, S0, S1, K, T0, r, sigma, dt)
    assert abs(old_pnl - 62.0415) < 1e-2

    # New trade: short 50 puts, fair=P_eod, executed 0.05 favourably (sold above fair)
    P_eod   = _bsm_put(S1, K, T0 - dt, r, sigma)
    traded  = P_eod + 0.05
    new_pnl = new_trade_pnl(qty=-50, fair_price_eod=P_eod, traded_price=traded)
    assert abs(new_pnl - 2.5) < 1e-6

    total = old_pnl + new_pnl
    assert abs(total - 64.5415) < 1e-2

    # The decomposition must be additive — verifying by reconstruction
    assert abs(total - (old_pnl + new_pnl)) < 1e-12

    print(f"C0           = {C0:.4f}")
    print(f"C1           = {C1:.4f}")
    print(f"old book pnl = {old_pnl:.4f}")
    print(f"new fair P   =  {P_eod:.4f}, sold @ {traded:.4f} (favourable +0.05)")
    print(f"new trade pnl = {new_pnl:.4f}")
    print(f"total pnl    = {total:.4f}")
    print("\n✓ All checks passed.")
