"""
FI 16 — Relative Value Spreads (G-spread, Z-spread, ASW)
==========================================================

OBJECTIVE
    For a 5y corporate bond at market price 98, compute the three standard
    spreads that desk traders quote on it.

      1. G-spread:  Corp YTM - Treasury par rate (same maturity)
      2. Z-spread:  constant spread to the ZERO CURVE such that PV = market
      3. ASW spread:  approximate spread vs the swap curve at same tenor
      4. Reverse direction: given a quoted Z-spread, imply the bond price

ESTIMATED TIME
    25 min

TOPICS
    YTM (single rate that reprices the bond):
        sum_i cf_i / (1 + y)^t_i  =  Market_price
    Z-spread (parallel shift of the zero curve):
        sum_i cf_i * exp(-(z(t_i) + s_Z) * t_i)  =  Market_price
    G-spread (yield difference):
        s_G  =  Corp_YTM  -  Treasury_YTM (same maturity)
    ASW spread:  for vanilla annual bonds, asw ≈ corp_YTM - swap_rate
                 (rigorous ASW uses the annuity factor; this is the
                 production approximation desk traders use)

REFERENCE
    See 03_fixed_income/cheatsheets/market_quoting.md §10 for full definitions.
    Fabozzi "Bond Markets" ch. 5; Choudhry "Analysing and Interpreting
    the Yield Curve" ch. 8.

EXPECTED OUTPUT  (5y, 6% annual coupon corp, market price 98, on the curve
                  bootstrapped from FI 10/11)
    Corp YTM       = 0.064810
    Treasury 5y    = 0.055000
    G-spread       = 0.009810   (98.10 bp)
    Z-spread       = 0.009279   (92.79 bp)
    ASW spread     = 0.009810   (98.10 bp)  ← same as G-spread when swap≈Treas

    Implied price at quoted z-spread 120 bp: 96.8194
"""
import numpy as np
from scipy.optimize import brentq


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def corporate_ytm(market_price: float, face: float, coupon_rate: float,
                  T: int, freq: int = 1) -> float:
    """Solve for ytm such that PV at ytm = market_price.

    Use brentq on [-0.05, 0.30]. Cashflows: coupon every period, +face at end.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def g_spread(corp_ytm: float, benchmark_treasury_par: float) -> float:
    """G-spread = corp YTM - benchmark Treasury par rate (same maturity)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def z_spread(market_price: float, face: float, coupon_rate: float, T: int,
             zero_rates: np.ndarray, tenors: np.ndarray, freq: int = 1) -> float:
    """Z-spread: constant additive shift to the zero curve such that

        sum_i cf_i * exp(-(z(t_i) + s) * t_i)  =  market_price

    Use brentq on s ∈ [-0.05, 0.10].

    `zero_rates` and `tenors` are the zero curve at integer year pillars.
    Interpolate (or assume tenor[i] = i+1 for simplicity).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def price_at_z_spread(z_spread: float, face: float, coupon_rate: float, T: int,
                      zero_rates: np.ndarray, tenors: np.ndarray,
                      freq: int = 1) -> float:
    """Reverse direction: given a quoted Z-spread, compute the implied price.

        price  =  sum_i cf_i * exp(-(z(t_i) + s) * t_i)
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def asw_spread_approx(corp_ytm: float, swap_par_rate: float) -> float:
    """Production approximation:  ASW ≈ corp YTM - swap par rate (same maturity).

    Rigorous ASW uses the annuity factor; this approximation works well for
    par-ish bonds and is what desk traders quote intraday.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Same curve as FI 10/11
    par_rates = np.array([0.050, 0.052, 0.053, 0.054, 0.055])
    D = np.empty(5); s = 0.0
    for k, c in enumerate(par_rates):
        D[k] = (1 - c * s) / (1 + c); s += D[k]
    tenors = np.array([1, 2, 3, 4, 5], dtype=float)
    zeros  = -np.log(D) / tenors

    # 5y corp bond, 6% annual coupon, price 98
    face, coupon_r, T = 100.0, 0.06, 5
    market_price = 98.0

    # Corp YTM
    ytm = corporate_ytm(market_price, face, coupon_r, T)
    assert abs(ytm - 0.064810) < 1e-5

    # G-spread vs Treasury 5y par
    gs = g_spread(ytm, benchmark_treasury_par=par_rates[-1])
    assert abs(gs - 0.009810) < 1e-5

    # Z-spread
    zs = z_spread(market_price, face, coupon_r, T, zeros, tenors)
    assert abs(zs - 0.009279) < 1e-5
    # G-spread > Z-spread when curve is upward-sloping (typically) — sanity sign
    assert gs > zs

    # Round-trip: Z-spread back to price MUST equal market_price
    P_implied = price_at_z_spread(zs, face, coupon_r, T, zeros, tenors)
    assert abs(P_implied - market_price) < 1e-6

    # Reverse: price at a quoted 120 bp Z-spread
    P_at_120bp = price_at_z_spread(0.012, face, coupon_r, T, zeros, tenors)
    assert abs(P_at_120bp - 96.8194) < 1e-3

    # ASW approximation
    asw = asw_spread_approx(ytm, swap_par_rate=par_rates[-1])
    assert abs(asw - 0.009810) < 1e-5

    print(f"Corp YTM       = {ytm:.6f}")
    print(f"Treasury 5y    = {par_rates[-1]:.6f}")
    print(f"G-spread       = {gs:.6f}   ({gs*1e4:.2f} bp)")
    print(f"Z-spread       = {zs:.6f}   ({zs*1e4:.2f} bp)")
    print(f"ASW spread     = {asw:.6f}   ({asw*1e4:.2f} bp)  ← same as G-spread when swap≈Treas")
    print()
    print(f"Implied price at quoted z-spread 120 bp: {P_at_120bp:.4f}")
    print("\n✓ All checks passed.")
