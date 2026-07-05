"""
FI 4 — Bond Pricing: Clean, Dirty, Accrued, YTM Back-out
========================================================

OBJECTIVE
    For a 5y semi-annual 4% coupon bond at 5% YTM:
      1. Compute the dirty (full) price.
      2. Compute accrued interest (60 days into a 180-day coupon period).
      3. Compute the clean price = dirty - accrued.
      4. Given the dirty price, back out the YTM via scipy.optimize.brentq.

ESTIMATED TIME
    20 min

TOPICS
    "Dirty" (full, what you pay) vs "clean" (quoted) prices
    Accrued interest: pro-rata of the next coupon
    YTM as IRR of the bond cashflows → solve via brentq

REFERENCE
    Fabozzi, "Bond Markets, Analysis, and Strategies"; Hull, ch. 4.

EXPECTED OUTPUT
    dirty price    = 95.623968
    accrued (60d)  = 0.666667
    clean price    = 94.957301
    YTM back-out   = 0.05000000

GRADING
    All asserts must pass and YTM back-out must reprice exactly.
"""
import numpy as np
from scipy.optimize import brentq


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def dirty_price(face: float, coupon: float, ytm: float, T: float, freq: int) -> float:
    """Present value of all remaining cashflows assuming we're at a coupon date."""
    n = int(T*freq)
    times = np.arange(1, n+1) / freq
    cfs = np.full(n, coupon*face) / freq
    cfs[-1] += face
    dfs = np.array([1 / (1 + ytm/freq)**(t*freq) for t in times])
    return np.sum(cfs * dfs)


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def accrued_interest(face: float, coupon: float, freq: int,
                     days_since_last_coupon: int, days_in_period: int) -> float:
    """Pro-rata of the NEXT coupon for time elapsed since the last.

    accrued = (face * coupon / freq) * (days_elapsed / days_in_period)
    """
    return face * (coupon / freq) * days_since_last_coupon / days_in_period


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def clean_price(dirty: float, accrued: float) -> float:
    """clean = dirty - accrued."""
    # TODO: implement
    return dirty - accrued


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def ytm_from_dirty(dirty: float, face: float, coupon: float,
                   T: float, freq: int) -> float:
    """Solve for ytm such that dirty_price(...) == dirty. Use brentq on
    [0.001, 0.20].
    """
    return brentq(lambda y: dirty_price(face, coupon, y, T, freq) - dirty, 
           a=0.001, b=0.20
        )


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    face, c, y, T, freq = 100.0, 0.04, 0.05, 5.0, 2

    d = dirty_price(face, c, y, T, freq)
    assert abs(d - 95.623968) < 1e-5

    a = accrued_interest(face, c, freq, days_since_last_coupon=60, days_in_period=180)
    assert abs(a - 0.666667) < 1e-5

    cln = clean_price(d, a)
    assert abs(cln - 94.957301) < 1e-5
    assert cln < d  # clean is always <= dirty

    y_back = ytm_from_dirty(d, face, c, T, freq)
    assert abs(y_back - 0.05) < 1e-8

    print(f"dirty price    = {d:.6f}")
    print(f"accrued (60d)  = {a:.6f}")
    print(f"clean price    = {cln:.6f}")
    print(f"YTM back-out   = {y_back:.8f}")
    print("\n✓ All checks passed.")
