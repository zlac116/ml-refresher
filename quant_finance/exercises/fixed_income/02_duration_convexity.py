"""
FI 2 — Duration and Convexity
=============================

OBJECTIVE
    For a 5-year, 4% semi-annual-coupon bond at 5% YTM:
      1. Price it.
      2. Compute Macaulay and Modified durations.
      3. Compute Convexity (annualised).
      4. Use duration + convexity to PREDICT the price change for a
         +50bp parallel shift; compare to the actual repricing.

ESTIMATED TIME
    20 min

TOPICS
    Bond pricing: PV of coupon strip + redemption
    Macaulay D = sum_i (t_i * PV_i) / Price
    Modified D = Macaulay / (1 + y/freq)
    Convexity = (sum_i PV_i * t_i * (t_i + 1/freq)) / (Price * (1 + y/freq)^2 * freq^2)
                — annualised form used in dP/P = -D_mod * dy + 0.5 * Conv * dy^2

REFERENCE
    Hull, ch. 4; Fabozzi "Bond Markets" ch. 4.

EXPECTED OUTPUT
    price             = 95.623968
    Macaulay D        = 4.569508
    Modified D        = 4.458056
    Convexity (ann)   = 23.194410
    predicted dP +50bp = -2.103761
    actual    dP +50bp = -2.104025

GRADING
    All asserts must pass.
"""
import numpy as np


def _cashflows(face: float, coupon: float, ytm: float, T: float, freq: int):
    """Return (times in years, cashflows, per-period yield)."""
    n = int(T * freq)
    times = np.arange(1, n + 1) / freq
    cf = np.full(n, face * coupon / freq)
    cf[-1] += face
    return times, cf, ytm / freq


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def bond_price(face: float, coupon: float, ytm: float, T: float, freq: int) -> float:
    """Price a fixed-coupon bond at YTM (annualised, compounded `freq` times)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def macaulay_duration(face: float, coupon: float, ytm: float, T: float, freq: int) -> float:
    """Macaulay duration in years."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def modified_duration(face: float, coupon: float, ytm: float, T: float, freq: int) -> float:
    """Modified duration = Macaulay / (1 + y/freq)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def convexity(face: float, coupon: float, ytm: float, T: float, freq: int) -> float:
    """Annualised convexity (the one that goes with dy in years).

    Conv = sum_i PV_i * k_i * (k_i + 1) / (P * (1 + y/freq)^2 * freq^2)
    where k_i = i (the period index, 1..n).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def predict_dprice(price: float, mod_dur: float, convex: float, dy: float) -> float:
    """Predicted dP = -mod_dur * dy * P + 0.5 * convex * dy^2 * P."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    face, c, y, T, freq = 100.0, 0.04, 0.05, 5.0, 2

    p   = bond_price       (face, c, y, T, freq)
    mac = macaulay_duration(face, c, y, T, freq)
    mod = modified_duration(face, c, y, T, freq)
    cvx = convexity        (face, c, y, T, freq)

    assert abs(p   - 95.623968) < 1e-5
    assert abs(mac -  4.569508) < 1e-5
    assert abs(mod -  4.458056) < 1e-5
    assert abs(cvx - 23.194410) < 1e-4

    dy = 0.005
    dp_pred = predict_dprice(p, mod, cvx, dy)
    dp_act  = bond_price(face, c, y + dy, T, freq) - p
    assert abs(dp_pred - -2.103761) < 1e-4
    assert abs(dp_act  - -2.104025) < 1e-4
    # Duration+convexity should be accurate to ~bp
    assert abs(dp_act - dp_pred) < 5e-3

    print(f"price             = {p:.6f}")
    print(f"Macaulay D        = {mac:.6f}")
    print(f"Modified D        = {mod:.6f}")
    print(f"Convexity (ann)   = {cvx:.6f}")
    print(f"predicted dP +50bp = {dp_pred:.6f}")
    print(f"actual    dP +50bp = {dp_act:.6f}")
    print("\n✓ All checks passed.")
