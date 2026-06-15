"""
FI 4 — Duration Hedging: Two-Bond DV01 Match + Key Rate Durations
==================================================================

OBJECTIVE
    Two canonical desk problems on top of the duration/convexity machinery:

      1. TWO-BOND HEDGE — long $10M of a 5y 6% bond. Hedge by going short a
         50/50 (DV01-weighted) mix of 2y and 10y bonds. Solve for the short
         notionals that zero the portfolio DV01.

      2. KEY RATE DURATIONS — bump the zero curve at 1y / 2y / 5y / 10y
         pillars by 1 bp each, reprice the 10y bond, attribute the price
         change to each pillar.

ESTIMATED TIME
    25 min

TOPICS
    DV01 (dollar value of 1 bp) per unit of face
    DV01-matched hedge: choose hedge notionals so total DV01 = 0
    KRD: shift one tenor pillar at a time → bond's sensitivity at that pillar
    Sum of KRDs ≈ DV01 of bond (parallel shift = sum of all pillar shifts)

REFERENCE
    Hull, ch. 4; Tuckman & Serrat "Fixed Income Securities" ch. 4-6.

EXPECTED OUTPUT
    P_2  = 100.000000, DV01_2 (per $100) = 0.018585
    P_5  = 100.000000, DV01_5 (per $100) = 0.042651
    P_10 = 100.000000, DV01_10 (per $100) = 0.074387

    DV01 target ($/bp)   = 4265.10
    Short 2y notional    = $11,474,276.32
    Short 10y notional   = $2,866,818.10
    Total hedge DV01     = 4265.10   (matches target)
    Net DV01 after hedge = 0.00

    KRD @  1y = 0.000646
    KRD @  2y = 0.002793
    KRD @  5y = 0.009956
    KRD @ 10y = 0.066477
    Sum KRDs  = 0.079873
"""
import numpy as np


# Helpers (reuse the canonical bond functions from FI 3)
def _bond_pv(face, coupon, ytm, T, freq=2):
    n = int(T * freq); times = np.arange(1, n+1)/freq
    cf = np.full(n, face*coupon/freq); cf[-1] += face
    return np.sum(cf / (1 + ytm/freq)**(freq*times))


def _macaulay(face, coupon, ytm, T, freq=2):
    n = int(T * freq); times = np.arange(1, n+1)/freq
    cf = np.full(n, face*coupon/freq); cf[-1] += face
    pv = cf / (1 + ytm/freq)**(freq*times)
    return np.sum(times*pv) / np.sum(pv)


def _modified(face, coupon, ytm, T, freq=2):
    return _macaulay(face, coupon, ytm, T, freq) / (1 + ytm/freq)


def _price_off_curve(face, coupon, T, freq, tenors, zeros):
    """Price a bond off a piecewise-linear continuously-compounded zero curve."""
    n = int(T * freq); times = np.arange(1, n+1)/freq
    cf = np.full(n, face*coupon/freq); cf[-1] += face
    z_at_t = np.interp(times, tenors, zeros)
    return np.sum(cf * np.exp(-z_at_t * times))


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def dv01_per_100_face(face: float, coupon: float, ytm: float, T: float, freq: int = 2) -> float:
    """DV01 per $100 face = modified_duration * Price * 0.0001.

    Returns the $ price change per +1 bp yield rise.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def two_bond_hedge_notionals(long_notional: float, dv01_long: float,
                             dv01_short_a: float, dv01_short_b: float,
                             split_a: float = 0.5) -> tuple[float, float]:
    """Hedge a long position with TWO short bonds, allocating `split_a` of the
    total DV01 to bond A and (1 - split_a) to bond B.

    Returns (notional_short_a, notional_short_b) such that:
        N_a/100 * dv01_short_a  +  N_b/100 * dv01_short_b  =  long_notional/100 * dv01_long

    All DV01s here are per $100 face.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def key_rate_durations(face: float, coupon: float, T: float, freq: int,
                       tenors: np.ndarray, zeros: np.ndarray,
                       bump: float = 0.0001) -> np.ndarray:
    """Compute KRD at each pillar in `tenors`. For each pillar i:
       - bump zeros[i] by `bump`, keep others fixed
       - reprice bond
       - KRD_i = P_base - P_bumped   (positive = sensitivity to that pillar)

    Returns array shape = (len(tenors),).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    y = 0.06
    P_2  = _bond_pv(100, 0.06, y, 2)
    P_5  = _bond_pv(100, 0.06, y, 5)
    P_10 = _bond_pv(100, 0.06, y, 10)
    DV01_2  = dv01_per_100_face(100, 0.06, y, 2)
    DV01_5  = dv01_per_100_face(100, 0.06, y, 5)
    DV01_10 = dv01_per_100_face(100, 0.06, y, 10)

    assert abs(DV01_2  - 0.018585) < 1e-4
    assert abs(DV01_5  - 0.042651) < 1e-4
    assert abs(DV01_10 - 0.074387) < 1e-4

    # Two-bond hedge: long $10M of 5y → hedge with 2y + 10y (50/50 by DV01)
    long_notional = 10_000_000.0
    dv01_target   = long_notional / 100 * DV01_5
    assert abs(dv01_target - 4265.1014) < 0.01

    N_2, N_10 = two_bond_hedge_notionals(long_notional, DV01_5, DV01_2, DV01_10,
                                         split_a=0.5)
    assert abs(N_2  - 11_474_276.32) < 1.0
    assert abs(N_10 -  2_866_818.10) < 1.0

    total_hedge_dv01 = N_2/100 * DV01_2 + N_10/100 * DV01_10
    assert abs(total_hedge_dv01 - dv01_target) < 0.01
    net = dv01_target - total_hedge_dv01
    assert abs(net) < 1e-3, f"net DV01 should be ~0, got {net}"

    # KRDs on 10y bond off a piecewise-linear zero curve
    tenors = np.array([1, 2, 5, 10], dtype=float)
    zeros  = np.array([0.040, 0.045, 0.050, 0.055])
    krds = key_rate_durations(100, 0.06, 10, 2, tenors, zeros)
    assert krds.shape == (4,)
    assert abs(krds[0] - 0.000646) < 1e-4
    assert abs(krds[1] - 0.002793) < 1e-4
    assert abs(krds[2] - 0.009956) < 1e-4
    assert abs(krds[3] - 0.066477) < 1e-4
    # Sum of KRDs ≈ DV01 from a parallel shift
    assert abs(krds.sum() - 0.079873) < 1e-4

    print(f"P_2  = {P_2:.6f}, DV01_2 (per $100) = {DV01_2:.6f}")
    print(f"P_5  = {P_5:.6f}, DV01_5 (per $100) = {DV01_5:.6f}")
    print(f"P_10 = {P_10:.6f}, DV01_10 (per $100) = {DV01_10:.6f}")
    print()
    print(f"DV01 target ($/bp)   = {dv01_target:.2f}")
    print(f"Short 2y notional    = ${N_2:,.2f}")
    print(f"Short 10y notional   = ${N_10:,.2f}")
    print(f"Total hedge DV01     = {total_hedge_dv01:.2f}   (matches target)")
    print(f"Net DV01 after hedge = {net:.2f}")
    print()
    for t, k in zip([1, 2, 5, 10], krds):
        print(f"KRD @ {t:>2}y = {k:.6f}")
    print(f"Sum KRDs  = {krds.sum():.6f}")
    print("\n✓ All checks passed.")
