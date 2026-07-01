"""
RISK 3 — Bucketed Sensitivities: DV01, Vega Buckets, Cross-Gamma
=================================================================

OBJECTIVE
    Production rates/derivatives risk dashboards show:
      1. DV01 per tenor bucket  → multiply by per-bucket parallel/curve shift.
      2. Vega per tenor bucket  → multiply by per-bucket vol shift.
      3. Cross-gamma matrix     → 0.5 * x' G x for joint underlier shocks.

ESTIMATED TIME
    20 min

TOPICS
    DV01 (dollar value of 1 basis point) per tenor bucket
    Parallel shift PnL = -sum(DV01_b * shift_bp_b)   (long bond loses if rates rise)
    Vega-bucket PnL    =  sum(vega_b * vol_shift_b)
    Cross-gamma PnL    =  0.5 * x' G x   (quadratic form; cross-terms matter)

REAL-WORLD NOTE
    FRTB-SA uses bucketed sensitivities to compute capital. Tenor buckets
    in production: {1d, 1w, 1m, 3m, 6m, 1y, 2y, 3y, 5y, 10y, 15y, 20y, 30y}.
    Here we use a coarse 5-bucket set for clarity.

REFERENCE
    BIS d352 (FRTB SA); Hull ch. 9 (rates); Bouchaud-Potters ch. 6.

EXPECTED OUTPUT
    bucket DV01 (per bp):  [10.00, 10.00, -33.75, 170.00, -315.00]
    net DV01 per bp:       -158.75
    parallel +25bp PnL:    3968.75
    steepener PnL:         5850.00
    vega bucket PnL:       1590.00
    cross-gamma PnL:       1.42

GRADING
    All asserts must pass.
"""
import numpy as np


# ── Bucket setup ───────────────────────────────────────────────────────────
TENORS = [1, 2, 5, 10, 30]                          # years
POSITIONS_NOTIONAL = np.array([100_000, 50_000, -75_000, 200_000, -150_000])
DV01_PER_1K = np.array([0.10, 0.20, 0.45, 0.85, 2.10])  # $ per bp per $1k


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def bucket_dv01(positions_notional: np.ndarray, dv01_per_1k: np.ndarray) -> np.ndarray:
    """positions_notional / 1000 * dv01_per_1k. Returns one DV01 per bucket."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def parallel_pnl(bucket_dv01_arr: np.ndarray, shift_bp: float) -> float:
    """PnL = -sum(DV01) * shift_bp. Rates UP → long bond loses → negative PnL."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def curve_pnl(bucket_dv01_arr: np.ndarray, shifts_bp: np.ndarray) -> float:
    """PnL = -sum(DV01_b * shift_b). For an arbitrary tenor-by-tenor shift."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def vega_pnl(vega_buckets: np.ndarray, vol_shifts_points: np.ndarray) -> float:
    """Vega contribution = sum(vega_b * vol_shift_b)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def cross_gamma_pnl(gamma_matrix: np.ndarray, shocks: np.ndarray) -> float:
    """0.5 * x' G x. Captures cross-terms a per-asset gamma vector misses."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dv01 = bucket_dv01(POSITIONS_NOTIONAL, DV01_PER_1K)
    expected = np.array([10.0, 10.0, -33.75, 170.0, -315.0])
    assert np.allclose(dv01, expected)
    assert abs(dv01.sum() - -158.75) < 1e-6

    pp = parallel_pnl(dv01, shift_bp=25)
    assert abs(pp - 3968.75) < 1e-6

    steep_shifts = np.array([10, 10, 0, 20, 30])
    sp = curve_pnl(dv01, steep_shifts)
    assert abs(sp - 5850.0) < 1e-3

    vega_b   = np.array([1000, 1500, 800, -500])
    vol_shft = np.array([1.0, 0.5, -0.2, 0.0])
    vp = vega_pnl(vega_b, vol_shft)
    assert abs(vp - 1590.0) < 1e-6

    gamma = np.array([
        [10.0,  2.0, -1.0],
        [ 2.0,  8.0,  0.5],
        [-1.0,  0.5, 12.0],
    ])
    shocks = np.array([0.5, -0.3, 0.2])
    gp = cross_gamma_pnl(gamma, shocks)
    assert abs(gp - 1.42) < 1e-6

    print(f"bucket DV01 (per bp):  [{', '.join(f'{x:.2f}' for x in dv01)}]")
    print(f"net DV01 per bp:       {dv01.sum():.2f}")
    print(f"parallel +25bp PnL:    {pp:.2f}")
    print(f"steepener PnL:         {sp:.2f}")
    print(f"vega bucket PnL:       {vp:.2f}")
    print(f"cross-gamma PnL:       {gp:.2f}")
    print("\n✓ All checks passed.")
