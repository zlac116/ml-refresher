"""
VOL 3 — Vol Surface Bilinear Interpolation
==========================================

OBJECTIVE
    Given an implied vol surface on a grid of (T, K) points, evaluate the
    implied vol at off-grid points using scipy's RegularGridInterpolator
    (bilinear). The Surface itself is a simple smile + sqrt-T term structure.

ESTIMATED TIME
    15 min

TOPICS
    scipy.interpolate.RegularGridInterpolator (canonical 2-D regular-grid interp)
    Bilinear interpolation behavior at grid corners vs interior
    Why log-strike / log-moneyness is often preferred (not done here for brevity)

REAL-WORLD NOTE
    Production surfaces are stored on (T, K-or-log-K) grids with bilinear
    or cubic interpolation in K, often linear-in-variance in T.
    SVI/SSVI is the canonical arbitrage-free parameterisation for equities.

REFERENCE
    scipy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
    Gatheral, "The Volatility Surface".

EXPECTED OUTPUT
    vol[1, 2] (0.5y ATM):     0.185858
    vol[2, 3] (1y, K=1.1):    0.181000
    interp (T=0.75, K=1.05):  0.183429
    interp (T=0.30, K=0.95):  0.189672
    interp (T=1.50, K=1.15):  0.178358

GRADING
    All asserts must pass.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Grid (deterministic surface so the user sees exact numbers)
tenors_grid  = np.array([0.25, 0.5, 1.0, 2.0])
strikes_grid = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
_vol = np.zeros((len(tenors_grid), len(strikes_grid)))
for _i, _t in enumerate(tenors_grid):
    for _j, _k in enumerate(strikes_grid):
        _vol[_i, _j] = 0.20 + 0.10 * (_k - 1) ** 2 - 0.02 * np.sqrt(_t)
vol_surface = _vol  # exposed name for the exercise


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def build_interp(tenors: np.ndarray, strikes: np.ndarray, vols: np.ndarray):
    """Return a RegularGridInterpolator with method='linear' configured for
    inputs in axis order (T, K). vols.shape must be (len(tenors), len(strikes)).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def interpolate_vol(interp, T: float, K: float) -> float:
    """Evaluate `interp` at one (T, K) point. Return a scalar float."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def interpolate_vol_batch(interp, points: np.ndarray) -> np.ndarray:
    """Evaluate `interp` at many (T, K) points. `points` shape (N, 2)."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    interp = build_interp(tenors_grid, strikes_grid, vol_surface)

    # Exact grid points
    v_atm = interpolate_vol(interp, 0.5, 1.0)
    assert abs(v_atm - 0.185858) < 1e-5

    v_1_1_1 = interpolate_vol(interp, 1.0, 1.1)
    assert abs(v_1_1_1 - 0.181000) < 1e-5

    # Off-grid
    v_off = interpolate_vol(interp, 0.75, 1.05)
    assert abs(v_off - 0.183429) < 1e-5

    # Batch
    pts = np.array([[0.30, 0.95], [1.50, 1.15]])
    vs = interpolate_vol_batch(interp, pts)
    assert vs.shape == (2,)
    assert abs(vs[0] - 0.189672) < 1e-5
    assert abs(vs[1] - 0.178358) < 1e-5

    print(f"vol[1, 2] (0.5y ATM):     {v_atm:.6f}")
    print(f"vol[2, 3] (1y, K=1.1):    {v_1_1_1:.6f}")
    print(f"interp (T=0.75, K=1.05):  {v_off:.6f}")
    print(f"interp (T=0.30, K=0.95):  {vs[0]:.6f}")
    print(f"interp (T=1.50, K=1.15):  {vs[1]:.6f}")
    print("\n✓ All checks passed.")
