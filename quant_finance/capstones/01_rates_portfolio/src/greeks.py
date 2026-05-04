"""Finite-difference Greeks engine for the rates portfolio.

Computes:
- KRD per bucket (1y, 2y, 5y, 10y, 30y) via tent-function curve bumps
- Vega per swaption (expiry × tail) cell
- Theta via day-roll

The engine is generic on the pricer signature: pass a function `price(disc_curve, ...)`
and the bumper handles the curve perturbation + central-difference computation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .curves import KRD_BUCKETS_Y, Curve


def krd_bucket(
    price_fn: Callable[[Curve], float],
    disc: Curve,
    bucket_idx: int,
    delta_bp: float = 1.0,
) -> float:
    """KRD for one bucket via central-difference tent-bump.

    price_fn   : function disc -> PV (in $)
    disc       : base discount curve
    bucket_idx : index into KRD_BUCKETS_Y
    delta_bp   : bucket bump magnitude in bps (default 1bp)

    Returns ΔPV per 1bp move on this bucket (positive = PV rises when bucket rate falls).
    """
    bumped_up = disc.bump_bucket(bucket_idx, +delta_bp)
    bumped_dn = disc.bump_bucket(bucket_idx, -delta_bp)
    pv_up = price_fn(bumped_up)
    pv_dn = price_fn(bumped_dn)
    return (pv_dn - pv_up) / 2.0  # central-diff per bp


def krds_all_buckets(
    price_fn: Callable[[Curve], float], disc: Curve, delta_bp: float = 1.0
) -> np.ndarray:
    """Compute KRDs for all standard buckets."""
    return np.array(
        [krd_bucket(price_fn, disc, i, delta_bp) for i in range(len(KRD_BUCKETS_Y))]
    )


def vega_swaption_cell(
    price_fn: Callable[[float], float], sigma: float, vol_bump: float = 0.01
) -> float:
    """Vega for a single swaption cell via central-diff bump on σ.

    price_fn : function sigma -> PV
    sigma    : base vol
    vol_bump : bump magnitude (default 1 vol point = 0.01)

    Returns ΔPV per 1 vol-point shift (positive = long-vega).
    """
    return (price_fn(sigma + vol_bump) - price_fn(sigma - vol_bump)) / 2.0


def theta_day_roll(
    price_today_fn: Callable[[], float],
    price_tomorrow_fn: Callable[[], float],
    days: float = 1.0,
) -> float:
    """Theta via day-roll: PV(t + Δt) − PV(t).

    Both functions evaluate PV under today's market data; the difference is the
    time-decay component (Δt = `days` calendar days).
    """
    return price_tomorrow_fn() - price_today_fn()
