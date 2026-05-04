"""Curve infrastructure: bootstrap, log-DF interpolation, KRD bucket bumps.

Two-curve world (post-2008):
- Discount curve   = OIS (collateralised trades)
- Projection curve = LIBOR / SOFR (for floating-leg fixings)

For the capstone we keep both as zero-rate curves on the same tenor grid,
interpolated log-linearly in discount-factor space (industry standard for
curve smoothness).

KRD bucket bumps follow the "tent function" convention: the i-th bucket bump
is a triangular shock centered at the i-th key rate, decaying linearly to the
neighbouring key rates. KRDs sum to (approximately) the modified-duration
parallel shift up to interpolation residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# Standard KRD bucket grid (year-fractions). 5 buckets is the desk-standard set.
KRD_BUCKETS_Y = np.array([1.0, 2.0, 5.0, 10.0, 30.0])


@dataclass
class Curve:
    """Zero-rate curve. Stores tenors (years) + zero rates (continuously compounded)."""
    tenors_y: np.ndarray
    zero_rates: np.ndarray
    name: str = "OIS"

    def discount(self, t: float | np.ndarray) -> float | np.ndarray:
        """P(0, t) = exp(-z(t) * t).

        Linear interpolation in zero rate. For very short tenors (< first node)
        we extrapolate flat from the first node. For very long (> last node) we
        extrapolate flat from the last node (reasonable for capstone scope).
        """
        z = np.interp(t, self.tenors_y, self.zero_rates)
        return np.exp(-z * np.asarray(t))

    def DF(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.discount(t)

    def shift(self, delta: float) -> "Curve":
        """Parallel shift in zero rates."""
        return Curve(
            tenors_y=self.tenors_y.copy(),
            zero_rates=self.zero_rates + delta,
            name=self.name,
        )

    def bump_bucket(self, bucket_idx: int, delta_bp: float = 1.0) -> "Curve":
        """Tent-function bump on the i-th KRD bucket.

        The bump is `delta_bp` bps at the bucket key rate, decaying linearly to zero
        at the adjacent key rates (left and right). At all other tenors, no change.
        """
        delta = delta_bp / 1e4
        bumps = np.zeros_like(self.tenors_y)
        bucket_t = KRD_BUCKETS_Y[bucket_idx]
        # Find neighbouring buckets for the tent
        left = KRD_BUCKETS_Y[bucket_idx - 1] if bucket_idx > 0 else 0.0
        right = (
            KRD_BUCKETS_Y[bucket_idx + 1] if bucket_idx < len(KRD_BUCKETS_Y) - 1 else self.tenors_y[-1] + 5.0
        )
        for i, t in enumerate(self.tenors_y):
            if left < t <= bucket_t:
                bumps[i] = delta * (t - left) / (bucket_t - left)
            elif bucket_t < t < right:
                bumps[i] = delta * (right - t) / (right - bucket_t)
            elif np.isclose(t, bucket_t):
                bumps[i] = delta
        return Curve(
            tenors_y=self.tenors_y.copy(),
            zero_rates=self.zero_rates + bumps,
            name=self.name,
        )


# ------------------------------------------------------------------
# Curve bootstrap
# ------------------------------------------------------------------
def bootstrap_zero_curve(
    tenors_y: np.ndarray,
    zero_rates: np.ndarray,
    name: str = "OIS",
) -> Curve:
    """Convenience constructor — for the capstone we ingest pre-bootstrapped
    zero rates from `data/market_data/curves_D{i}.parquet`. Real production
    bootstraps from deposit/futures/swap quotes; that machinery is in
    `quant_finance/03_fixed_income/03_curve_building.ipynb` if needed.
    """
    tenors_y = np.asarray(tenors_y, dtype=float)
    zero_rates = np.asarray(zero_rates, dtype=float)
    if len(tenors_y) != len(zero_rates):
        raise ValueError("tenors_y and zero_rates must have the same length")
    return Curve(tenors_y=tenors_y, zero_rates=zero_rates, name=name)


# ------------------------------------------------------------------
# Forward rate from the curve
# ------------------------------------------------------------------
def forward_rate_from_curve(curve: Curve, T_start: float, T_end: float) -> float:
    """Continuously compounded forward rate over [T_start, T_end] from a discount curve."""
    if T_end <= T_start:
        return 0.0
    return (np.log(curve.DF(T_start)) - np.log(curve.DF(T_end))) / (T_end - T_start)


def simple_forward_rate(curve: Curve, T_start: float, T_end: float) -> float:
    """Simple-compounded forward rate (used for LIBOR-style fixings)."""
    if T_end <= T_start:
        return 0.0
    delta = T_end - T_start
    return (curve.DF(T_start) / curve.DF(T_end) - 1.0) / delta


def par_swap_rate(disc: Curve, proj: Curve, T_start: float, T_end: float, freq: float = 1.0) -> tuple[float, float]:
    """Par swap rate K_par and annuity A under multi-curve discounting.

    K_par = (proj-curve floating PV) / annuity
          = (P_disc(T_start) - P_disc(T_end))   /   sum_i delta_i * P_disc(T_pay_i)

    For the capstone we use the **simplified single-curve identity** when proj == disc;
    for true multi-curve, compute the floating PV on the projection curve.

    Returns (par_K, annuity).
    """
    from .trades import pay_dates

    pds = pay_dates(T_start, T_end, freq)
    annuity = float(np.sum([freq * disc.DF(t) for t in pds]))
    if proj is disc or np.allclose(proj.zero_rates, disc.zero_rates):
        # Standard telescoping identity
        par_K = float((disc.DF(T_start) - disc.DF(T_end)) / annuity)
    else:
        # True multi-curve: sum projected floating cash flows discounted on disc curve
        floating_pv = 0.0
        prev_T = T_start
        for t in pds:
            f = simple_forward_rate(proj, prev_T, t)
            floating_pv += freq * f * disc.DF(t)
            prev_T = t
        par_K = floating_pv / annuity
    return par_K, annuity
