"""Bachelier (normal) swaption pricer for low-/negative-rate regimes.

Used when the forward is near zero or negative (Black-76 lognormal blows up).
See `bachelier_cheatsheet.md` and the conversion rule σ_n ≈ σ_LN · √(FK).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bachelier_annuity_premium(
    F: float, K: float, T: float, sigma_n: float, annuity: float, payer: bool = True
) -> float:
    """Premium under Bachelier (arithmetic dynamics).

    sigma_n is in the same units as F (e.g. absolute rate units, decimal).
    """
    if T <= 0 or sigma_n <= 0:
        intrinsic = max(F - K, 0.0) if payer else max(K - F, 0.0)
        return annuity * intrinsic
    d = (F - K) / (sigma_n * np.sqrt(T))
    pdf_d = norm.pdf(d)
    if payer:
        return annuity * ((F - K) * norm.cdf(d) + sigma_n * np.sqrt(T) * pdf_d)
    return annuity * ((K - F) * norm.cdf(-d) + sigma_n * np.sqrt(T) * pdf_d)


def lognormal_vol_to_normal(sigma_LN: float, F: float, K: float) -> float:
    """Hagan-Kennedy off-ATM conversion: σ_n ≈ σ_LN · √(FK).

    Crude ATM rule σ_n ≈ σ_LN · F is the F = K limit. Use this off-ATM.
    """
    return sigma_LN * np.sqrt(max(F * K, 1e-12))
