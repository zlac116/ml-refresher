"""SABR fitting harness.

Port from `quant_finance/01_options/02_bs_family_and_asset_classes.ipynb` Part 2.

Recommended workflow per swaption (expiry × tail) cell:
1. Fix β by market convention (β = 0.5 for rates).
2. Solve α from the ATM market vol via the cubic in α.
3. Least-squares fit (ρ, ν) to the off-ATM smile.

This is the "α-from-ATM" trick — without it, joint (α, ρ, ν) calibration is
parameter-degenerate (many (α, β) pairs give the same ATM vol).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def sabr_lognormal_vol(
    F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float
) -> float:
    """Hagan 2002 lognormal implied vol for SABR.

    Splits to ATM-limit branch when |F - K| is small to avoid the 0/0 in z/x(z).
    """
    if F <= 0 or K <= 0:
        raise ValueError("Hagan formula requires F, K > 0.")

    if abs(F - K) < 1e-12:
        # ATM-limit
        FK_beta = F ** (1.0 - beta)
        return (alpha / FK_beta) * (
            1.0
            + (
                (1.0 - beta) ** 2 / 24.0 * alpha**2 / FK_beta**2
                + rho * beta * nu * alpha / (4.0 * FK_beta)
                + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
            )
            * T
        )

    log_FK = np.log(F / K)
    FK_avg_beta = (F * K) ** ((1.0 - beta) / 2.0)
    z = (nu / alpha) * FK_avg_beta * log_FK
    x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))

    term_a = alpha / (
        FK_avg_beta
        * (
            1.0
            + (1.0 - beta) ** 2 / 24.0 * log_FK**2
            + (1.0 - beta) ** 4 / 1920.0 * log_FK**4
        )
    )
    term_b = z / x_z
    term_c = 1.0 + (
        (1.0 - beta) ** 2 / 24.0 * alpha**2 / FK_avg_beta**2
        + rho * beta * nu * alpha / (4.0 * FK_avg_beta)
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    ) * T

    return term_a * term_b * term_c


def sabr_atm_alpha(sigma_atm: float, F: float, T: float, beta: float, rho: float, nu: float) -> float:
    """Solve the ATM SABR cubic for α given ATM vol, β, ρ, ν.

    Setting σ_LN(F, F) = σ_ATM in the ATM-limit Hagan formula and rearranging in α:

        C3 * α³ + C2 * α² + C1 * α + C0 = 0

    where
        C3 = T (1-β)² / (24 F^{2(1-β)})
        C2 = T ρ β ν / (4 F^{1-β})
        C1 = 1 + T (2 - 3ρ²) ν² / 24
        C0 = -σ_ATM · F^{1-β}

    Returns the smallest positive real root.
    """
    Fpow = F ** (1.0 - beta)
    C3 = T * (1.0 - beta) ** 2 / (24.0 * Fpow**2)
    C2 = T * rho * beta * nu / (4.0 * Fpow)
    C1 = 1.0 + T * (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    C0 = -sigma_atm * Fpow

    roots = np.roots([C3, C2, C1, C0])
    real_pos = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
    if not real_pos:
        # Fallback: linear approximation σ_ATM = α / F^{1-β}
        return sigma_atm * Fpow
    return min(real_pos)


def sabr_calibrate_cell(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    F: float,
    T: float,
    beta: float = 0.5,
    rho_init: float = -0.3,
    nu_init: float = 0.4,
) -> dict:
    """Calibrate SABR for one (expiry × tail) cell.

    Strategy: fix β, solve α from ATM via cubic, least-squares fit (ρ, ν) to off-ATM
    market vols. Returns {'alpha', 'beta', 'rho', 'nu', 'residuals'}.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)

    # Identify ATM strike (closest to F)
    atm_idx = int(np.argmin(np.abs(strikes - F)))
    sigma_atm = market_vols[atm_idx]

    def residuals(params):
        rho, nu = params
        rho = np.clip(rho, -0.99, 0.99)
        nu = max(nu, 1e-6)
        try:
            alpha = sabr_atm_alpha(sigma_atm, F, T, beta, rho, nu)
        except Exception:
            return np.ones_like(market_vols) * 1e6
        model_vols = np.array(
            [sabr_lognormal_vol(F, K, T, alpha, beta, rho, nu) for K in strikes]
        )
        return model_vols - market_vols

    result = least_squares(
        residuals,
        x0=[rho_init, nu_init],
        bounds=([-0.99, 1e-6], [0.99, 5.0]),
        method="trf",
    )
    rho_fit, nu_fit = result.x
    alpha_fit = sabr_atm_alpha(sigma_atm, F, T, beta, rho_fit, nu_fit)
    final_resids = residuals([rho_fit, nu_fit])

    return {
        "alpha": float(alpha_fit),
        "beta": float(beta),
        "rho": float(rho_fit),
        "nu": float(nu_fit),
        "residuals": final_resids,
        "sse": float(np.sum(final_resids**2)),
    }
