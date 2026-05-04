"""LMM caplet-vol bootstrap.

Port from notebook `05_libor_market_model.ipynb`. Forward-bootstraps per-period
instantaneous LMM vols from a strip of flat cap vols using the cumulative-variance
identity:

    σ_cap(T_n)² · T_n = Σ_{i=1}^{n} σ_i² · (T_i - T_{i-1})

This is essentially root-finding on the cumulative variance — solve for σ_i given
all earlier σ's.
"""

from __future__ import annotations

import numpy as np


def bootstrap_lmm_caplet_vols(
    cap_maturities: np.ndarray, cap_vols_flat: np.ndarray, T0: float = 1.0
) -> np.ndarray:
    """Bootstrap per-period instantaneous LMM vols from a strip of flat cap vols.

    cap_maturities : array of cap maturities (T_1, T_2, ..., T_N) in years
    cap_vols_flat  : flat cap vol for each maturity (array of same length)
    T0             : start of the first accrual period

    Returns array of σ_i values (per-period instantaneous vols).

    Raises ValueError if the input cap-vol surface implies negative variance at any step.
    """
    cap_maturities = np.asarray(cap_maturities, dtype=float)
    cap_vols_flat = np.asarray(cap_vols_flat, dtype=float)
    if len(cap_maturities) != len(cap_vols_flat):
        raise ValueError("cap_maturities and cap_vols_flat must have the same length")

    N_ = len(cap_maturities)
    sigma_inst = np.zeros(N_)
    T_prev = T0
    cum_var = 0.0
    for n in range(N_):
        T_n = cap_maturities[n]
        target_var = cap_vols_flat[n] ** 2 * T_n
        period = T_n - T_prev
        sigma_n_sq = (target_var - cum_var) / period
        if sigma_n_sq < 0:
            raise ValueError(
                f"Bootstrapped variance negative at step {n} (T={T_n}). "
                f"Cap-vol surface is inconsistent with monotone cumulative variance."
            )
        sigma_inst[n] = np.sqrt(sigma_n_sq)
        cum_var += sigma_n_sq * period
        T_prev = T_n
    return sigma_inst


def rebonato_correlation(
    tenor_dates: np.ndarray, beta: float = 0.05, rho_inf: float = 0.5
) -> np.ndarray:
    """Rebonato three-parameter correlation matrix on the forward strip.

    ρ_ij = ρ_∞ + (1 - ρ_∞) · exp(-β · |T_i - T_j|)

    Used as the default correlation feed to the LMM simulator.
    """
    T_i = np.asarray(tenor_dates[:-1], dtype=float)
    T_diff = np.abs(T_i[:, None] - T_i[None, :])
    return rho_inf + (1.0 - rho_inf) * np.exp(-beta * T_diff)
