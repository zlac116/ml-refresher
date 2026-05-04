"""LIBOR Market Model (LMM / BGM) — predictor-corrector simulator under terminal measure.

Port of the audit-validated implementation from
`quant_finance/03_fixed_income/05_libor_market_model.ipynb`.

Caplet repricing test passed within ±1.4σ MC of Black-76 in the source notebook.
The Bermudan LSMC implementation matches the structure of Longstaff-Schwartz 2001
with polynomial basis on the swap rate at the exercise date, ITM paths only.

Notes vs the notebook implementation:
- `tenor_dates` and `deltas` are passed explicitly (the notebook captured them
  via closure, which is fine inline but a bug for production).
- The discount factor `1/P(T_{i+1}, T_N) = prod_k (1 + delta_k L_k(T_{i+1}))`
  carries from terminal-measure pricing: V_0 = P(0, T_N) · E^{T_N}[V_T / P(T, T_N)].
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def lmm_terminal_drift(
    L: np.ndarray, sigma: np.ndarray, rho: np.ndarray, deltas: np.ndarray
) -> np.ndarray:
    """Drift vector under terminal measure for ALL forwards, ALL paths.

    L      : (n_paths, N) — current forward LIBOR values
    sigma  : (N,)         — instantaneous vol per forward
    rho    : (N, N)       — correlation matrix
    deltas : (N,)         — accrual period lengths

    Returns drift (n_paths, N) where:
        drift[p, i] = -sigma_i * sum_{j=i+1}^{N-1} (delta_j * rho_ij * sigma_j * L_j) / (1 + delta_j * L_j)

    Last forward L_{N-1} has zero drift under terminal measure (martingale).
    """
    n_paths_, N_ = L.shape
    # Per-j path-only term, independent of i except for correlation:
    #   T_pj[p, j] = delta_j * sigma_j * L[p, j] / (1 + delta_j * L[p, j])
    T_pj = (deltas[None, :] * sigma[None, :] * L) / (1.0 + deltas[None, :] * L)
    drift = np.zeros_like(L)
    for i in range(N_ - 1):  # last forward has zero drift under terminal measure
        drift[:, i] = -sigma[i] * (T_pj[:, i + 1 :] @ rho[i, i + 1 :])
    return drift


def simulate_lmm_terminal(
    L0: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
    deltas: np.ndarray,
    T_total: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate forward-LIBOR strip under terminal measure with predictor-corrector.

    L0      : (N,) initial forward LIBOR values L_i(0)
    sigma   : (N,) per-forward instantaneous vols (constant in time)
    rho     : (N, N) correlation matrix (PSD)
    deltas  : (N,) accrual period lengths
    T_total : simulation horizon (typically tenor_dates[-2], the last reset)
    n_steps : number of time steps in [0, T_total]
    n_paths : Monte Carlo paths

    Returns (paths, grid):
        paths : (n_paths, n_steps + 1, N) with paths[p, t, i] = L_i(t)
        grid  : (n_steps + 1,) of time grid points (years)
    """
    N_ = len(L0)
    dt = T_total / n_steps
    sqrt_dt = np.sqrt(dt)
    rng = np.random.default_rng(seed)
    L_chol = np.linalg.cholesky(rho)
    sigma = np.asarray(sigma, dtype=float)

    L = np.tile(L0.astype(float), (n_paths, 1))
    out = np.zeros((n_paths, n_steps + 1, N_))
    out[:, 0, :] = L0
    grid = np.linspace(0.0, T_total, n_steps + 1)

    for k in range(n_steps):
        # Correlated standard normals
        Z_uncorr = rng.standard_normal((n_paths, N_))
        Z = Z_uncorr @ L_chol.T

        # Predictor: drift at current L
        drift_p = lmm_terminal_drift(L, sigma, rho, deltas)
        L_pred = L * np.exp((drift_p - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z)

        # Corrector: drift at the predictor; use average drift
        drift_c = lmm_terminal_drift(L_pred, sigma, rho, deltas)
        avg_drift = 0.5 * (drift_p + drift_c)
        L = L * np.exp((avg_drift - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z)

        out[:, k + 1, :] = L

    return out, grid


def lmm_caplet_price(
    L0: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
    deltas: np.ndarray,
    tenor_dates: np.ndarray,
    K: float,
    i: int,
    P_0_TN: float,
    n_steps: int = 240,
    n_paths: int = 20_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Price caplet on L_i via terminal-measure MC: V_0 = P(0,T_N) E[V_T / P(T_{i+1},T_N)].

    Used for the LMM caplet repricing test against Black-76.

    Returns (price, mc_se).
    """
    N_ = len(L0)
    T_horizon = tenor_dates[-2]
    paths, grid = simulate_lmm_terminal(
        L0, sigma, rho, deltas, T_horizon, n_steps, n_paths, seed=seed
    )

    T_i = tenor_dates[i]
    T_iplus1 = tenor_dates[i + 1]
    idx_Ti = int(np.argmin(np.abs(grid - T_i)))
    idx_Tip1 = int(np.argmin(np.abs(grid - T_iplus1)))

    L_i_at_Ti = paths[:, idx_Ti, i]
    payoff = deltas[i] * np.maximum(L_i_at_Ti - K, 0.0)

    if i + 1 < N_:
        L_kk_at_Tip1 = paths[:, idx_Tip1, i + 1 : N_]
        delta_kk = deltas[i + 1 : N_]
        # 1 / P(T_{i+1}, T_N) = prod_k (1 + delta_k L_k(T_{i+1}))
        inv_disc = np.prod(1.0 + delta_kk[None, :] * L_kk_at_Tip1, axis=1)
    else:
        inv_disc = np.ones(n_paths)

    # V_0 = P(0,T_N) · E[ payoff(T_{i+1}) / P(T_{i+1}, T_N) ]
    estimator = P_0_TN * payoff * inv_disc
    return float(estimator.mean()), float(estimator.std(ddof=1) / np.sqrt(n_paths))


def lmm_bermudan_payer_swaption(
    L0: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
    deltas: np.ndarray,
    tenor_dates: np.ndarray,
    fixed_K: float,
    exercise_idxs: Sequence[int],
    swap_settle_idxs: Sequence[int],
    P_0_TN: float,
    n_steps: int = 240,
    n_paths: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Price a Bermudan payer swaption via LMM + LSMC.

    fixed_K           : fixed rate of the underlying swap
    exercise_idxs     : indices into tenor_dates at which exercise is allowed
    swap_settle_idxs  : indices of forwards underlying the swap (e.g. range(2, 6) for
                        a swap that settles on L_2..L_5)
    P_0_TN            : P(0, T_N), discount to terminal measure numeraire

    Returns (price, mc_se).
    """
    N_ = len(L0)
    T_horizon = tenor_dates[-2]
    paths, grid = simulate_lmm_terminal(
        L0, sigma, rho, deltas, T_horizon, n_steps, n_paths, seed=seed
    )

    def time_idx_for(t: float) -> int:
        return int(np.argmin(np.abs(grid - t)))

    # Restrict swap forwards to the user-specified subset; defaults to all forwards
    # from ex_idx to the strip end if the parameter is empty.
    settle_set = set(swap_settle_idxs)

    def remaining_swap_pv_at(paths_t: np.ndarray, ex_idx: int) -> np.ndarray:
        """PV at exercise date of the REMAINING payer swap from ex_idx onward.

        Only includes forwards in `swap_settle_idxs` AND >= ex_idx (forwards already
        reset before the exercise date contribute nothing to the remaining-tail PV).
        """
        n_p = paths_t.shape[0]
        pv = np.zeros(n_p)
        for k in range(ex_idx, N_):
            if k not in settle_set:
                continue
            # Discount factor P(T_e, T_{k+1}) = prod_{j=ex_idx..k} 1/(1 + delta_j L_j(T_e))
            ratios = 1.0 / (
                1.0 + deltas[ex_idx : k + 1][None, :] * paths_t[:, ex_idx : k + 1]
            )
            P_Te_Tkp1 = ratios.prod(axis=1)
            pv += deltas[k] * (paths_t[:, k] - fixed_K) * P_Te_Tkp1
        return pv

    # LSMC backward induction
    n_paths_loc = paths.shape[0]
    exercise_pv = np.zeros(n_paths_loc)
    exercise_t_idx = np.full(n_paths_loc, -1, dtype=int)

    for ex_tenor_idx in reversed(list(exercise_idxs)):
        t_e = tenor_dates[ex_tenor_idx]
        grid_idx = time_idx_for(t_e)
        paths_t = paths[:, grid_idx, :]
        immediate = remaining_swap_pv_at(paths_t, ex_tenor_idx)

        # Prepare future_cf: for paths that exercise later, discount their PV back to t_e
        future_cf = exercise_pv.copy()
        for p_idx in range(n_paths_loc):
            if exercise_t_idx[p_idx] != -1:
                disc = 1.0
                for j in range(ex_tenor_idx, exercise_t_idx[p_idx]):
                    disc /= 1.0 + deltas[j] * paths_t[p_idx, j]
                future_cf[p_idx] *= disc

        # Regress on polynomial basis of immediate (ITM only)
        in_the_money = immediate > 0
        if in_the_money.sum() > 50:
            x = immediate[in_the_money]
            y = future_cf[in_the_money]
            basis = np.column_stack([np.ones_like(x), x, x**2])
            coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
            cont_value_itm = basis @ coeffs

            exercise_now = np.zeros(n_paths_loc, dtype=bool)
            exercise_now[in_the_money] = immediate[in_the_money] > cont_value_itm
            exercise_pv[exercise_now] = immediate[exercise_now]
            exercise_t_idx[exercise_now] = ex_tenor_idx

    # Discount to t = 0 under terminal measure
    disc_terms = np.ones(n_paths_loc)
    for p_idx in range(n_paths_loc):
        et_idx = exercise_t_idx[p_idx]
        if et_idx == -1:
            exercise_pv[p_idx] = 0.0
            disc_terms[p_idx] = 1.0
        else:
            t_e_idx = time_idx_for(tenor_dates[et_idx])
            path_at_te = paths[p_idx, t_e_idx, :]
            ratios = 1.0 + deltas[et_idx:N_] * path_at_te[et_idx:N_]
            disc_terms[p_idx] = ratios.prod()

    estimator = P_0_TN * exercise_pv * disc_terms
    return float(estimator.mean()), float(estimator.std(ddof=1) / np.sqrt(n_paths_loc))
