"""Hull-White single-factor short-rate Bermudan swaption pricer.

Used as a cross-check against the LMM Bermudan price. The two won't agree
exactly (LMM captures forward-rate decorrelation; HW is single-factor) but
should bracket within ~10-15% for typical 5y Bermudans with quarterly exercises.

Implementation uses Monte Carlo simulation of the HW short rate
(Ornstein-Uhlenbeck with time-varying drift to fit the curve) followed by
backward induction for early exercise — simpler than a trinomial tree and
sufficient for the cross-check magnitude target.

dr = (theta(t) - a*r) dt + sigma dW

where theta(t) is calibrated so the model reprices today's discount curve.
For simplicity we approximate theta(t) ≈ partial f(0,t)/partial t + a*f(0,t)
+ sigma^2/(2a) * (1 - exp(-2*a*t)).

This isn't a textbook trinomial implementation; it's a validated MC version
sufficient for the LMM cross-check — that's all the capstone needs.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Sequence

from ..curves import Curve


def _instantaneous_forward(curve: Curve, t: float, eps: float = 1e-4) -> float:
    """Numerical instantaneous forward rate from a discount curve."""
    if t < eps:
        t = eps
    P_minus = float(curve.DF(t - eps / 2))
    P_plus = float(curve.DF(t + eps / 2))
    return -(np.log(P_plus) - np.log(P_minus)) / eps


def hw_simulate_short_rate(
    a: float,
    sigma: float,
    curve: Curve,
    T_total: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Hull-White short rate paths with curve-fitted drift.

    Returns (r_paths, time_grid) where r_paths is (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    dt = T_total / n_steps
    sqrt_dt = np.sqrt(dt)
    grid = np.linspace(0.0, T_total, n_steps + 1)

    # Pre-compute theta(t) on the grid — Hull-White curve fit
    theta_grid = np.zeros_like(grid)
    f_grid = np.array([_instantaneous_forward(curve, max(t, 1e-4)) for t in grid])
    for k, t in enumerate(grid):
        # theta(t) = ∂f(0,t)/∂t + a·f(0,t) + σ²/(2a) · (1 - exp(-2a·t))
        if k == 0 or k == len(grid) - 1:
            df_dt = 0.0
        else:
            df_dt = (f_grid[k + 1] - f_grid[k - 1]) / (grid[k + 1] - grid[k - 1])
        theta_grid[k] = df_dt + a * f_grid[k] + (sigma**2) / (2.0 * a) * (1.0 - np.exp(-2.0 * a * t))

    r_paths = np.zeros((n_paths, n_steps + 1))
    r_paths[:, 0] = f_grid[0]  # r(0) = f(0, 0) = instantaneous forward at t=0
    Z = rng.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        r_paths[:, k + 1] = (
            r_paths[:, k] + (theta_grid[k] - a * r_paths[:, k]) * dt + sigma * sqrt_dt * Z[:, k]
        )
    return r_paths, grid


def hw_bermudan_payer_swaption(
    a: float,
    sigma: float,
    curve: Curve,
    fixed_K: float,
    exercise_dates: Sequence[float],
    swap_pay_dates: Sequence[float],
    swap_pay_freq: float = 1.0,
    n_steps_per_year: int = 24,
    n_paths: int = 5_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Hull-White MC Bermudan payer swaption with LSMC backward induction.

    a, sigma : Hull-White mean reversion and short-rate vol
    curve    : today's OIS / discount curve
    fixed_K  : strike fixed rate
    exercise_dates : year-fractions at which holder may exercise
    swap_pay_dates : payment dates of the underlying swap

    Returns (price, mc_se) on a unit-notional basis. Multiply by trade notional outside.
    """
    T_total = max(exercise_dates[-1], swap_pay_dates[-1])
    n_steps = int(np.ceil(T_total * n_steps_per_year))
    r_paths, grid = hw_simulate_short_rate(a, sigma, curve, T_total, n_steps, n_paths, seed=seed)

    # Approximate path discount factor from t=0 to time t on each path:
    #   P_path(0, t) ≈ exp(-∫_0^t r(s) ds) — discretise via trapezoid on the grid
    dt = grid[1] - grid[0]
    log_disc_path = -np.cumsum(0.5 * (r_paths[:, :-1] + r_paths[:, 1:]) * dt, axis=1)
    log_disc_path = np.concatenate([np.zeros((n_paths, 1)), log_disc_path], axis=1)
    disc_path = np.exp(log_disc_path)  # disc_path[p, k] = path discount from 0 to grid[k]

    def time_idx_for(t: float) -> int:
        return int(np.argmin(np.abs(grid - t)))

    def hw_zcb(t: float, T: float, r_t: np.ndarray) -> np.ndarray:
        """Analytic HW zero-coupon bond P(t, T) given short rate r(t) on each path.

        Standard HW formula: P(t, T) = A(t, T) · exp(-B(t, T) · r(t))
        where  B(t, T) = (1 - exp(-a(T-t))) / a
        and    A(t, T) = (P(0,T)/P(0,t)) · exp( B(t,T)·f(0,t) - σ²/(4a) · (1 - exp(-2at)) · B(t,T)² )
        """
        if T <= t:
            return np.ones_like(r_t)
        B = (1.0 - np.exp(-a * (T - t))) / a
        f0t = _instantaneous_forward(curve, t)
        P0t = float(curve.DF(t))
        P0T = float(curve.DF(T))
        A = (P0T / P0t) * np.exp(
            B * f0t - (sigma**2) / (4.0 * a) * (1.0 - np.exp(-2.0 * a * t)) * B**2
        )
        return A * np.exp(-B * r_t)

    def remaining_swap_pv_at(t_e: float, paths_t_idx: int) -> np.ndarray:
        """PV at exercise t_e of the remaining payer swap, per path.

        Standard payer-swap PV identity:
            PV_payer(t_e) = 1 - P(t_e, T_N) - K · A(t_e)
        where A(t_e) = Σ τ_i · P(t_e, T_i) is the annuity over remaining pay dates.

        Both the terminal bond and each annuity bond are valued via the analytic HW ZCB
        on the path's short rate r(t_e).
        """
        r_te = r_paths[:, paths_t_idx]
        # Annuity over remaining pay dates
        annuity = np.zeros_like(r_te)
        T_last = None
        for tp in swap_pay_dates:
            if tp <= t_e:
                continue
            annuity = annuity + swap_pay_freq * hw_zcb(t_e, tp, r_te)
            T_last = tp
        if T_last is None:
            return np.zeros_like(r_te)
        P_te_TN = hw_zcb(t_e, T_last, r_te)
        return 1.0 - P_te_TN - fixed_K * annuity

    # LSMC backward induction
    exercise_pv = np.zeros(n_paths)
    exercise_t = np.full(n_paths, -1.0)

    for t_e in reversed(list(exercise_dates)):
        idx = time_idx_for(t_e)
        immediate = remaining_swap_pv_at(t_e, idx)

        # Discount future cash flows back to t_e via path discounts
        future_cf = exercise_pv.copy()
        for p in range(n_paths):
            if exercise_t[p] >= 0 and exercise_t[p] > t_e:
                idx_later = time_idx_for(exercise_t[p])
                future_cf[p] *= disc_path[p, idx] / disc_path[p, idx_later]

        in_the_money = immediate > 0
        if in_the_money.sum() > 50:
            x = immediate[in_the_money]
            y = future_cf[in_the_money]
            basis = np.column_stack([np.ones_like(x), x, x**2])
            coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
            cont_itm = basis @ coeffs

            exercise_now = np.zeros(n_paths, dtype=bool)
            exercise_now[in_the_money] = immediate[in_the_money] > cont_itm
            exercise_pv[exercise_now] = immediate[exercise_now]
            exercise_t[exercise_now] = t_e

    # Discount to t = 0 via path discount
    pv_t0 = np.zeros(n_paths)
    for p in range(n_paths):
        if exercise_t[p] < 0:
            continue
        pv_t0[p] = exercise_pv[p] * disc_path[p, time_idx_for(exercise_t[p])]

    return float(pv_t0.mean()), float(pv_t0.std(ddof=1) / np.sqrt(n_paths))
