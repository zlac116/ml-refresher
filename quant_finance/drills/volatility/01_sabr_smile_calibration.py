"""
VOL 1 — SABR Smile Calibration (Hagan 2002)
===========================================

OBJECTIVE
    Given a synthetic market vol smile (5 strikes around ATM), fit the
    SABR model parameters (alpha, rho, nu) holding beta fixed at 0.5
    (canonical for rates / commodity smiles). Verify the fit recovers
    the true parameters to RMSE < 1 bp.

ESTIMATED TIME
    20 min

TOPICS
    Hagan-Kumar-Lesniewski-Woodward (2002) lognormal-SABR approximation
    scipy.optimize.least_squares with bounds
    beta fixed by convention; calibrating only (alpha, rho, nu)

REFERENCE
    Hagan et al., "Managing Smile Risk", Wilmott Magazine, 2002.

EXPECTED OUTPUT  (true alpha=0.02, rho=-0.30, nu=0.45, beta=0.5)
    market vols:  [0.205361 0.158325 0.128089 0.11918  0.124919]
    alpha_fit:    0.020000
    rho_fit:     -0.300000
    nu_fit:       0.450000
    fit rmse:     0.0000 bp

GRADING
    All asserts must pass. Recovered params must equal truth to 1e-4.
"""
import numpy as np
from scipy.optimize import least_squares


def _sabr_black_vol(F: float, K: float, T: float,
                    alpha: float, beta: float, rho: float, nu: float) -> float:
    """Hagan 2002 lognormal SABR implied Black vol (no shift)."""
    if abs(F - K) < 1e-12:
        FK_beta = F ** (1 - beta)
        first = alpha / FK_beta
        bracket = 1 + (
            ((1 - beta) ** 2 / 24) * alpha ** 2 / FK_beta ** 2
            + 0.25 * rho * beta * nu * alpha / FK_beta
            + (2 - 3 * rho ** 2) / 24 * nu ** 2
        ) * T
        return first * bracket
    log_FK = np.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * FK_beta * log_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    first = alpha / (
        FK_beta * (1 + (1 - beta) ** 2 / 24 * log_FK ** 2
                   + (1 - beta) ** 4 / 1920 * log_FK ** 4)
    )
    second = z / x_z
    third = 1 + (
        ((1 - beta) ** 2 / 24) * alpha ** 2 / FK_beta ** 2
        + 0.25 * rho * beta * nu * alpha / FK_beta
        + (2 - 3 * rho ** 2) / 24 * nu ** 2
    ) * T
    return first * second * third


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def sabr_vols(strikes: np.ndarray, F: float, T: float,
              alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
    """Vectorise _sabr_black_vol across a strike array."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def calibrate_sabr(strikes: np.ndarray, market_vols: np.ndarray,
                   F: float, T: float, beta: float = 0.5) -> tuple[float, float, float]:
    """Fit (alpha, rho, nu) via scipy.optimize.least_squares.

    Use bounds: alpha in [1e-6, 1.0], rho in (-1, 1), nu in [1e-6, 5.0].
    Initial guess: alpha=0.03, rho=0.0, nu=0.5.

    Returns (alpha_fit, rho_fit, nu_fit).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    F, T, beta = 0.025, 1.0, 0.5
    true = dict(alpha=0.02, rho=-0.30, nu=0.45)
    strikes = np.array([0.015, 0.020, 0.025, 0.030, 0.035])

    market_vols = sabr_vols(strikes, F, T,
                            alpha=true["alpha"], beta=beta,
                            rho=true["rho"], nu=true["nu"])
    assert market_vols.shape == (5,)
    assert abs(market_vols[2] - 0.128089) < 1e-5  # ATM check

    a_fit, r_fit, n_fit = calibrate_sabr(strikes, market_vols, F, T, beta)
    assert abs(a_fit - 0.02 ) < 1e-4, a_fit
    assert abs(r_fit - -0.30) < 1e-4, r_fit
    assert abs(n_fit - 0.45 ) < 1e-4, n_fit

    fit_vols = sabr_vols(strikes, F, T, a_fit, beta, r_fit, n_fit)
    rmse_bp = float(np.sqrt(((fit_vols - market_vols) ** 2).mean()) * 1e4)
    assert rmse_bp < 1.0, f"calibration RMSE too high: {rmse_bp} bp"

    print(f"market vols:  {market_vols.round(6)}")
    print(f"alpha_fit:    {a_fit:.6f}")
    print(f"rho_fit:     {r_fit:.6f}")
    print(f"nu_fit:       {n_fit:.6f}")
    print(f"fit rmse:     {rmse_bp:.4f} bp")
    print("\n✓ All checks passed.")
