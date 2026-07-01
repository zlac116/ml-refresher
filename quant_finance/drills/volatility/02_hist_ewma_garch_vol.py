"""
VOL 2 — Historical vs EWMA vs GARCH(1,1) Volatility
====================================================

OBJECTIVE
    On a return series with vol clustering, compute three vol estimators:
      1. Rolling 20-day historical (equal-weight).
      2. EWMA RiskMetrics (lambda = 0.94 daily).
      3. GARCH(1,1) with fixed parameters (omega, alpha, beta).

ESTIMATED TIME
    20 min

TOPICS
    Equal-weight rolling vs exponentially-weighted estimators
    RiskMetrics recursion:    sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}
    GARCH(1,1) recursion:     sigma^2_t = omega + alpha*r^2_{t-1} + beta*sigma^2_{t-1}
    Unconditional GARCH var = omega / (1 - alpha - beta)
    Annualisation = std * sqrt(252)

REAL-WORLD NOTE
    Most front-office risk systems run EWMA (lambda 0.94 daily / 0.97 monthly).
    GARCH adds a long-run mean-reverting term to capture fat tails.
    Modern practice usually FITS the GARCH parameters; here we use fixed
    canonical values for clarity.

REFERENCE
    RiskMetrics Technical Document (J.P. Morgan, 1996).
    Bollerslev (1986); Hull, ch. 23.

EXPECTED OUTPUT  (seed=42, regime-switching synthetic series of 500 obs)
    hist20 last:   0.552131
    ewma   last:   0.467323
    garch  last:   0.335051
    uncond vol:    0.365632

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

# ── synthetic returns with regime switching ────────────────────────────────
np.random.seed(42)
_n = 500
_vol_lo, _vol_hi = 0.01, 0.03
_regime = np.zeros(_n)
for _i in range(1, _n):
    _regime[_i] = 1 - _regime[_i - 1] if np.random.rand() < 0.02 else _regime[_i - 1]
_vols = np.where(_regime == 0, _vol_lo, _vol_hi)
rets = pd.Series(np.random.normal(0, _vols), name="ret")


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def hist_ann_vol(rets: pd.Series, window: int = 20) -> pd.Series:
    """Rolling equal-weight historical ann vol of `rets`."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def ewma_ann_vol(rets: pd.Series, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA ann vol.

    Recursion: sigma2_t = lam * sigma2_{t-1} + (1-lam) * r_{t-1}^2
    Init: sigma2_0 = r_0^2.   Returns numpy array shape (len(rets),).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def garch11_ann_vol(rets: pd.Series, omega: float = 1e-6,
                    alpha: float = 0.10, beta: float = 0.85) -> np.ndarray:
    """GARCH(1,1) ann vol with fixed parameters.

    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
    Init sigma2_0 = sample variance.
    Returns array shape (len(rets),).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    h = hist_ann_vol(rets)
    assert len(h) == len(rets)
    assert abs(h.iloc[-1] - 0.552131) < 1e-4

    e = ewma_ann_vol(rets, lam=0.94)
    assert isinstance(e, np.ndarray) and e.shape == (len(rets),)
    assert abs(e[-1] - 0.467323) < 1e-4

    g = garch11_ann_vol(rets, omega=1e-6, alpha=0.10, beta=0.85)
    assert g.shape == (len(rets),)
    assert abs(g[-1] - 0.335051) < 1e-4

    uncond = rets.std() * np.sqrt(252)
    assert abs(uncond - 0.365632) < 1e-4

    print(f"hist20 last:   {h.iloc[-1]:.6f}")
    print(f"ewma   last:   {e[-1]:.6f}")
    print(f"garch  last:   {g[-1]:.6f}")
    print(f"uncond vol:    {uncond:.6f}")
    print("\n✓ All checks passed.")
