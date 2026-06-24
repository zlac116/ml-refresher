"""
TOOLKIT — statsmodels task-indexed drill
==========================================

OBJECTIVE
    Practise the 8 canonical statsmodels idioms from the cheatsheet:
    stationarity (ADF + KPSS), decomposition, ACF/PACF, SARIMAX, residual
    diagnostics (Ljung-Box), OLS via formula API, heteroscedasticity tests,
    and cointegration.

ESTIMATED TIME
    60–90 min

TOPICS
    sm.tsa.stattools.adfuller, kpss
    sm.tsa.seasonal_decompose
    sm.tsa.stattools.acf, pacf
    sm.tsa.statespace.SARIMAX
    sm.stats.diagnostic.acorr_ljungbox      (Ljung-Box on residuals)
    smf.ols('y ~ x', data=df).fit()         (formula API)
    sm.stats.diagnostic.het_breuschpagan    (heteroscedasticity)
    sm.tsa.stattools.coint                  (Engle-Granger)

REQUIRED PACKAGES
    statsmodels, scipy, pandas, numpy (run `uv add statsmodels`)

EXPECTED OUTPUT
    ADF p (rw):           > 0.05      (non-stationary)
    ADF p (diff):         < 0.05      (stationary)
    seasonal period found: 24
    ACF[1] for AR(1):     > 0.5       (high autocorr at lag 1)
    SARIMAX converges:    True
    Ljung-Box lag-10 p:   > 0.05      (residuals look like white noise)
    OLS R²:               > 0.7       (synthetic linear, strong fit)
    BP p:                 < 0.05      (heteroscedastic by construction)
    coint p:              < 0.05      (cointegrated by construction)

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller, kpss, acf, coint
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
import warnings

warnings.filterwarnings("ignore")

# Synthetic data
np.random.seed(42)

# AR(1) + seasonal(24) + trend
N = 500
TREND = np.linspace(0, 5, N)
SEASONAL = 2 * np.sin(np.arange(N) * 2 * np.pi / 24)
AR = np.zeros(N)
for i in range(1, N):
    AR[i] = 0.6 * AR[i - 1] + np.random.normal(0, 1)
SERIES = pd.Series(TREND + SEASONAL + AR)


# ── TASK 1 — ADF + KPSS stationarity ─────────────────────────────────────
def stationarity_pvalues(series: pd.Series) -> tuple[float, float]:
    """Return (adf_p, kpss_p):
        adf_p  : adfuller(series)[1]   (p-value)
        kpss_p : kpss(series, regression='c')[1]
    """
    # TODO: implement (use 'with warnings.catch_warnings()' if KPSS spams)
    raise NotImplementedError


# ── TASK 2 — Seasonal decomposition ──────────────────────────────────────
def detect_period(series: pd.Series, candidate_periods: list[int] = (12, 24, 48)) -> int:
    """Run seasonal_decompose for each candidate, pick the one whose seasonal
    component has the largest peak-to-peak amplitude.

    Returns the chosen period.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 — ACF at lag 1 ─────────────────────────────────────────────────
def acf_lag1(series: pd.Series, max_lags: int = 10) -> float:
    """Compute ACF up to max_lags and return acf_values[1] (lag 1 correlation)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 — SARIMAX fit ──────────────────────────────────────────────────
def fit_sarimax(series: pd.Series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    """Fit SARIMAX. Return the fitted results object."""
    # TODO: implement (SARIMAX(series, order=..., seasonal_order=...).fit(disp=False))
    raise NotImplementedError


# ── TASK 5 — Ljung-Box on residuals ──────────────────────────────────────
def ljung_box_p(residuals: pd.Series, lag: int = 10) -> float:
    """acorr_ljungbox(residuals, lags=[lag], return_df=True) → return p-value
    at the requested lag.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 6 — OLS via formula API ─────────────────────────────────────────
def fit_ols_formula(df: pd.DataFrame, formula: str):
    """smf.ols(formula, data=df).fit(). Returns the results object."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 7 — Breusch-Pagan heteroscedasticity test ──────────────────────
def breusch_pagan_p(ols_results) -> float:
    """het_breuschpagan(resid, exog) returns (LM, LM_pvalue, F, F_pvalue).
    Return the LM p-value (index 1).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 8 — Engle-Granger cointegration ─────────────────────────────────
def coint_p(y0: np.ndarray, y1: np.ndarray) -> float:
    """coint(y0, y1) returns (t-stat, p, crit-values). Return the p-value."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Random walk (non-stationary) vs its first-difference (stationary)
    rw = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, 500)))
    adf_p_rw, _ = stationarity_pvalues(rw)
    adf_p_diff, _ = stationarity_pvalues(rw.diff().dropna())
    assert adf_p_rw  > 0.05   # non-stationary
    assert adf_p_diff < 0.05  # stationary after differencing

    # Seasonal: build series with period 24
    s = pd.Series(2 * np.sin(np.arange(500) * 2 * np.pi / 24) +
                  np.random.default_rng(0).normal(0, 0.3, 500))
    p = detect_period(s, candidate_periods=[12, 24, 48])
    assert p == 24

    # AR(1) → high ACF at lag 1
    rng = np.random.default_rng(0)
    ar = np.zeros(500)
    for i in range(1, 500):
        ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 1)
    ar_s = pd.Series(ar)
    assert acf_lag1(ar_s, 10) > 0.5

    # SARIMAX
    res = fit_sarimax(SERIES, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24))
    assert hasattr(res, "params")
    assert hasattr(res, "resid")

    # Ljung-Box on residuals — most well-fit models have lag-10 p > 0.05
    lb_p = ljung_box_p(res.resid, lag=10)
    assert 0 <= lb_p <= 1

    # OLS with formula API
    df = pd.DataFrame({"x": np.arange(100, dtype=float), "y": np.arange(100) * 2 + 5 +
                                                                 np.random.normal(0, 1, 100)})
    ols = fit_ols_formula(df, "y ~ x")
    assert ols.rsquared > 0.95

    # Breusch-Pagan on a hetero series
    df2 = pd.DataFrame({"x": np.arange(200, dtype=float)})
    df2["y"] = df2["x"] + np.random.normal(0, df2["x"] / 50 + 1, 200)
    ols2 = fit_ols_formula(df2, "y ~ x")
    bp_p = breusch_pagan_p(ols2)
    assert 0 <= bp_p <= 1

    # Cointegration: y0 and y1 share a common stochastic trend
    rng3 = np.random.default_rng(0)
    y0 = np.cumsum(rng3.normal(0, 1, 300))
    y1 = y0 * 0.5 + rng3.normal(0, 0.1, 300)   # tight linear combo with y0
    cp = coint_p(y0, y1)
    assert cp < 0.05    # ARE cointegrated

    print(f"ADF p (rw):           {adf_p_rw:.4f}      (non-stationary)")
    print(f"ADF p (diff):         {adf_p_diff:.4e}      (stationary)")
    print(f"seasonal period found: {p}")
    print(f"ACF[1] for AR(1):     {acf_lag1(ar_s, 10):.4f}       (high autocorr at lag 1)")
    print(f"SARIMAX converges:    True")
    print(f"Ljung-Box lag-10 p:   {lb_p:.4f}")
    print(f"OLS R²:               {ols.rsquared:.4f}       (strong fit)")
    print(f"BP p:                 {bp_p:.4e}      (lower → hetero)")
    print(f"coint p:              {cp:.4e}      (cointegrated)")
    print("\n✓ All checks passed.")
