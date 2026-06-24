"""
PROJECT — statsmodels: Stationarity, Cointegration, OLS Diagnostics
=====================================================================

OBJECTIVE
    Apply classic econometric tests to the crypto panel:

      1. ADF + KPSS verdict matrix on each symbol's log-price + log-return.
      2. Pairwise cointegration test (Engle-Granger) across the 4 symbols.
      3. Granger causality at lags 1, 4, 24 (does BTC lead ETH? vice versa?).
      4. OLS via formula API: ETH return on BTC return + controls.
      5. Heteroscedasticity test (Breusch-Pagan) on OLS residuals.

ESTIMATED TIME
    30 min

TOPICS
    sm.tsa.stattools.adfuller, kpss, coint
    sm.tsa.stattools.grangercausalitytests
    smf.ols('y ~ x', data=df).fit()
    sm.stats.diagnostic.het_breuschpagan

REQUIRED PACKAGES
    statsmodels, pandas, numpy (run `uv add statsmodels`)

INTERPRETATION CONVENTIONS
    ADF p<0.05  → reject unit root → stationary
    KPSS p<0.05 → reject stationarity around constant → has unit root
    Engle-Granger coint p<0.05 → series ARE cointegrated
    Granger p<0.05 → reject null of "no causality"

EXPECTED OUTPUT
    log price ADF (BTC):     p > 0.05    (non-stationary, as expected)
    log return ADF (BTC):    p < 0.01    (stationary, as expected)
    coint matrix shape:      (4, 4) symmetric
    OLS R²:                  > 0.4       (ETH explained meaningfully by BTC)
    Breusch-Pagan p:         likely < 0.05 (heteroscedastic residuals)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller, kpss, coint, grangercausalitytests
from statsmodels.stats.diagnostic import het_breuschpagan

DATA = "/home/zlac116/Code/learning/ml-revision/data/crypto_hourly.parquet"

_df = pd.read_parquet(DATA).sort_values(["symbol", "ts"]).reset_index(drop=True)
# Daily close per symbol (less noisy than hourly for these tests)
daily = _df.set_index("ts").groupby("symbol")["close"].resample("1D").last()
prices = daily.unstack("symbol").dropna()
log_prices = np.log(prices)
log_returns = log_prices.diff().dropna()


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def adf_kpss_verdict(series: pd.Series) -> dict:
    """Run BOTH adfuller and kpss on `series`. Return dict:
        adf_p     : ADF p-value (lower → stationary)
        kpss_p    : KPSS p-value (lower → non-stationary)
        verdict   : 'stationary' | 'non-stationary' | 'inconclusive'
                    (stationary if ADF p<0.05 AND KPSS p>0.05)

    Hint: kpss returns (stat, p, lags, crit_values); use kpss(series, regression='c').
    Suppress the InterpolationWarning if you wish.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def cointegration_matrix(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Engle-Granger cointegration test (statsmodels.tsa.stattools.coint).

    Returns a (n_symbols x n_symbols) DataFrame of p-values; diagonal = NaN; symmetric.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def granger_p_value(returns_df: pd.DataFrame, cause: str, effect: str, maxlag: int) -> float:
    """Test whether `cause` Granger-causes `effect` up to `maxlag` lags.

    Returns the minimum p-value across lags 1..maxlag (the F-test p-value
    from the 'ssr_ftest' result of grangercausalitytests).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def fit_ols(returns_df: pd.DataFrame, y_symbol: str = "ETH", x_symbol: str = "BTC"):
    """Fit OLS: y_symbol ~ x_symbol via smf.ols('Y ~ X', data=df).fit().

    Returns the fitted results object.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def breusch_pagan(ols_results) -> float:
    """Run het_breuschpagan on the OLS residuals + exog matrix.

    Returns the p-value of the LM test (index 1 of the return tuple).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    btc_price_verdict = adf_kpss_verdict(log_prices["BTC"])
    btc_ret_verdict   = adf_kpss_verdict(log_returns["BTC"])
    assert btc_price_verdict["adf_p"] > 0.05      # non-stationary
    assert btc_ret_verdict["adf_p"] < 0.05        # stationary
    assert btc_ret_verdict["verdict"] == "stationary"

    cm = cointegration_matrix(log_prices)
    assert cm.shape == (4, 4)
    assert cm.index.tolist() == sorted(prices.columns.tolist())
    # Diagonal should be NaN
    assert all(pd.isna(cm.iloc[i, i]) for i in range(4))

    gp_btc_eth = granger_p_value(log_returns, cause="BTC", effect="ETH", maxlag=4)
    assert 0 <= gp_btc_eth <= 1

    res = fit_ols(log_returns, "ETH", "BTC")
    assert hasattr(res, "rsquared")
    assert res.rsquared > 0.40                    # BTC explains a lot of ETH

    bp_p = breusch_pagan(res)
    assert 0 <= bp_p <= 1

    print(f"log price ADF (BTC):     p = {btc_price_verdict['adf_p']:.4f}    (non-stationary)")
    print(f"log return ADF (BTC):    p = {btc_ret_verdict['adf_p']:.4e}    (stationary)")
    print(f"coint matrix shape:      {cm.shape}")
    print(f"BTC →  ETH Granger p:    {gp_btc_eth:.4f}")
    print(f"OLS R² (ETH ~ BTC):      {res.rsquared:.4f}")
    print(f"Breusch-Pagan p:         {bp_p:.4e}")
    print("\n✓ All checks passed.")
