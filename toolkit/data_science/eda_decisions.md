# EDA → Decision Wall-Poster

The 14-item core workflow. Run top-to-bottom on any new dataset. Companion to
`eda_playbook.ipynb`, which executes every step on `data/crypto_hourly.parquet`.

---

## Universal core (run on every problem)

| # | Step | One-line check | Decision rule |
|---|---|---|---|
| 1 | Missing data | `df.isna().mean()` | `>30%` drop column · `5–30%` impute · `<5%` drop rows |
| 2 | Cardinality + dtype | `df.nunique()`, `df.dtypes` | Drop near-IDs · cast obvious numerics · low-card → `category` |
| 3 | Target distribution | `describe()`, hist, `jarque_bera` | `|skew|>0.5` → log/Box-Cox · `excess_kurt>3` → MAE/Huber over MSE |
| 4 | Feature ↔ target | `df.corr()['y']`; `groupby('class').mean()` | Low `|r|` does **not** preclude non-linear signal — keep for trees |
| 5 | Multicollinearity | VIF, `df.corr()` heatmap | `VIF>5` drop one or use ridge/lasso |

## Regression (post-fit on residuals)

| # | Step | Test | If reject |
|---|---|---|---|
| 6 | Linearity | residual vs fitted scatter | Add polynomial / splines, or use trees |
| 7 | Heteroscedasticity | `het_breuschpagan(resid, X)` | Use HC3 SEs (`cov_type='HC3'`) or transform target |
| 8 | Residual normality | `jarque_bera(resid)` | OK for point estimates; bootstrap CIs instead of parametric |

## Classification

| # | Step | Test | If skewed/hard |
|---|---|---|---|
| 9 | Class balance | `y.value_counts(normalize=True)` | `<30/70` → AUC/F1 (not accuracy), `class_weight='balanced'`, stratified split |
| 10 | Per-class separability | single-feature AUC per feature | All `<0.55` → genuinely hard · best `>0.7` → easy problem |

## Time series

| # | Step | Test | Action |
|---|---|---|---|
| 11 | Stationarity | `adfuller` **and** `kpss` | Both must agree — see verdict matrix below |
| 12 | Autocorrelation | `plot_acf`, `plot_pacf` (lags 0..50) | Significant spikes → AR/MA terms; PACF cut at `p` → AR(p) |
| 13 | Seasonality | `seasonal_decompose(period=N)` | Strong seasonal → Fourier features or SARIMA |
| 14 | Volatility clustering | ACF of `y**2` | Persistent → GARCH or rolling-vol feature |

> **#14 trap:** raw returns can look uncorrelated while squared returns are
> strongly autocorrelated. Always check both.

---

## ADF + KPSS verdict matrix

|                                | KPSS p < 0.05 (rejects stationarity) | KPSS p > 0.05 (does not reject) |
|---|---|---|
| **ADF p < 0.05** (rejects unit root) | Difference- or trend-stationary; decompose first | **Stationary** ✓ |
| **ADF p > 0.05** (does not reject)   | **Non-stationary** — needs differencing | Inconclusive — likely trend-stationary; detrend then retest |

---

## EDA finding → ML decision

| Finding | Affects | Decision |
|---|---|---|
| Skewed target | Loss, transform | Log target OR MAE/Huber loss |
| Heavy-tailed residuals | Loss, CIs, model class | Quantile/Huber loss; bootstrap CIs; trees over linear |
| Class imbalance | Split, metric, weights | Stratified split; AUC/F1; `class_weight='balanced'` |
| Multicollinearity (VIF>5) | Model class, regularisation | Ridge/lasso, or drop redundant features |
| High-cardinality categorical | Encoding | Target encoding (with CV) over one-hot |
| Non-stationary series | Feature engineering | Diff, returns, or detrend before modelling |
| Strong autocorrelation | CV scheme | `TimeSeriesSplit` / purged CV — **never** shuffle |
| Volatility clustering | Features | Rolling-vol features; consider GARCH |
| Heteroscedasticity | SEs / loss | HC3 SEs for inference; weighted least squares |
| Missing > 30% | Preprocessing | Drop column entirely |
| Outliers in features | Preprocessing | Robust scaling, winsorize, or use trees |
| Two-sample test rejects (KS) | Architecture | Regime-conditional models |
| JB rejects normality | CIs, tests, risk | Bootstrap CIs; non-parametric tests; historical/EVT VaR |

---

## Test → question lookup

| Question | Test | Module |
|---|---|---|
| Same distribution? | `ks_2samp` | `scipy.stats` |
| Same mean? | `ttest_ind(equal_var=False)` (Welch) | `scipy.stats` |
| Same variance? | `levene` | `scipy.stats` |
| Same median (non-parametric)? | `mannwhitneyu` | `scipy.stats` |
| Normal distribution? | `jarque_bera` (large n), `shapiro` (small n) | `scipy.stats` |
| Stationary? | `adfuller` + `kpss` | `statsmodels.tsa.stattools` |
| Cointegrated? | `coint` | `statsmodels.tsa.stattools` |
| Granger causality? | `grangercausalitytests` | `statsmodels.tsa.stattools` |
| Heteroscedastic? | `het_breuschpagan` | `statsmodels.stats.diagnostic` |
| White-noise residuals? | `acorr_ljungbox` | `statsmodels.stats.diagnostic` |

---

## p-value mnemonic

> **Low p → reject null → effect IS detected → samples ARE different.**
> **High p → fail to reject → no detected effect.**
>
> *"p is the probability of seeing data this extreme **if the null were true**."*

A small p-value is evidence **against** H₀, never for it.

---

## Bias rules (time series)

- Any reference statistic in a feature must be **past-only**: `.shift(1).rolling(W)`, never `.rolling(W)` alone.
- Any CV split must be **chronological**: `TimeSeriesSplit`, never `KFold(shuffle=True)`.
- Forward targets via `.shift(-h)` are fine **only** if the train/val split happens *after* the shift, with a purge of `h` rows at the boundary.

> **Default operator: `.shift(1).rolling(W)`. An unshifted `.rolling()` should look physically wrong.**
