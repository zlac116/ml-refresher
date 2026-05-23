# Time-Series — End-to-End Cheat Sheet

Operational checklist for any time-series problem. **Time series breaks most ML assumptions** — random splits leak the future into the past, IID doesn't hold, residuals are autocorrelated. Treat this as a different discipline, not "regression but with a date column".

---

## Overall process

```
   ┌────────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌──────────────┐
   │ FRAME      │ ─►│ EXPLORE  │ ─►│ FEATURE │ ─►│ MODEL    │ ─►│ BACKTEST     │
   │ + CHRONO   │   │ STAT'TY  │   │ ENGINEER│   │ + TUNE   │   │ (WALK-       │
   │ SPLIT      │   │ DECOMPOSE│   │ (LAGS)  │   │          │   │  FORWARD)    │
   └────────────┘   └──────────┘   └─────────┘   └──────────┘   └──────────────┘
                                                                       │
                                                                       ▼
                                                    ┌──────────────────────────┐
                                                    │ DEPLOY + ROLL FORECASTS  │
                                                    │ + MONITOR DRIFT          │
                                                    └──────────────────────────┘
```

---

## 1. Frame the problem

| Question | Why it matters |
|---|---|
| **Univariate or multivariate?** | Multivariate → VAR, vector models, or feature-based ML |
| **One-step or multi-step horizon?** | Multi-step needs recursive or direct strategies |
| **Single series or hierarchy?** (e.g. SKUs within categories) | Hierarchical / reconciliation methods needed |
| **Are exogenous regressors available?** | SARIMAX / ML with covariates outperforms univariate |
| **What's the irreducible noise floor?** | Sets the realistic best-case error |
| **Is the problem really forecasting?** | If you want anomaly detection or change-point — different tools |

---

## 2. Explore the series

### Visual inspection (mandatory)

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(12, 9))
y.plot(ax=ax[0], title="Raw series")
y.rolling(30).mean().plot(ax=ax[1], title="30-day rolling mean")
y.diff().plot(ax=ax[2], title="1-step difference")
```

Look for: trend, seasonality, regime changes, outliers, missing periods, level shifts.

### Decomposition

```python
from statsmodels.tsa.seasonal import STL
stl = STL(y, period=12).fit()
stl.plot()
```

Components: **trend + seasonal + residual**. If residual looks random → decomposition captures the structure.

### Stationarity tests

| Test | Null hypothesis | Reject means |
|---|---|---|
| **ADF** (Augmented Dickey-Fuller) | Has a unit root (non-stationary) | Stationary |
| **KPSS** | Stationary | Non-stationary |
| **Use both** | — | They check different things — confirms with both |

```python
from statsmodels.tsa.stattools import adfuller, kpss
print(f"ADF p-value:  {adfuller(y)[1]:.4f}")    # want < 0.05
print(f"KPSS p-value: {kpss(y)[1]:.4f}")        # want > 0.05
```

**Why test stationarity:** classical models (ARIMA family) assume it. If non-stationary, difference until stationary (and use the differencing order `d` in ARIMA).

### Autocorrelation

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(y, lags=40)
plot_pacf(y, lags=40)
```

ACF and PACF reveal seasonality, ARIMA orders, and the relevant lag depth for ML features.

---

## 3. Chronological split — NEVER random

```python
n = len(y)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

y_tr = y.iloc[:train_end]
y_va = y.iloc[train_end:val_end]
y_te = y.iloc[val_end:]
```

**Why never random split:** random split shuffles future into training. Model "memorises" the future and gives unrealistically good metrics. Production behaviour will be catastrophically worse.

**Walk-forward validation** (the proper TS analogue of k-fold):

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, test_size=horizon)
for tr_idx, va_idx in tscv.split(X):
    # train on X[tr_idx], validate on X[va_idx]
```

Two flavours:
- **Expanding window** — training set grows each fold. Use when more history = better.
- **Rolling window** — training set stays fixed-size. Use when distant past is irrelevant or distribution shifts.

---

## 4. Feature engineering (for ML models on time series)

Classical models (ARIMA, ETS) need very little. ML models on time-series tabular features need a lot:

| Feature family | Examples |
|---|---|
| **Lags** | `y_{t-1}`, `y_{t-7}` (weekly), `y_{t-365}` (annual) — match dominant period |
| **Rolling stats** | rolling mean / std / min / max over windows (7, 14, 30) |
| **Differences** | `y_t − y_{t-1}` (1st-order), `y_t − y_{t-7}` (seasonal) |
| **Calendar** | dayofweek, month, quarter, hour, is_weekend, is_holiday |
| **Cyclical encoding** | sin / cos of dayofweek, month (preserves cyclicity) |
| **Exogenous lags** | lagged versions of related series (weather, marketing spend) |
| **Time since event** | days since promo, time since last spike |
| **Fourier terms** | for multiple seasonalities |

**Critical:** when computing rolling/lag features, the value at time `t` must only use information available at `t`. `df['y'].rolling(7).mean()` is fine; `df['y'].rolling(7, center=True).mean()` LEAKS the future.

```python
df["lag_1"] = df["y"].shift(1)             # OK
df["roll7"] = df["y"].shift(1).rolling(7).mean()   # shift before rolling avoids look-ahead
```

**Why `shift(1)` before rolling:** when you predict at time `t`, you don't yet know `y_t`. Rolling without shifting includes `y_t` in the mean — leakage.

---

## 5. Model selection

| Family | When to use | Strengths | Weaknesses |
|---|---|---|---|
| **Naive (last value)** | Always train as baseline | Trivial, surprisingly strong | No trend or seasonality |
| **Seasonal naive** | Strong seasonality, no trend | Trivial, often beats fancy models | Ignores trend |
| **Drift / linear extrapolation** | Steady trend | Simple | Ignores seasonality |
| **ETS (Exponential Smoothing)** | Trend + seasonality, low noise | Robust, fast, no stationarity needed | Univariate only |
| **ARIMA / SARIMA** | Stationary or with differencing | Classic, well-understood | Univariate; manual order selection |
| **SARIMAX** | + exogenous regressors | Adds covariates to SARIMA | Slow on big data |
| **Prophet** | Business series with holidays | Easy, auto-detects multi-seasonality | Black-box, slow at scale |
| **TBATS** | Multiple seasonalities | Handles non-integer periods | Slow |
| **VAR** | Multivariate stationary | Captures cross-series feedback | Needs stationarity, blows up in dim |
| **State-space / Kalman** | Latent dynamics, missing data | Principled uncertainty, online updates | Set-up cost |
| **ML on lag features** (XGBoost, LightGBM) | Many series, exogenous features | Scales, flexible, accurate | No native uncertainty, needs feature engineering |
| **N-BEATS / N-HiTS** | Univariate, plenty of data | SOTA on M4-style competitions | Compute-heavy |
| **DeepAR / Temporal Fusion Transformer** | Many related series, covariates | Probabilistic, multivariate | Compute-heavy |
| **LSTM / GRU** | Long sequences, complex patterns | Captures non-linear dynamics | Hungry for data, no native uncertainty |

**Rule of thumb:** always train **Naive + Seasonal-Naive + Auto-ARIMA** as baselines. If your fancy model can't beat them, the fancy model is wrong somewhere. The naive baselines win surprisingly often.

---

## 6. Forecasting strategies for multi-step horizons

| Strategy | How it works | When to use |
|---|---|---|
| **Recursive** | Forecast 1 step, feed back as input, repeat | Default, simple |
| **Direct** | Train separate model per horizon (h=1, h=2, ..., h=H) | Avoids error accumulation; expensive |
| **Multi-output (DirRec)** | One model outputs H values | Captures correlations across horizons |
| **Seq2seq** | Encoder-decoder neural net | Long horizons, lots of data |

**Recursive accumulates error** (each step's noise feeds the next). **Direct doesn't**, but you train H models. For short horizons (≤ 24 steps), recursive is fine. For long horizons or critical accuracy, direct.

---

## 7. Hyperparameter tuning

Use `TimeSeriesSplit` instead of KFold. Otherwise the routine is the same as regression — Optuna for non-trivial search spaces.

```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
gs = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_root_mean_squared_error")
```

For ARIMA: use `pmdarima.auto_arima()` instead of grid search — it uses information criteria (AIC / BIC) and is much faster.

---

## 8. Evaluation metrics

| Metric | Notes |
|---|---|
| **RMSE / MAE** | Same as regression, computed per-step or per-horizon |
| **MAPE / sMAPE** | % error; standard in business forecasting, breaks at y near 0 |
| **MASE** (Mean Absolute Scaled Error) | Error / naive-baseline error; < 1 means beating naive. **Recommended.** |
| **CRPS** (Continuous Ranked Probability Score) | Probabilistic accuracy — proper scoring rule |
| **Pinball loss** | Quantile-specific, for prediction intervals |

**Always report per-horizon** errors, not just overall mean. A model great at h=1 and terrible at h=24 is very different from one that's mediocre throughout.

**MASE is the most honest metric** for time series — it's normalised against the naive baseline. MASE = 1.0 means "exactly as good as predicting yesterday's value". Lower is better; > 1 means worse than naive.

---

## 9. Backtesting (the proper TS evaluation)

Pick a backtest window — say the last year — and at each point in that window:

1. Train on data up to that point
2. Forecast h steps ahead
3. Compare to actuals
4. Roll forward one step (or one horizon)

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

tscv = TimeSeriesSplit(n_splits=12, test_size=horizon)
errors = []
for tr_idx, te_idx in tscv.split(y):
    model.fit(y.iloc[tr_idx])
    pred = model.forecast(horizon)
    errors.append(np.abs(pred - y.iloc[te_idx]))
```

**Why backtesting matters:** a single train/test number is one data point. A backtest gives a *distribution* of out-of-sample performance — more reliable, exposes regime changes.

---

## 10. Uncertainty intervals

Point forecasts alone are insufficient for any real decision. Get intervals via:

| Method | Pros | Cons |
|---|---|---|
| **ARIMA / ETS analytical** | Closed form from fitted model | Assumes normal residuals |
| **Bootstrap residuals** | Distribution-free | Slow |
| **Quantile regression** (GBM) | Direct P10/P50/P90 | Multiple models per horizon |
| **Conformal prediction** | Distribution-free, valid finite-sample | Slightly conservative |
| **Bayesian (Prophet, PyMC)** | Full posterior | Slow |
| **DeepAR / TFT** | Native probabilistic | Compute-heavy |

For most production cases, **quantile gradient boosting on lag features** is the practical default.

---

## 11. Common gotchas (TS-specific)

- **Random split** — uses future to predict past. Forever the #1 mistake.
- **Centered rolling windows** — `rolling(7, center=True)` uses 3 future points. Leakage.
- **Resampling that includes the current row** — `cumsum()`, `expanding()` include `y_t`; shift first.
- **Train-test gap** — if you predict at `t`, but features at `t` depend on `t-h+1` not yet observed in production, you can't actually compute them.
- **Calendar features without holidays** — model can't see Christmas dip; add holiday flags.
- **Ignoring multiple seasonalities** — hourly data has daily, weekly, AND annual seasonality. Use Fourier terms.
- **Treating stationarity as a binary** — many series are seasonally stationary but trend non-stationary. Use seasonal differencing too.
- **Reporting RMSE only at horizon 1** — mean over horizons hides the truth. Report per-horizon.
- **Forgetting timezone / DST** when joining hourly series across regions.

---

## Quick-reference

| Decision | Default | Switch when |
|---|---|---|
| Split | Chronological 70/15/15 | Always |
| CV | TimeSeriesSplit (expanding) | Distribution shifts → rolling |
| Baseline | Naive + Seasonal-Naive + Auto-ARIMA | Always train all three |
| Default classical model | ETS or SARIMA | Multiple seasonalities → TBATS / Prophet |
| Default ML model | LightGBM on lag features | Many related series → DeepAR / TFT |
| Stationarity check | ADF + KPSS | Always |
| Lag features | Auto-set by ACF / PACF peaks | Domain-driven: 1, 7, 30, 365 |
| Multi-step strategy | Recursive | Long horizon / critical accuracy → direct |
| Primary metric | MASE + per-horizon RMSE | Probabilistic forecast → CRPS |
| Intervals | Quantile GBM (P10/P50/P90) | Classical model → analytical PI |
| Backtest | TimeSeriesSplit 5–12 folds | Always |

---

## What to remember

1. **Chronological split, always.** Random splits leak the future and inflate metrics. Walk-forward backtesting is the only honest evaluation for time series.
2. **Never use information from time `t` in features at time `t`.** Centered rolling, expanding without `shift(1)`, target encoding without temporal CV — all are silent leakage traps.
3. **Beat the naive baseline before celebrating.** MASE < 1 is the bar. Naive and Seasonal-Naive forecasts beat most ML models on most real series — if yours doesn't, the fancy model is wrong somewhere.
