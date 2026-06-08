# 03 — Time Series

Forecasting + sequential dependence. Different evaluation discipline from
i.i.d. regression — no shuffled splits, no leakage from future to past.

## Files

- [`time_series.ipynb`](time_series.ipynb) — narrative walkthrough.
- [`time_series_cheatsheet.md`](time_series_cheatsheet.md) — code-first reference.

## What's covered

- Stationarity (ADF, KPSS), differencing.
- ARIMA / SARIMA / SARIMAX (exogenous).
- Exponential smoothing (Holt-Winters).
- Backtesting: walk-forward, expanding vs rolling window.
- Loss for forecasting: MAE / RMSE / MAPE / sMAPE / pinball (quantile).

## Prerequisites

- `01_regression/` for the model-fitting baseline.
- `fundamentals/mathematics.ipynb` for autocovariance intuition.

## Related

- [`quant_finance/05_volatility/`](../../quant_finance/05_volatility/) — GARCH and realized vol are the volatility cousins.
- Crypto data lives in [`data/crypto_hourly.parquet`](../../data/) for experimentation.

## Next

- `04_neural_networks/` — LSTMs / transformers when ARIMA is too rigid.
