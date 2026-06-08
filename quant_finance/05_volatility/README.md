# 05 — Volatility

Estimating + modelling vol, beyond constant σ. The bridge between
realised-world (time-series) and risk-neutral (option-implied) vol.

## Notebooks (in order)

| # | Notebook | Focus |
|---|---|---|
| 01 | `01_garch.ipynb` | GARCH(p,q), EGARCH; volatility clustering |
| 02 | `02_realized_vol.ipynb` | Realised vol estimators; intraday data |
| 03 | `03_local_vol_dupire.ipynb` | Dupire's formula — local vol from option prices |

## Prerequisites

- `01_options/06_implied_vol_surface.ipynb` — what we're matching for Dupire.
- `ml/03_time_series/` — GARCH builds on ARMA foundations.

## Related

- Crypto hourly data lives in [`../../data/crypto_hourly.parquet`](../../data/) — useful for realised-vol notebooks.
- [`01_options/07_heston.ipynb`](../01_options/07_heston.ipynb) — stochastic vol (local vol's richer cousin).
