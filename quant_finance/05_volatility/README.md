# 05 — Volatility

Estimating + modelling vol, beyond constant σ. The bridge between
realised-world (time-series) and risk-neutral (option-implied) vol.

## Learning notes (in order)

| # | File | Focus |
|---|---|---|
| 01 | [`01_garch.md`](01_garch.md) | GARCH(p,q), EGARCH; volatility clustering |
| 02 | [`02_realized_vol.md`](02_realized_vol.md) | Realised vol estimators; intraday data |
| 03 | [`03_local_vol_dupire.md`](03_local_vol_dupire.md) | Dupire's formula — local vol from option prices |

## Prerequisites

- `01_options/06_implied_vol_surface.md` — what we're matching for Dupire.
- `ml/03_time_series/` — GARCH builds on ARMA foundations.

## Related

- Crypto hourly data lives in [`../../data/crypto_hourly.parquet`](../../data/) — useful for realised-vol notes.
- [`01_options/07_heston.md`](../01_options/07_heston.md) — stochastic vol (local vol's richer cousin).
