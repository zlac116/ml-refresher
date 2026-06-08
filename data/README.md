# data/

Shared datasets used across multiple capstones and notebooks.

## Contents

- `crypto_hourly.parquet` — hourly OHLCV for selected crypto pairs. Used by notebooks in `ml/03_time_series/`, `quant_finance/05_volatility/`, and some `quant_finance/capstones/` projects.
- `fetch_crypto.py` — re-generates the parquet from a public API. Run if the file is missing or stale:
  ```bash
  uv run python data/fetch_crypto.py
  ```

## Conventions

- Only datasets that are **read by ≥2 modules** belong here. Single-use data lives with its capstone.
- Large datasets are **gitignored** — `fetch_crypto.py` (or each capstone's loader) is the source of truth for how to regenerate them.
- Format preference: parquet > csv > pickle. Parquet is columnar, compressed, typed, fast to load.

## Adding a new dataset

1. Add a fetcher script (`fetch_<name>.py`) that downloads + writes the parquet.
2. Document the schema briefly here (columns + dtypes).
3. Add the parquet pattern to `.gitignore` if it's > 1 MB.
