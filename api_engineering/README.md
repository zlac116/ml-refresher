# API Engineering

Production REST APIs in Python — FastAPI ≥ 0.115, Pydantic v2, SQLAlchemy 2.0.

## Contents

- [`api_engineering_cheatsheet.md`](api_engineering_cheatsheet.md) — code-first reference. §0 leads with the universal endpoint pattern.
- [`tutorial/`](tutorial/) — guided walkthrough that builds the same patterns step-by-step.
- [`capstones/`](capstones/) — graded exercises.

## Capstones

| # | Capstone | What it teaches |
|---|---|---|
| 01 | [`capstones/01_trades_ledger/`](capstones/01_trades_ledger/) | Full layered service: auth + JWT/Argon2, BOLA-safe ownership, async SQLAlchemy 2.0, tests, Docker |

## Reading order

1. Read `api_engineering_cheatsheet.md` §0 (general pattern) — 5 minutes.
2. Walk through `tutorial/` end-to-end.
3. Attempt the trades_ledger capstone, hitting the cheatsheet for any section you're unsure of.
4. After, re-read §20 (anti-patterns table) — most mistakes you made are there.

## Related

- [`toolkit/ml_project_methodology.md`](../toolkit/ml_project_methodology.md) — the upstream ML side.
- [`toolkit/mlflow_cheatsheet.md`](../toolkit/mlflow_cheatsheet.md) — for ML-model-serving APIs.
- [`quant_finance/capstones/04_lmm_nn_surrogate/api_extension/`](../quant_finance/capstones/04_lmm_nn_surrogate/api_extension/) — production wrapping of an ML model via FastAPI + MLflow.
