# ML

Core machine learning — supervised models from regression through neural
networks, with one capstone per subject area.

## Subjects (in learning order)

| # | Subject | Focus |
|---|---|---|
| 01 | [`01_regression/`](01_regression/) | Linear, regularised, GLM-family — simplest entry point |
| 02 | [`02_classification/`](02_classification/) | Logistic, tree-based, naive Bayes, evaluation metrics |
| 03 | [`03_time_series/`](03_time_series/) | Stationarity, ARIMA, exponential smoothing, forecasting |
| 04 | [`04_neural_networks/`](04_neural_networks/) | PyTorch MLPs, training loop, early stopping, two capstones |

## Per-subject structure

Each subject dir contains:
- `<subject>.ipynb` — narrative walkthrough (the "tutorial").
- `<subject>_cheatsheet.md` — code-first reference.
- Capstones (subject-specific, where applicable — currently only `04_neural_networks/` has capstones).

## Reading order

1. Open the subject's tutorial notebook.
2. Reach for the cheatsheet when the notebook moves too fast.
3. Attempt the subject's capstone (where applicable).
4. After, read the capstone's `LESSONS.md` to lock in the patterns.

## Related

- [`toolkit/ml_project_methodology.md`](../toolkit/ml_project_methodology.md) — the universal phase-by-phase pipeline.
- [`toolkit/eda_decisions.md`](../toolkit/eda_decisions.md) — model selection upstream.
- [`toolkit/mlflow_cheatsheet.md`](../toolkit/mlflow_cheatsheet.md) — tracking + registry.
- [`quant_finance/`](../quant_finance/) — ML applied to financial models.
