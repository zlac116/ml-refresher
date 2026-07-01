# toolkit/data_science — Task-indexed drills

Drill scripts in the same exercise format as `toolkit/data_analysis/mini_drills/`:
docstring with OBJECTIVE/TIME/TOPICS/EXPECTED OUTPUT/GRADING → setup → TODO stubs → `__main__` grader.

Each drill replaces the previous notebook (`gradient_boosting.ipynb`, `optuna.ipynb`, etc.) with the **same content** in a runnable Python script. Implement each function, then `uv run <file>` to grade yourself.

## Drills

| Drill | Time | Tasks |
|---|---|---|
| `eda_playbook_drill.py`        | 60–90 min | shape/dtypes/missing, target balance, numeric summary, correlation pairs, baseline regression, Shapiro residual normality, ADF on residuals |
| `gradient_boosting_drill.py`   | 60–90 min | baseline LGBM + early stopping, best_iteration, gain vs split importance, quantile regression [p10/p90], monotonic constraints, joblib persistence |
| `optuna_drill.py`              | 60–90 min | toy study (TPE), ML objective with suggest_*, MedianPruner, SQLite persistence, multi-objective, trials_dataframe |
| `shap_drill.py`                | 60–90 min | TreeExplainer + shap_values, mean abs SHAP ranking, single-row explanation, slicing by feature value, interaction values |
| `statsmodels_drill.py`         | 60–90 min | ADF/KPSS stationarity, seasonal decomposition, ACF lag-1, SARIMAX fit, Ljung-Box on residuals, OLS via formula API, Breusch-Pagan, Engle-Granger cointegration |
| `deployment_drill.py`          | 60–90 min | joblib bundle, predict_one (stateless + reorder), Pydantic schema, JSON logger + uuid correlation IDs, async wrapper, FastAPI /health + /ready via TestClient |

## Companion reference docs (unchanged)

- `eda_decisions.md` — when to use which test (markdown reference)
- `ml_project_methodology.md` — overall ML project workflow (markdown reference)

## Naming convention

Files are suffixed `_drill.py` to avoid shadowing the actual library imports
(`shap.py` would shadow `import shap`, etc.).

## How to run

```bash
cd /home/zlac116/Code/learning/ml-revision/toolkit/data_science
uv run gradient_boosting_drill.py
```

## Required packages (per drill)

| Drill | uv add ... |
|---|---|
| `eda_playbook_drill` | `pandas scikit-learn statsmodels scipy` |
| `gradient_boosting_drill` | `lightgbm scikit-learn joblib` |
| `optuna_drill` | `optuna scikit-learn` |
| `shap_drill` | `shap lightgbm scikit-learn` |
| `statsmodels_drill` | `statsmodels scipy pandas` |
| `deployment_drill` | `joblib pydantic scikit-learn fastapi httpx` |

Run `uv add` for whichever are missing in your environment.
