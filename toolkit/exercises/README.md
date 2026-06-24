# toolkit/exercises — Practice projects

One project per cheatsheet, each forcing 70-80% of that cheatsheet's patterns
naturally on a real quant question. Most run on `data/crypto_hourly.parquet`.

## Format

These are now **Python scripts** in the same drill format as `data_analysis_drills/`:

- Top docstring: OBJECTIVE / ESTIMATED TIME / TOPICS / EXPECTED OUTPUT / GRADING
- Synthetic or real data set up at module scope
- 4–6 task stubs (`raise NotImplementedError` until you fill them in)
- Bottom block: assertions + printed expected output + "✓ All checks passed."

Each is timed at **≤ 30 min** when you know the cheatsheet — longer if learning.

```bash
cd /home/zlac116/Code/learning/ml-revision/toolkit/exercises
uv run numpy_scipy_project.py        # raises NotImplementedError until you implement
```

## Index

| Project | Question it answers | Cheatsheet exercised |
|---|---|---|
| [`pandas_project.py`](pandas_project.py) | Cross-sectional features + BTC vol regime → forward returns | groupby, transform, pivot, resample, rank |
| [`numpy_scipy_project.py`](numpy_scipy_project.py) | Per-asset drawdowns + bootstrap Sharpe CI | cumsum, np.maximum.accumulate, scipy.stats moments, bootstrap |
| [`plotting_project.py`](plotting_project.py) | 2×3 strategy tearsheet (equity / DD / Sharpe / hist / heatmap / scatter) | subplots, fill_between, annotations, imshow heatmap |
| [`sklearn_project.py`](sklearn_project.py) | Predict 4h direction with chronological split + permutation importance | Pipeline, time-series hygiene, permutation_importance |
| [`statsmodels_project.py`](statsmodels_project.py) | Stationarity / cointegration / Granger / OLS / Breusch-Pagan | ADF+KPSS, coint, grangercausalitytests, smf.ols, het_breuschpagan |
| [`gradient_boosting_project.py`](gradient_boosting_project.py) | 24h vol forecaster + monotonic constraint + p10/p90 coverage | lgb.LGBMRegressor, early_stopping, monotone_constraints, quantile objective |
| [`optuna_project.py`](optuna_project.py) | Hyperparameter sweep with TPE + MedianPruner + SQLite persistence | trial.suggest_*, TPESampler, MedianPruner, trials_dataframe |
| [`shap_project.py`](shap_project.py) | Explain the vol forecaster (rank, ablation, single-row) | TreeExplainer, mean|shap|, top-K ablation |
| [`deployment_project.py`](deployment_project.py) | Wrap a model as a stateless predict_one with order invariance + JSON logging | joblib bundle, pydantic.create_model, structured logs, uuid correlation IDs |

## Required packages

The full set spans several libraries — add as needed via `uv add`:

```bash
# Core (most projects)
uv add numpy pandas scipy matplotlib

# Per-project extras
uv add scikit-learn        # sklearn_project, deployment_project
uv add lightgbm            # gradient_boosting, optuna, shap projects
uv add optuna              # optuna_project
uv add shap                # shap_project
uv add statsmodels         # statsmodels_project
uv add joblib pydantic     # deployment_project
uv add pyarrow             # parquet support (used by all data-driven projects)
```

## How to use

1. Open the matching cheatsheet (e.g. `toolkit/data_analysis/pandas.ipynb`)
   on one screen, the project script on the other.
2. Read the top docstring + task signatures, then start implementing — DO NOT
   peek at the cheatsheet until you've tried each task from scratch.
3. `uv run <name>_project.py` — if assertions pass, you're done with that task.
4. If the assertion fails, the printed value tells you what your function
   returned vs what was expected.

Recognition collapses under retrieval; only recall survives.

## Suggested sequence

The pandas project is the canonical starting point — it covers the most
ground and the techniques transfer to all the others. After that, do the
projects in roughly the order you'd build a real ML pipeline:

```
pandas → numpy_scipy → plotting → sklearn → gradient_boosting
                                                  ↓
                  deployment ← shap ← optuna ←─┘
                       └─ statsmodels (independent econometrics track)
```

## When you're done

You're done with toolkit drilling when:

- You can read a task signature and verbalise the implementation plan in
  ~30s before writing code.
- You stop opening the cheatsheet for routine ops (groupby, resample,
  rolling, merge-on-index).
- You catch your own look-ahead bias by reflexively spotting `.rolling()`
  without `.shift(1)` in time-series contexts.
- You have *opinions* about when to reach for polars vs pandas, for LightGBM
  vs XGBoost, for SHAP vs permutation importance.

Then move to statistical-inference work or the longer-form capstones
(data engineering, MLflow, production ML).
