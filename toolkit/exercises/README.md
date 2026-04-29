# toolkit/exercises — Practice projects

One project per cheatsheet, each forcing 70-80% of that cheatsheet's patterns
naturally on a real quant question. All run on `data/crypto_hourly.parquet`.

| Project | Question it answers | Cheatsheet exercised |
|---|---|---|
| `pandas_project.ipynb` | Conditional on BTC vol regime × momentum rank, what's the average forward 24h return? | groupby, MultiIndex, resample, pivot, qcut, method-chaining |
| `numpy_scipy_project.ipynb` | Per-asset drawdown attribution + bootstrap Sharpe CI | cumsum, np.maximum.accumulate, scipy.stats moments, KS/Welch tests |
| `plotting_project.ipynb` | Build a publication-quality strategy tearsheet | subplots grid, fill_between, annotations, log axes, colormap rules |
| `sklearn_project.ipynb` | Predict 4h direction with proper CV + calibration + threshold | Pipeline, TimeSeriesSplit, FrozenEstimator, permutation_importance |
| `statsmodels_project.ipynb` | Stationarity / cointegration / Granger across the 4-asset panel | ADF+KPSS, coint, Granger, SARIMAX + Ljung-Box, smf.ols formula |
| `gradient_boosting_project.ipynb` | Tune & deploy a 24h vol forecaster | early stopping, monotonic constraints, custom objective, quantile regression |
| `optuna_project.ipynb` | Hyperparameter sweep with proper budget discipline | TPE, MedianPruner, SQLite persistence, viz, multi-objective |
| `shap_project.ipynb` | Explain the vol forecaster end-to-end | TreeExplainer, beeswarm, dependence, waterfall, interaction values, top-K refit |
| `deployment_project.ipynb` | Wrap the model as a stateless predict_one with ops endpoints | joblib bundle, Pydantic via create_model, async, structured logs, /health + /ready |

## How to use

Each notebook has:

1. **Project goal** — what you'll deliver
2. **Why this exercises the cheatsheet** — concrete justification
3. **6-9 sub-tasks**, each with:
   - A description of what to compute
   - A bulleted list of patterns the sub-task forces
   - An empty code cell underneath
4. **What success looks like** — the bar to clear

Open the cheatsheet (e.g. `toolkit/pandas.ipynb`) on one screen, the project
notebook on the other. Don't peek at the cheatsheet until you've tried each
sub-task from a blank cell — recognition collapses under retrieval, only recall
survives.

## Suggested sequence

The pandas project is the canonical starting point — it covers the most
ground and the techniques transfer to all the others. After that, do the
projects in roughly the order you'd build a real ML pipeline: numpy_scipy
(analytics) → plotting (visualisation) → sklearn (modelling) →
gradient_boosting (better modelling) → optuna (tuning) → shap (interpretation)
→ statsmodels (classical TS work, often a baseline) → deployment (shipping).

## When you're done

You're done with toolkit drilling when:

- You can read a sub-task description and verbalise the pandas plan in 30s
  before writing code.
- You stop opening the cheatsheet for routine ops (groupby, resample, rolling, merge-on-index).
- You catch your own look-ahead bias by reflexively spotting `.rolling()`
  without `.shift(1)`.
- You have *opinions* about when to use polars, when to use pandas.

Then move to statistical-inference work (the next gap the experts flagged) or
to the longer-form projects (data engineering, MLflow).
