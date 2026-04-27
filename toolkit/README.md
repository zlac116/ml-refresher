# Toolkit — task-indexed cheatsheets

Look up the idiom you need by **task**, not by function name. Each entry is:
question → canonical pattern → why + common mistake.

| Notebook | Topics |
|---|---|
| `pandas.ipynb` | loading, selection, sorting, NA, groupby, rolling, resample, reshape, merge, datetime, iteration |
| `numpy_scipy.ipynb` | array creation, indexing, reductions, vectorisation, broadcasting, linear algebra, RNG, scipy.stats |
| `plotting.ipynb` | the `fig, ax` pattern, common chart types, time axes, twin axes, seaborn, saving figures |
| `sklearn.ipynb` | pipelines, splits + CV, scoring strings, predict variants, common models, calibration, metrics |
| `statsmodels.ipynb` | ADF/KPSS, decomposition, ACF/PACF, exponential smoothing, SARIMA, residual diagnostics |
| `gradient_boosting.ipynb` | XGBoost + LightGBM fit/predict, hyperparameters, early stopping, categorical handling, quantile regression |
| `optuna.ipynb` | basic study, ML-realistic objective, pruning, persistence, multi-objective |
| `shap.ipynb` | TreeExplainer, the new Explanation API, beeswarm/bar/waterfall/scatter plots, KernelExplainer |
| `deployment.ipynb` | joblib bundles, Pydantic schemas (incl. `create_model`), FastAPI patterns, property-based tests |

Each notebook starts with a single setup cell — run it once, then jump to whichever entry you need.

Open any notebook with `make lab` (or `make notebook`) from the project root.
