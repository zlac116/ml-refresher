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

## EDA workflow (`eda_playbook.ipynb` + `eda_decisions.md`)

Two companion artefacts for the *workflow* of exploratory analysis, sitting
between the teaching pipelines and the library cheatsheets:

| File | Purpose |
|---|---|
| `eda_playbook.ipynb` | Runnable 14-step checklist. One cell per check, one verdict per output. Demonstrated on `data/crypto_hourly.parquet`; swap the data path + columns to re-use on any dataset. |
| `eda_decisions.md` | Printable wall-poster: the **finding → ML decision** table, ADF/KPSS verdict matrix, test → question lookup, p-value mnemonic, and bias rules for time series. |

Run the playbook on any new dataset before touching modelling. Use the
decision table to translate verdicts into actions.

## Practice projects (`drills/`)

Each cheatsheet has a matching practice project in reference-topic dirs (`toolkit/data_analysis/`, `toolkit/data_science/`) — a
real quant question that forces 70-80% of the cheatsheet's patterns naturally.
See `toolkit/data_analysis/PROJECTS_README.md` for the full list and suggested sequence.
