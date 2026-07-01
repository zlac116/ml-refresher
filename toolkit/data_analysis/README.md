# toolkit/data_analysis — Task-indexed drills

Drill scripts in the same exercise format as `toolkit/data_analysis/mini_drills/`:
docstring with OBJECTIVE/TIME/TOPICS/EXPECTED OUTPUT/GRADING → setup → TODO stubs → `__main__` grader.

Each drill replaces the previous notebook (`numpy_scipy.ipynb`, `pandas.ipynb`, etc.) with the **same content** in a runnable Python script. Implement each function, then `uv run <file>` to grade yourself.

## Drills

| Drill | Time | Tasks |
|---|---|---|
| `numpy_scipy_drill.py` | 60–90 min | array creation, indexing, axis reductions, np.where, broadcasting, np.linalg, RNG, scipy.stats moments + tests + distributions, cumulative ops, view-vs-copy |
| `pandas_drill.py`      | 60–90 min | CSV round-trip, time-slice, boolean masks, sort multi-key, missing data, groupby + transform, rolling, resample, pivot, melt |
| `plotting_drill.py`    | 60–90 min | fig/ax pattern, subplots grid, bar/hist, time-axis formatting, twin axes, annotations, save |
| `sklearn_drill.py`     | 60–90 min | Pipeline, TimeSeriesSplit, neg_log_loss sign, predict_proba, CalibratedClassifierCV, permutation_importance, joblib save/load |

## Naming convention

Files are suffixed `_drill.py` (not just `pandas.py`) to avoid shadowing the actual library imports — Python prepends the script's directory to `sys.path`, so a file called `pandas.py` would block `import pandas`.

## How to run

```bash
cd /home/zlac116/Code/learning/ml-revision/toolkit/data_analysis
uv run numpy_scipy_drill.py        # raises NotImplementedError until you implement
```

## Required packages (per drill)

- `numpy_scipy_drill.py`: `numpy`, `scipy`
- `pandas_drill.py`: `numpy`, `pandas`
- `plotting_drill.py`: `numpy`, `pandas`, `matplotlib`
- `sklearn_drill.py`: `numpy`, `scikit-learn`, `joblib`

`uv add` whichever are missing in your active environment.
