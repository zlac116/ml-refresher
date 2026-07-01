"""
PROJECT — Optuna: Hyperparameter Sweep with Budget Discipline
==============================================================

OBJECTIVE
    Tune a LightGBM vol forecaster using Optuna:

      1. Define the objective function (suggest hyperparameters, return val MAE).
      2. Run a study with TPE sampler + MedianPruner (early-stop bad trials).
      3. Persist the study to SQLite for resume capability.
      4. Inspect via trials_dataframe; print best params + value.
      5. (Optional) Multi-objective: MAE vs fit time.

ESTIMATED TIME
    25 min

TOPICS
    trial.suggest_float / suggest_int / suggest_categorical
    optuna.create_study(direction='minimize', sampler=TPESampler, pruner=MedianPruner)
    optuna.integration.LightGBMPruningCallback (or report+should_prune)
    storage='sqlite:///optuna.db' for resume
    study.trials_dataframe()

REQUIRED PACKAGES
    optuna, lightgbm, scikit-learn (run `uv add optuna lightgbm scikit-learn`)

EXPECTED OUTPUT
    n_trials run:         20  (small budget for the drill)
    best val MAE:         < 0.005
    best params printed:  (varies — TPE is stochastic but seeded)
    study persisted to:   /tmp/optuna_proj.db
"""
import os
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

DATA = "/home/zlac116/Code/learning/ml-revision/data/crypto_hourly.parquet"
DB_PATH = "/tmp/optuna_proj.db"


def _load_xy_split():
    """Minimal repeat of FI's feature build for self-contained tuning."""
    df = pd.read_parquet(DATA)
    df = df[df.symbol == "BTC"].sort_values("ts").reset_index(drop=True)
    df["ret_1h"] = df["close"].pct_change()
    df["vol_24h_trailing"] = df["ret_1h"].rolling(24).std()
    df["ret_4h"] = df["close"].pct_change(4)
    df["sma_ratio"] = df["close"] / df["close"].rolling(24).mean()
    df["target"] = df["ret_1h"].rolling(24).std().shift(-24)
    df = df.dropna()
    feats = ["ret_1h", "vol_24h_trailing", "ret_4h", "sma_ratio"]
    n = len(df); n_tr = int(n * 0.7); n_va = int(n * 0.85)
    return (df[feats].iloc[:n_tr].values,    df["target"].iloc[:n_tr].values,
            df[feats].iloc[n_tr:n_va].values, df["target"].iloc[n_tr:n_va].values)


X_TR, Y_TR, X_VA, Y_VA = _load_xy_split()


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna.

    Suggest these hyperparameters:
        num_leaves         : int in [16, 128]
        learning_rate      : float in [0.01, 0.3], log=True
        feature_fraction   : float in [0.6, 1.0]
        min_child_samples  : int in [5, 100]

    Fit LightGBM (objective='regression_l1', n_estimators=200, random_state=42)
    with eval_set=[(X_VA, Y_VA)] and early_stopping(20).
    Return val MAE.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def run_study(n_trials: int = 20, storage_path: str = DB_PATH) -> optuna.Study:
    """Create a study with:
        direction='minimize'
        sampler=TPESampler(seed=42)
        pruner=MedianPruner()
        storage=f'sqlite:///{storage_path}'  (so it's persisted)
        study_name='vol_forecaster'
        load_if_exists=True

    Run `study.optimize(objective, n_trials=n_trials, show_progress_bar=False)`.

    Returns the study object.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def study_summary(study: optuna.Study) -> dict:
    """Return a dict with:
        best_value     : best (lowest) val MAE
        best_params    : dict of best hyperparameters
        n_trials       : total trials completed
        n_pruned       : trials with state == TrialState.PRUNED
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Clean any pre-existing DB so the run is reproducible
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = run_study(n_trials=20)
    assert isinstance(study, optuna.Study)

    summary = study_summary(study)
    assert summary["n_trials"] == 20, summary["n_trials"]
    assert summary["best_value"] < 0.005, summary["best_value"]
    assert set(summary["best_params"].keys()) == {
        "num_leaves", "learning_rate", "feature_fraction", "min_child_samples",
    }

    # Persistence — DB file should exist after the run
    assert os.path.exists(DB_PATH)

    # trials_dataframe sanity
    tdf = study.trials_dataframe()
    assert len(tdf) == 20
    assert "value" in tdf.columns

    print(f"n_trials run:         {summary['n_trials']}")
    print(f"n pruned:             {summary['n_pruned']}")
    print(f"best val MAE:         {summary['best_value']:.6f}")
    print(f"best params:          {summary['best_params']}")
    print(f"study persisted to:   {DB_PATH}")
    print("\n✓ All checks passed.")
