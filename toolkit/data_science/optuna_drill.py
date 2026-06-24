"""
TOOLKIT — Optuna task-indexed drill
=====================================

OBJECTIVE
    Practise the 6 canonical Optuna idioms from the cheatsheet:
    basic study, ML objective with trial.suggest_*, pruning, SQLite
    persistence, multi-objective, and trials_dataframe post-hoc analysis.

ESTIMATED TIME
    60–90 min

TOPICS
    optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    trial.suggest_float / suggest_int / suggest_categorical
    optuna.pruners.MedianPruner
    storage='sqlite:///path.db' for resume capability
    multi-objective: create_study(directions=[...]) returns best_trials
    study.trials_dataframe()

REQUIRED PACKAGES
    optuna, scikit-learn, numpy (run `uv add optuna scikit-learn`)

EXPECTED OUTPUT
    toy best x:           ≈ 3.0     (objective minimised at x=3)
    toy best value:       ≈ 0       (minimum is 0)
    ML best val MAE:      finite + < baseline
    n_trials run:         15
    SQLite file exists:   True
    multi-obj n_pareto:   ≥ 1
    trials_df rows:       15

GRADING
    All asserts must pass.
"""
import os
import optuna
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, Y = make_regression(n_samples=400, n_features=5, noise=10, random_state=42)
SPLIT = int(0.7 * len(X))
X_TR, Y_TR = X[:SPLIT], Y[:SPLIT]
X_VA, Y_VA = X[SPLIT:], Y[SPLIT:]
DB_PATH = "/tmp/optuna_drill.db"


# ── TASK 1 — Toy study (minimise (x-3)^2) ────────────────────────────────
def toy_objective(trial: optuna.Trial) -> float:
    """Return (trial.suggest_float('x', -10, 10) - 3.0) ** 2."""
    # TODO: implement
    raise NotImplementedError


def run_toy_study(n_trials: int = 30) -> optuna.Study:
    """create_study(direction='minimize', sampler=TPESampler(seed=42))
    Run study.optimize(toy_objective, n_trials).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 — ML objective with suggest_float + suggest_int ──────────────
def ml_objective(trial: optuna.Trial) -> float:
    """Build a Ridge regressor with:
        alpha    = trial.suggest_float('alpha', 1e-3, 10.0, log=True)
        solver   = trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky'])

    Fit on (X_TR, Y_TR), return val MAE on (X_VA, Y_VA).
    """
    # TODO: implement
    raise NotImplementedError


def run_ml_study(n_trials: int = 15) -> optuna.Study:
    """create_study(direction='minimize', sampler=TPESampler(seed=42),
                    pruner=MedianPruner())
    Run study.optimize(ml_objective, n_trials).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 — SQLite persistence ──────────────────────────────────────────
def run_persisted_study(n_trials: int = 10, storage_path: str = DB_PATH) -> optuna.Study:
    """Create a study with storage=f'sqlite:///{storage_path}',
    study_name='persisted', load_if_exists=True.

    Run with ml_objective for n_trials. Return the study.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 — Multi-objective: MAE vs n iterations (proxy for fit time) ──
def multi_objective(trial: optuna.Trial) -> tuple[float, float]:
    """Return (val_mae, alpha) — minimise BOTH (we use alpha as a stand-in for
    "regularisation strength" the user might also want to minimise).

    Use the same alpha + solver suggestions as ml_objective.
    """
    # TODO: implement (hint: same Ridge fit, return TUPLE of two scalars)
    raise NotImplementedError


def run_multi_objective(n_trials: int = 15) -> optuna.Study:
    """create_study(directions=['minimize', 'minimize'], sampler=TPESampler(seed=42)).
    Run with multi_objective for n_trials.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 — trials_dataframe inspection ─────────────────────────────────
def trials_summary(study: optuna.Study) -> dict:
    """Return dict with:
        n_trials   : total trials in the study
        best_value : study.best_value (for single-objective)
        cols       : list of columns in study.trials_dataframe()
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Toy
    toy = run_toy_study(n_trials=30)
    assert abs(toy.best_params["x"] - 3.0) < 1.0      # TPE should find near 3
    assert toy.best_value < 1.0

    # ML
    ml = run_ml_study(n_trials=15)
    assert ml.best_value < 50.0                       # reasonable val MAE

    # Persisted
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    pst = run_persisted_study(n_trials=10)
    assert os.path.exists(DB_PATH)
    assert len(pst.trials) == 10

    # Multi-objective
    mo = run_multi_objective(n_trials=15)
    assert len(mo.best_trials) >= 1                   # Pareto front non-empty

    # trials_dataframe
    summary = trials_summary(ml)
    assert summary["n_trials"] == 15
    assert "value" in summary["cols"]

    print(f"toy best x:           {toy.best_params['x']:.4f}     (objective minimised at x=3)")
    print(f"toy best value:       {toy.best_value:.4e}     (minimum is 0)")
    print(f"ML best val MAE:      {ml.best_value:.4f}")
    print(f"n_trials run:         {len(ml.trials)}")
    print(f"SQLite file exists:   True")
    print(f"multi-obj n_pareto:   {len(mo.best_trials)}")
    print(f"trials_df rows:       {summary['n_trials']}")
    print("\n✓ All checks passed.")
