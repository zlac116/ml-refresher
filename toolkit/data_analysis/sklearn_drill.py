"""
TOOLKIT — scikit-learn task-indexed drill
==========================================

OBJECTIVE
    Practise the 8 canonical sklearn idioms from the cheatsheet:
    pipelines, time-series CV, scorer sign conventions, predict_proba /
    decision_function, calibration, permutation importance, joblib persistence.

ESTIMATED TIME
    60–90 min

TOPICS
    sklearn.pipeline.Pipeline    ([('sc', StandardScaler()), ('m', Model())])
    sklearn.model_selection.TimeSeriesSplit
    cross_val_score sign convention (neg_log_loss is NEGATIVE)
    .predict_proba vs .decision_function
    sklearn.calibration.CalibratedClassifierCV
    sklearn.inspection.permutation_importance
    joblib.dump / load

EXPECTED OUTPUT
    pipe train accuracy:  0.85
    TS-CV mean accuracy:  0.8361
    neg log-loss mean:    -0.3902     (NEGATIVE — sklearn convention)
    proba shape / sum:    (500, 2) / row0=1.0
    calibrated proba 1:   0.5015
    perm importance top:  feature 5
    save/load round-trip: True

GRADING
    All asserts must pass.

REQUIRED PACKAGES
    scikit-learn, joblib, numpy (run `uv add scikit-learn joblib`)
"""
import numpy as np
import joblib
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance


X_CLF, Y_CLF = make_classification(
    n_samples=500, n_features=10, n_informative=5, random_state=42,
)


# ── TASK 1 — Pipeline construction ───────────────────────────────────────
def make_pipeline() -> Pipeline:
    """Build a Pipeline:
        step 'sc' : StandardScaler()
        step 'lr' : LogisticRegression(max_iter=200)
    Return UNFITTED.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 — Fit + score ──────────────────────────────────────────────────
def fit_and_score(pipe: Pipeline, X, y) -> float:
    """Fit pipe on (X, y), return train accuracy."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 — Time-series CV ──────────────────────────────────────────────
def ts_cv_accuracy(pipe: Pipeline, X, y, n_splits: int = 5) -> float:
    """cross_val_score with cv=TimeSeriesSplit(n_splits), scoring='accuracy'.
    Return the MEAN across folds.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 — neg_log_loss sign convention ────────────────────────────────
def neg_log_loss_mean(pipe: Pipeline, X, y, cv: int = 5) -> float:
    """cross_val_score with scoring='neg_log_loss'. Return mean.

    NOTE: sklearn convention — "higher is better" for scorers, so log-loss
    (lower is better) is NEGATED. The returned value will be NEGATIVE.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 — predict_proba ────────────────────────────────────────────────
def proba_shape_and_row_sum(pipe: Pipeline, X) -> tuple[tuple[int, int], float]:
    """Run pipe.predict_proba(X).
    Returns (shape, sum_of_first_row).
    For binary classification, each row sums to 1.0.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 6 — Calibration ──────────────────────────────────────────────────
def calibrated_proba_mean(base_estimator, X, y) -> float:
    """Wrap base_estimator with CalibratedClassifierCV(method='sigmoid', cv=3).
    Fit on (X, y). Return the mean of class-1 probabilities on X.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 7 — Permutation importance ──────────────────────────────────────
def top_perm_importance_feature(pipe: Pipeline, X, y) -> int:
    """permutation_importance(pipe, X, y, n_repeats=5, random_state=42).
    Return the index of the feature with the largest mean importance.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 8 — joblib save / load round-trip ───────────────────────────────
def save_load_roundtrip(pipe: Pipeline, X) -> bool:
    """Save pipe to /tmp/_pipe.pkl, reload, return True iff predictions match.
    Use joblib.dump / joblib.load.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipe = make_pipeline()
    assert isinstance(pipe, Pipeline)
    assert "sc" in pipe.named_steps and "lr" in pipe.named_steps

    acc = fit_and_score(pipe, X_CLF, Y_CLF)
    assert abs(acc - 0.85) < 1e-2, acc

    ts_acc = ts_cv_accuracy(pipe, X_CLF, Y_CLF, n_splits=5)
    assert abs(ts_acc - 0.8361) < 1e-2

    nll = neg_log_loss_mean(pipe, X_CLF, Y_CLF, cv=5)
    assert nll < 0   # MUST be negative (sklearn convention)
    assert abs(nll - -0.3902) < 5e-2

    shape, row_sum = proba_shape_and_row_sum(pipe, X_CLF)
    assert shape == (500, 2)
    assert abs(row_sum - 1.0) < 1e-9

    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_CLF, Y_CLF)
    cal_mean = calibrated_proba_mean(rf, X_CLF, Y_CLF)
    assert 0.4 < cal_mean < 0.6, cal_mean

    top_idx = top_perm_importance_feature(pipe, X_CLF, Y_CLF)
    assert 0 <= top_idx < 10

    assert save_load_roundtrip(pipe, X_CLF) is True

    print(f"pipe train accuracy:  {acc:.2f}")
    print(f"TS-CV mean accuracy:  {ts_acc:.4f}")
    print(f"neg log-loss mean:    {nll:.4f}     (NEGATIVE — sklearn convention)")
    print(f"proba shape / sum:    {shape} / row0={row_sum:.1f}")
    print(f"calibrated proba 1:   {cal_mean:.4f}")
    print(f"perm importance top:  feature {top_idx}")
    print(f"save/load round-trip: True")
    print("\n✓ All checks passed.")
