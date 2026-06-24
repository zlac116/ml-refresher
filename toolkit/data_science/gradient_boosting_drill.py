"""
TOOLKIT — gradient boosting (LightGBM + XGBoost) drill
========================================================

OBJECTIVE
    Practise the 6 canonical gradient-boosting idioms from the cheatsheet:
    basic fit/predict, early stopping, feature importance (gain vs split),
    quantile regression, monotonic constraints, and joblib persistence.

ESTIMATED TIME
    60–90 min

TOPICS
    lgb.LGBMRegressor(objective='regression_l1', n_estimators=500,
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    .best_iteration_
    .booster_.feature_importance(importance_type='gain' / 'split')
    objective='quantile', alpha=0.1 / 0.9
    monotone_constraints=[1, 0, 0, ...]
    joblib.dump / load

REQUIRED PACKAGES
    lightgbm, scikit-learn, numpy, joblib (run `uv add lightgbm scikit-learn joblib`)

EXPECTED OUTPUT
    baseline val MAE:        < 30
    best_iteration:          < n_estimators (early stopped)
    top feature by gain:     feature 0 (synthetic generator made it most informative)
    p90 - p10 width mean:    > 0  (quantile interval positive)
    monotone feature works:  prediction non-decreasing in that feature
    save/load round-trip:    predictions match

GRADING
    All asserts must pass.
"""
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error


X_REG, Y_REG = make_regression(
    n_samples=500, n_features=5, n_informative=3, noise=10, random_state=42,
)
# Train/val split (chronological style — first 70% train, rest val)
_split = int(len(X_REG) * 0.7)
X_TR, Y_TR = X_REG[:_split], Y_REG[:_split]
X_VA, Y_VA = X_REG[_split:], Y_REG[_split:]


# ── TASK 1 — Baseline LightGBM with early stopping ───────────────────────
def fit_baseline(X_train, y_train, X_val, y_val) -> lgb.LGBMRegressor:
    """Fit LGBMRegressor with:
        objective='regression_l1'  (MAE)
        n_estimators=500, random_state=42
        eval_set=[(X_val, y_val)]
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 — Inspect best_iteration_ ─────────────────────────────────────
def early_stop_summary(model) -> int:
    """Return model.best_iteration_ (integer, < n_estimators if early-stopped)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 — Feature importance (gain vs split) ─────────────────────────
def feature_importance_top(model) -> tuple[int, int]:
    """Return (top_idx_gain, top_idx_split) where each is the index of the
    most-important feature under that importance type.

    Use model.booster_.feature_importance(importance_type='gain' | 'split').
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 — Quantile regression for [p10, p90] ─────────────────────────
def quantile_models(X_train, y_train, X_val, y_val) -> tuple:
    """Fit two LGBMRegressors:
        m_lo: objective='quantile', alpha=0.1
        m_hi: objective='quantile', alpha=0.9
    Same n_estimators=200, eval_set, early_stopping(20), log_evaluation(0).
    Returns (m_lo, m_hi).
    """
    # TODO: implement
    raise NotImplementedError


def interval_width(m_lo, m_hi, X) -> float:
    """Mean width of [p10, p90] interval on X = mean(p90 - p10)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 — Monotonic constraint ────────────────────────────────────────
def fit_monotone(X_train, y_train, monotone_idx: int = 0) -> lgb.LGBMRegressor:
    """Fit LGBMRegressor with monotone_constraints set to +1 at monotone_idx,
    0 elsewhere. Use objective='regression_l1', n_estimators=100,
    random_state=42, no early stopping required (no eval set).
    """
    # TODO: implement
    raise NotImplementedError


def verify_monotonicity(model, n_features: int, monotone_idx: int = 0) -> bool:
    """Verify the prediction is non-decreasing in feature monotone_idx.
    Build a test grid where only that feature varies (others = 0), predict,
    and check pred is sorted in non-decreasing order.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 6 — Save / load via joblib ──────────────────────────────────────
def save_load_lgbm(model, X, path: str = "/tmp/_lgbm.pkl") -> bool:
    """Save model with joblib.dump, reload, check predictions match within 1e-9."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = fit_baseline(X_TR, Y_TR, X_VA, Y_VA)
    val_mae = mean_absolute_error(Y_VA, m.predict(X_VA))
    assert val_mae < 30, val_mae

    best_it = early_stop_summary(m)
    assert best_it > 0 and best_it < 500

    top_gain, top_split = feature_importance_top(m)
    assert 0 <= top_gain < 5
    assert 0 <= top_split < 5

    m_lo, m_hi = quantile_models(X_TR, Y_TR, X_VA, Y_VA)
    width = interval_width(m_lo, m_hi, X_VA)
    assert width > 0   # p90 > p10 on average

    m_mono = fit_monotone(X_TR, Y_TR, monotone_idx=0)
    assert verify_monotonicity(m_mono, n_features=5, monotone_idx=0) is True

    assert save_load_lgbm(m, X_VA) is True

    print(f"baseline val MAE:        {val_mae:.4f}")
    print(f"best_iteration:          {best_it}")
    print(f"top feature by gain:     feature {top_gain}")
    print(f"top feature by split:    feature {top_split}")
    print(f"p90 - p10 width mean:    {width:.4f}")
    print(f"monotone feature works:  True")
    print(f"save/load round-trip:    True")
    print("\n✓ All checks passed.")
