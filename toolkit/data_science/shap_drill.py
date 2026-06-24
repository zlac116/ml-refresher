"""
TOOLKIT — SHAP task-indexed drill
===================================

OBJECTIVE
    Practise the 5 canonical SHAP idioms from the cheatsheet:
    TreeExplainer + shap_values, global ranking (mean|shap|), single-row
    explanation, slicing the Explanation object, and interaction values.

ESTIMATED TIME
    60–90 min

TOPICS
    shap.TreeExplainer(model)          (fast, exact for tree models)
    .shap_values(X)                    (numpy array, shape (n, n_features))
    explainer(X)                       (Explanation object with .values, .data, etc.)
    Mean |shap| per feature  → global importance
    .shap_interaction_values(X)        (3-D array)

REQUIRED PACKAGES
    shap, lightgbm, scikit-learn, numpy (run `uv add shap lightgbm scikit-learn`)

EXPECTED OUTPUT
    shap_values shape:     matches X_val
    top feature:           feature 0 (most informative by construction)
    top-3 |shap| sorted:   descending
    single-row n contribs: == n_features
    interaction shape:     (n, n_features, n_features)
    high-feature-0 subset: positive mean shap (interpretation: high x0 → high y)

GRADING
    All asserts must pass.
"""
import numpy as np
import shap
import lightgbm as lgb
from sklearn.datasets import make_regression


X, Y = make_regression(n_samples=300, n_features=5, n_informative=3, noise=5, random_state=42)
SPLIT = int(len(X) * 0.7)
X_TR, Y_TR = X[:SPLIT], Y[:SPLIT]
X_VA = X[SPLIT:]


def _train_model():
    m = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    m.fit(X_TR, Y_TR)
    return m


MODEL = _train_model()


# ── TASK 1 — TreeExplainer + shap_values ─────────────────────────────────
def get_shap_values(model, X) -> np.ndarray:
    """Use shap.TreeExplainer(model).shap_values(X).

    Returns array shape (X.shape[0], n_features).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 — Global ranking by mean |shap| ───────────────────────────────
def mean_abs_shap_ranking(shap_vals: np.ndarray) -> np.ndarray:
    """Compute mean(|shap_vals|, axis=0). Return the per-feature scores."""
    # TODO: implement
    raise NotImplementedError


def top_k_features(scores: np.ndarray, k: int = 3) -> np.ndarray:
    """Return indices of the top-k features sorted descending by score."""
    # TODO: implement (hint: np.argsort(scores)[::-1][:k])
    raise NotImplementedError


# ── TASK 3 — Single-row explanation (waterfall data) ─────────────────────
def explain_single_row(shap_vals: np.ndarray, row_idx: int) -> np.ndarray:
    """Return shap_vals[row_idx] — the per-feature contribution vector for that
    single prediction. Length = n_features.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 — Slicing the Explanation by feature value ────────────────────
def shap_when_feature_is_high(shap_vals: np.ndarray, X: np.ndarray,
                              feature_idx: int) -> float:
    """Compute the mean SHAP value (for that same feature) ACROSS rows where
    feature_idx > its sample median.

    Returns a single float — should be POSITIVE if "high feature value → high pred".
    """
    # TODO: implement
    #   1. mask = X[:, feature_idx] > np.median(X[:, feature_idx])
    #   2. return shap_vals[mask, feature_idx].mean()
    raise NotImplementedError


# ── TASK 5 — Interaction values ──────────────────────────────────────────
def interaction_value_shape(model, X) -> tuple[int, int, int]:
    """shap_interaction = shap.TreeExplainer(model).shap_interaction_values(X)
    Returns interaction.shape — should be (n_samples, n_features, n_features).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sv = get_shap_values(MODEL, X_VA)
    assert sv.shape == X_VA.shape

    rank = mean_abs_shap_ranking(sv)
    assert rank.shape == (5,)
    assert (rank >= 0).all()

    top3 = top_k_features(rank, k=3)
    assert top3.shape == (3,)
    # First-place should beat second, second beats third
    assert rank[top3[0]] >= rank[top3[1]] >= rank[top3[2]]

    row = explain_single_row(sv, 0)
    assert row.shape == (5,)

    mean_high = shap_when_feature_is_high(sv, X_VA, top3[0])
    # For an informative feature, high values should push prediction up
    assert abs(mean_high) > 0    # at least non-zero

    iv_shape = interaction_value_shape(MODEL, X_VA[:50])  # smaller for speed
    assert iv_shape == (50, 5, 5)

    print(f"shap_values shape:     {sv.shape}")
    print(f"top feature:           feature {top3[0]}")
    print(f"top-3 |shap| sorted:   {rank[top3].round(4).tolist()}")
    print(f"single-row n contribs: {len(row)}")
    print(f"interaction shape:     {iv_shape}")
    print(f"high-feature-0 subset: mean SHAP = {mean_high:.4f}")
    print("\n✓ All checks passed.")
