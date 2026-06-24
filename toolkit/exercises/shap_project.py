"""
PROJECT — SHAP: Explain a Vol Forecaster
==========================================

OBJECTIVE
    Apply SHAP to a trained LightGBM vol forecaster:

      1. Train (or quickly retrain) the model on a feature subset.
      2. Compute SHAP values via TreeExplainer.
      3. Rank features by mean(|shap|) — should agree with gain importance.
      4. Find the most extreme prediction; explain it via its shap_values row.
      5. Top-K ablation: drop the k features with smallest mean|shap|, retrain,
         confirm MAE doesn't get much worse — sanity check on "importance" matching
         "usefulness".

ESTIMATED TIME
    25 min

TOPICS
    shap.TreeExplainer(model) for fast, exact SHAP on tree models
    .shap_values(X) — returns array of shape (n_samples, n_features)
    Mean |shap| per feature = global importance
    Single-row explanation = local importance (predict_one)

REQUIRED PACKAGES
    shap, lightgbm, scikit-learn (run `uv add shap lightgbm scikit-learn`)

EXPECTED OUTPUT
    feature df rows:        > 17000
    train/val sizes:        ~70/30
    shap shape:             matches X_val
    top-3 by mean|shap|:    printed (vol_24h_trailing likely on top)
    top-3 ablation MAE:     within 30% of full-model MAE
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import mean_absolute_error

DATA = "/home/zlac116/Code/learning/ml-revision/data/crypto_hourly.parquet"


def _build_xy():
    """Self-contained feature build for the SHAP drill."""
    df = pd.read_parquet(DATA)
    df = df[df.symbol == "BTC"].sort_values("ts").reset_index(drop=True)
    df["ret_1h"]            = df["close"].pct_change()
    df["ret_4h"]            = df["close"].pct_change(4)
    df["ret_24h"]           = df["close"].pct_change(24)
    df["vol_24h_trailing"]  = df["ret_1h"].rolling(24).std()
    df["sma_ratio"]         = df["close"] / df["close"].rolling(24).mean()
    df["target"]            = df["ret_1h"].rolling(24).std().shift(-24)
    df = df.dropna()
    feats = ["ret_1h", "ret_4h", "ret_24h", "vol_24h_trailing", "sma_ratio"]
    n = len(df); split = int(n * 0.7)
    return (df[feats].iloc[:split].values, df["target"].iloc[:split].values,
            df[feats].iloc[split:].values, df["target"].iloc[split:].values,
            feats)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def train_model(X_train, y_train) -> lgb.LGBMRegressor:
    """Train LGBMRegressor with objective='regression_l1', random_state=42,
    n_estimators=200, log_evaluation(0).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def shap_values(model, X) -> np.ndarray:
    """Compute SHAP values via TreeExplainer.

    Returns array of shape (X.shape[0], n_features).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def global_importance_ranking(shap_vals: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Mean absolute SHAP per feature, sorted descending.

    Returns pd.Series indexed by feature name.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def explain_most_extreme(model, shap_vals: np.ndarray, X) -> tuple[int, np.ndarray]:
    """Find the index of the row with the LARGEST predicted vol, return
        (row_index, shap_row)
    where shap_row is shap_vals[row_index] (per-feature contribution).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def top_k_ablation(X_train, y_train, X_val, y_val, feature_names: list[str],
                   keep_features: list[str]) -> float:
    """Retrain with ONLY the columns named in `keep_features` (in their input order).
    Score val MAE. Returns the val MAE.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_tr, y_tr, X_va, y_va, feats = _build_xy()
    assert len(X_tr) > 10_000

    m = train_model(X_tr, y_tr)
    full_mae = mean_absolute_error(y_va, m.predict(X_va))
    assert full_mae < 0.01

    sv = shap_values(m, X_va)
    assert sv.shape == X_va.shape, sv.shape

    rank = global_importance_ranking(sv, feats)
    assert rank.index[0] == "vol_24h_trailing", rank.index[0]   # the dominant feature
    assert len(rank) == len(feats)
    # Mean|shap| should be non-negative and decreasing
    assert (rank >= 0).all()
    assert (rank.values[:-1] >= rank.values[1:]).all()

    idx, row_shap = explain_most_extreme(m, sv, X_va)
    assert 0 <= idx < len(X_va)
    assert row_shap.shape == (len(feats),)

    # Ablation: drop the bottom 2, keep top 3
    top3 = rank.index[:3].tolist()
    ablated_mae = top_k_ablation(X_tr, y_tr, X_va, y_va, feats, keep_features=top3)
    assert ablated_mae < full_mae * 1.5     # shouldn't blow up — top 3 carry the signal

    print(f"feature df rows:        {X_tr.shape[0] + X_va.shape[0]}")
    print(f"train/val sizes:        {X_tr.shape[0]} / {X_va.shape[0]}")
    print(f"shap shape:             {sv.shape}")
    print(f"top-3 by mean|shap|:    {rank.index[:3].tolist()}")
    print(f"top-3 ablation MAE:     {ablated_mae:.6f}   (full = {full_mae:.6f})")
    print("\n✓ All checks passed.")
