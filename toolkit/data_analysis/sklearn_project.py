"""
PROJECT — scikit-learn: 4h-Ahead BTC Direction (with leakage discipline)
=========================================================================

OBJECTIVE
    Build a binary classifier for "BTC close goes UP over the next 4 hours"
    with proper time-series hygiene:

      1. Past-only features + chronologically split train/val/test (no leakage)
      2. Pipeline-wrapped candidates (LogReg, RF, GBM)
      3. Score on validation set + pick a winner
      4. Permutation importance for interpretability

ESTIMATED TIME
    30 min

TOPICS
    sklearn.pipeline.Pipeline (scaler + estimator)
    Chronological 70/15/15 split — NEVER random for time series
    cross_val_score with cv=TimeSeriesSplit
    permutation_importance

REQUIRED PACKAGES
    scikit-learn, pandas, numpy, scipy (run `uv add scikit-learn` if missing)

EXPECTED OUTPUT
    feature df shape:    (17496, 14)
    target balance:      0.5068
    train / val / test:  12247 / 2624 / 2625
    chosen model:        LogReg | RF | GBM (depends on your impl)
    val accuracy:        > 0.50 (better than chance)
    top 3 features:      printed (order may vary)

GRADING
    Asserts check shapes, no leakage, sensible accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

DATA = "/home/zlac116/Code/learning/ml-revision/data/crypto_hourly.parquet"


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def build_features(parquet_path: str) -> pd.DataFrame:
    """Build a BTC-only feature frame with PAST-ONLY features:

        ret_1h, ret_4h, ret_24h     = pct_change over those horizons
        vol_24h                     = rolling 24h std of ret_1h
        sma_ratio                   = close / close.rolling(24).mean()
        mom_4h                      = pct_change(4)
        target                      = (close.pct_change(4).shift(-4) > 0).astype(int)

    Drop rows with NaN. Return a DataFrame with these columns + 'close' + 'ts'.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def chrono_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Chronological split (NEVER random for time series).

    Returns (df_train, df_val, df_test) sliced in time order.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def make_pipelines() -> dict[str, Pipeline]:
    """Return three named pipelines, each pre-scaled where applicable:
        'logreg' : StandardScaler + LogisticRegression(max_iter=500)
        'rf'     : RandomForestClassifier(n_estimators=200, random_state=42)
        'gbm'    : GradientBoostingClassifier(random_state=42)

    Tree models don't need scaling, but the Pipeline shape lets you swap easily.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def evaluate_on_val(pipelines: dict, X_train, y_train, X_val, y_val) -> dict[str, float]:
    """Fit each pipeline on (X_train, y_train), score accuracy on (X_val, y_val).

    Return dict of {name: val_accuracy}.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def top_k_permutation_importance(model, X_val, y_val, feature_names, k: int = 3) -> list[str]:
    """Run permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
    and return the top-k feature names by mean importance.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_features(DATA)
    assert df.shape[0] >= 17_000, df.shape
    assert "target" in df.columns
    assert 0.45 < df["target"].mean() < 0.55  # roughly balanced
    feature_cols = [c for c in df.columns if c not in ("ts", "target", "close")]
    assert len(feature_cols) >= 5

    tr, va, te = chrono_split(df)
    assert tr["ts"].max() < va["ts"].min()      # NO time leakage
    assert va["ts"].max() < te["ts"].min()
    assert abs(len(tr)/len(df) - 0.70) < 0.02

    pipes = make_pipelines()
    assert set(pipes.keys()) == {"logreg", "rf", "gbm"}

    X_tr, y_tr = tr[feature_cols].values, tr["target"].values
    X_va, y_va = va[feature_cols].values, va["target"].values

    scores = evaluate_on_val(pipes, X_tr, y_tr, X_va, y_va)
    assert all(0.45 < v < 0.65 for v in scores.values())   # sane range

    best_name = max(scores, key=scores.get)
    best_model = pipes[best_name]
    top_feats = top_k_permutation_importance(best_model, X_va, y_va, feature_cols, k=3)
    assert len(top_feats) == 3
    assert all(f in feature_cols for f in top_feats)

    print(f"feature df shape:    {df.shape}")
    print(f"target balance:      {df['target'].mean():.4f}")
    print(f"train / val / test:  {len(tr)} / {len(va)} / {len(te)}")
    print(f"chosen model:        {best_name}")
    print(f"val accuracy:        {scores[best_name]:.4f}")
    print(f"top 3 features:      {top_feats}")
    print("\n✓ All checks passed.")
