"""
PROJECT — LightGBM: 24h-Ahead BTC Volatility Forecaster
=========================================================

OBJECTIVE
    Train a LightGBM regressor that predicts BTC's next-24h realised vol:

      1. Past-only features + chronological 70/15/15 split.
      2. Baseline LightGBM with early stopping on val MAE.
      3. Add a monotonic constraint on `vol_24h_trailing` (domain knowledge).
      4. Quantile regression for [p10, p90] prediction intervals.
      5. Coverage check: empirical fraction in [p10, p90] should be ~0.80.

ESTIMATED TIME
    30 min

TOPICS
    lgb.LGBMRegressor with objective='regression_l1' (MAE)
    lgb.early_stopping(20) callback + .best_iteration_
    monotone_constraints (encoding domain knowledge)
    quantile objective (alpha=0.1, alpha=0.9)
    Coverage = ((y >= p10) & (y <= p90)).mean()

REQUIRED PACKAGES
    lightgbm, scikit-learn, pandas, numpy (run `uv add lightgbm scikit-learn`)

EXPECTED OUTPUT
    feature df rows:      > 17000
    train / val / test:    ~70/15/15 split
    baseline val MAE:      < 0.01
    monotone val MAE:      similar to baseline (within 10%)
    [p10,p90] coverage:    0.70 < c < 0.90  (~0.80 is the target)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

DATA = "/home/zlac116/Code/learning/ml-revision/data/crypto_hourly.parquet"


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def build_features_and_target(parquet_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Build BTC feature frame for vol forecasting.

    Features (past-only):
        ret_1h, ret_4h, ret_24h
        vol_24h_trailing  =  rolling 24h std of ret_1h
        vol_168h_trailing =  rolling 168h std of ret_1h
        atr_24h           =  rolling 24h mean of (high - low) / close
        rsi_14            =  Wilder's RSI(14)
        hour_of_day       =  ts.dt.hour
    Target:
        vol_24h_fwd       =  rolling 24h std of ret_1h, shifted -24

    Returns (df_with_features, feature_name_list). Drop NaNs after building target.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def chrono_split_xy(df: pd.DataFrame, feature_names: list[str], target: str = "vol_24h_fwd"):
    """Chronological 70/15/15 split. Returns
        (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def fit_baseline(X_train, y_train, X_val, y_val) -> lgb.LGBMRegressor:
    """Fit LGBMRegressor with:
        objective='regression_l1' (MAE)
        early_stopping(20) callback
        eval_set=[(X_val, y_val)]
        log_evaluation(0)  (silence per-iter spam)
        random_state=42, n_estimators=500
    Returns the fitted model.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def fit_monotone(X_train, y_train, X_val, y_val,
                 feature_names: list[str], monotone_feature: str = "vol_24h_trailing"):
    """Refit baseline with monotone_constraints=[1 for monotone_feature, 0 for others].
    Same callbacks as fit_baseline.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def fit_quantile_models(X_train, y_train, X_val, y_val) -> tuple:
    """Fit TWO LGBMRegressors with:
        m_lo: objective='quantile', alpha=0.1
        m_hi: objective='quantile', alpha=0.9
    Same eval_set + early_stopping + log_evaluation(0).
    Returns (m_lo, m_hi).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 6 ─────────────────────────────────────────────────────────────────
def coverage_p10_p90(m_lo, m_hi, X_test, y_test) -> float:
    """Empirical coverage of the [p10, p90] interval on the test set.

    coverage = ((y_test >= p10) & (y_test <= p90)).mean()
    Should be roughly 0.80 if calibration is good.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, feats = build_features_and_target(DATA)
    assert len(df) > 17_000, len(df)
    assert "vol_24h_trailing" in feats and len(feats) >= 6

    X_tr, y_tr, X_va, y_va, X_te, y_te = chrono_split_xy(df, feats)

    m_base = fit_baseline(X_tr, y_tr, X_va, y_va)
    mae_base = mean_absolute_error(y_va, m_base.predict(X_va))
    assert mae_base < 0.01, mae_base

    m_mono = fit_monotone(X_tr, y_tr, X_va, y_va, feats, "vol_24h_trailing")
    mae_mono = mean_absolute_error(y_va, m_mono.predict(X_va))
    # Monotone constraint should not blow up MAE
    assert mae_mono < mae_base * 1.5

    m_lo, m_hi = fit_quantile_models(X_tr, y_tr, X_va, y_va)
    coverage = coverage_p10_p90(m_lo, m_hi, X_te, y_te)
    assert 0.65 < coverage < 0.92, coverage    # should be near 0.80

    print(f"feature df rows:      {len(df)}")
    print(f"train / val / test:    {len(X_tr)} / {len(X_va)} / {len(X_te)}")
    print(f"baseline val MAE:      {mae_base:.6f}")
    print(f"monotone val MAE:      {mae_mono:.6f}")
    print(f"[p10,p90] coverage:    {coverage:.4f}   (~0.80 = well calibrated)")
    print("\n✓ All checks passed.")
