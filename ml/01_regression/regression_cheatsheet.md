# Regression — End-to-End Cheat Sheet

Operational checklist for any regression problem. Read top to bottom before starting; jump to a section when you need the *why*.

---

## Overall process

```
   ┌────────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌──────────────┐
   │ FRAME      │ ─►│ EDA      │ ─►│ FEATURE │ ─►│ MODEL    │ ─►│ EVALUATE     │
   │ + SPLIT    │   │ + TARGET │   │ PIPELINE│   │ + TUNE   │   │ + RESIDUAL   │
   │            │   │ AUDIT    │   │         │   │          │   │ DIAGNOSTICS  │
   └────────────┘   └──────────┘   └─────────┘   └──────────┘   └──────────────┘
                                                                       │
                                                                       ▼
                                                    ┌──────────────────────────┐
                                                    │ DEPLOY + DRIFT MONITORING│
                                                    └──────────────────────────┘
```

---

## 1. Frame the problem

| Question | Why it matters |
|---|---|
| **What does the target distribution look like?** | Skewed → consider log/Box-Cox transform |
| **Are outliers meaningful or errors?** | Robust loss (Huber) vs MAE vs cleaning |
| **Is the error cost asymmetric?** | Quantile regression handles asymmetric loss |
| **Is the problem really regression?** | If you only care about a threshold, classification might be simpler |
| **What's the natural unit / scale of error?** | Drives metric choice — % error vs absolute |

---

## 2. EDA + target audit

**Look at the target FIRST:**

```python
y.describe()
y.skew(), y.kurt()
np.log1p(y).hist()   # if skewed
```

**Target transformations:**

| Target shape | Transform | Why |
|---|---|---|
| ~Normal | None | Linear models are happy |
| Right-skewed, positive (income, price) | `log1p` (i.e. log(1+y)) | Compresses heavy tail; makes errors multiplicative |
| Counts (purchases, visits) | `sqrt` or Poisson regression | Variance grows with mean |
| Heavy outliers | Robust scaler on y, or quantile transform | Avoid bias toward outliers |
| Strictly bounded [0,1] | Logit log(y/(1-y)) | So the model can output anything |

**Important:** if you transform the target, **inverse-transform predictions** for the final metric. e.g. `np.expm1(y_pred_log)`.

**Why log1p instead of log:** handles y = 0 cleanly (log(0) is -∞).

**Other EDA musts:**
- Pairwise correlations of features with target (Pearson for linear, Spearman for monotone)
- Heteroscedasticity check (variance changing with predicted value) — residuals plot
- Feature-feature multicollinearity — VIF > 10 means trouble for linear models
- Outliers in features — RobustScaler or winsorize

---

## 3. Train/val/test split — BEFORE any preprocessing

```python
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
```

**No `stratify` for regression** (target is continuous). If target distribution matters across splits, use stratified sampling on **binned target**:

```python
y_bins = pd.qcut(y, q=10, labels=False)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y_bins)
```

**If data is temporal → never random split.** Use chronological split. (See time-series cheat sheet.)

**Use a `Pipeline`** — it enforces the no-leakage discipline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("reg", Ridge(alpha=1.0)),
])
```

---

## 4. Feature pipeline

Mostly the same patterns as classification. Regression-specific notes:

| Concern | Treatment |
|---|---|
| Outlier features | RobustScaler (median/IQR), or winsorize |
| Skewed continuous features | log1p, Box-Cox, Yeo-Johnson (handles negatives) |
| Polynomial features | `PolynomialFeatures(degree=2)` — watch dimension explosion |
| Interaction terms | Trees find them automatically; linear models need explicit features |
| Multicollinearity | Drop highly correlated, or use Ridge/Lasso |

**Multicollinearity** matters for **linear models** (unstable coefficients) but not for tree-based models. Ridge regression is the friend when correlations are unavoidable.

---

## 5. Model selection

| Family | When to use | Strengths | Weaknesses |
|---|---|---|---|
| **OLS Linear** | Baseline, small n, interpretability | Closed-form, interpretable | Sensitive to outliers, assumes linearity |
| **Ridge (L2)** | Multicollinearity present | Stabilises coefficients | Doesn't do feature selection |
| **Lasso (L1)** | Want sparse model / feature selection | Zeros out unimportant features | Can be unstable |
| **ElasticNet** | Mix of correlated + sparse | Best of L1 and L2 | Two hyperparams to tune |
| **Polynomial / Splines** | Known non-linearity | Smooth, interpretable | Manual choice of knots/degree |
| **GAM** | Need interpretable non-linearity | Smooth per-feature splines | Slower to fit |
| **SVR (RBF)** | Small-to-medium tabular | Captures non-linearity | Slow, hard to tune |
| **Random Forest** | Tabular medium-n | Robust, no scaling, handles missing | Slow inference |
| **Gradient Boosting** (XGBoost/LightGBM/CatBoost) | Tabular default | Usually best on tabular | Sensitive to hyperparams |
| **Neural net** | High-dim, n > 100k | Flexible, deep features | Hungry for data, less interpretable |
| **Quantile regression** | Need intervals or asymmetric loss | P10/P50/P90 directly | One model per quantile |
| **Bayesian (PyMC, GPR)** | Need uncertainty + small n | Full posterior, principled CI | Slow, complex |

**Rule of thumb:** always train **Ridge** as a baseline. If GBM beats Ridge by < 5%, the Ridge may be the more honest model.

---

## 6. Hyperparameter tuning

```python
from sklearn.model_selection import GridSearchCV, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
gs.fit(X_tr, y_tr)
```

**Use Optuna (TPE sampler)** when search space > 3 dimensions — always for XGBoost.

**For GBM, the priority order:**
1. `n_estimators` + `learning_rate` (use **early stopping** — set `n_estimators = 2000`, stop when val loss flattens)
2. `max_depth` (3–8 typical) / `num_leaves` (LightGBM)
3. `min_child_weight` / `min_samples_leaf` — regularisation
4. `subsample` + `colsample_bytree` — row + column sampling
5. `reg_alpha` + `reg_lambda` — L1/L2 weights

---

## 7. Evaluation metrics

| Metric | Formula | When to use |
|---|---|---|
| **MAE** | mean( \|y − ŷ\| ) | Robust to outliers, same units as target, interpretable |
| **MSE / RMSE** | mean( (y − ŷ)² ) / √ | Penalises large errors more; standard default; differentiable |
| **MAPE** | mean( \|y − ŷ\| / \|y\| ) × 100 | % error, but breaks near y = 0 |
| **sMAPE** | symmetric MAPE | Bounded [0, 200%], handles y = 0 better |
| **R²** | 1 − SS_res / SS_tot | Variance explained; misleading outside training range |
| **Adjusted R²** | R² penalised for # features | Penalises adding useless features |
| **Pinball loss** | Quantile-specific asymmetric | Quantile regression |
| **Pearson r** | Correlation | Use when rank matters more than scale |

**Why RMSE is the default:** differentiable (so it's used as training loss), penalises large errors quadratically (matches typical business cost), same units as target.

**Use MAE if outliers** — RMSE will be dominated by them.

**Avoid R² as the primary reported metric outside training range** — it can be negative on test data and "variance explained" is less concrete than "average error of £X".

---

## 8. Residual diagnostics

After fitting, plot residuals (y − ŷ) against:

1. **Predicted values** — should look like random noise around 0. Funnel shape → heteroscedasticity → weighted least squares or log-transform target.
2. **Each feature** — non-random pattern → missing non-linearity → add polynomial / spline / use tree-based model.
3. **Time index (if temporal)** — pattern → temporal feature missing.
4. **Histogram of residuals** — should be ~Normal. Fat tails → robust loss (Huber).
5. **Q-Q plot** — straight line = Normal residuals. Curve → consider transform.

**Why residuals matter:** the assumptions of linear regression are about residuals, not the data. If residuals look like noise → model is well-specified. If they have structure → model is missing something.

---

## 9. Confidence / prediction intervals

A point prediction without uncertainty is half a result. Options:

| Method | Pros | Cons |
|---|---|---|
| **OLS analytical PI** | Closed form, fast | Assumes normal residuals |
| **Bootstrap** | Distribution-free, flexible | Slow |
| **Quantile regression** | Direct P10/P50/P90 | One model per quantile |
| **Conformal prediction** | Distribution-free, finite-sample valid | Slightly conservative |
| **Bayesian** | Full posterior | Slow, harder to set up |

For most production cases, **quantile gradient boosting** is the practical default — train P10, P50, P90 with `objective="quantile"`.

---

## 10. Final test evaluation (once!)

```python
y_pred = pipe.predict(X_te)
print(f"RMSE : {root_mean_squared_error(y_te, y_pred):.4f}")
print(f"MAE  : {mean_absolute_error(y_te, y_pred):.4f}")
print(f"R²   : {r2_score(y_te, y_pred):.4f}")
```

**Inverse-transform if y was transformed**, before reporting metrics in original units:

```python
y_pred_orig = np.expm1(y_pred_log)
```

---

## 11. Deployment + monitoring

| Concern | Tactic |
|---|---|
| Train-serve skew | Persist the **entire pipeline** (preprocessing + model) |
| Schema drift | Validate input schema at serve time |
| Concept drift | Monitor live RMSE / MAE; alert when degraded |
| Feature drift | PSI / KS on each feature vs training distribution |
| Prediction drift | Compare distribution of predictions vs training period |

**Re-train cadence:** monthly is fine for most business problems; weekly or daily for volatile domains (finance, e-commerce).

---

## Common pitfalls

- **Random split on temporal data** — uses the future to predict the past. Always chronological for time-dependent data.
- **Forgetting to inverse-transform predictions** when target was transformed.
- **Optimising R² without checking residuals** — high R² with totally wrong shape is possible.
- **Reporting RMSE on log-target as if it's RMSE on original target** — different units.
- **Polynomial features without regularisation** — overfits fast.
- **Treating multicollinearity as irrelevant for linear models** — kills interpretability of coefficients.
- **Trusting feature importance from one tree-based model** — use permutation or SHAP.
- **No baseline** → no way to tell if your model is any good.

---

## Quick-reference

| Decision | Default | Switch when |
|---|---|---|
| Train/val/test split | 70 / 10 / 20, random | Temporal → chronological |
| Target transform | None | Skewed → log1p; bounded → logit |
| Scaling | StandardScaler | Outliers → RobustScaler; trees → none |
| Baseline | Ridge | Always |
| Default tabular model | XGBoost / LightGBM | n < 1k → Ridge; interpretability → linear |
| Tuning | Optuna with TPE | Small grid → GridSearchCV |
| CV | KFold (k = 5) | Temporal → TimeSeriesSplit |
| Primary metric | RMSE | Outliers → MAE; % error needed → sMAPE |
| Confidence intervals | Quantile GBM (P10/P50/P90) | Strong distributional assumption → OLS PI |
| Residual diagnostics | Plot vs predicted + Q-Q | Always |

---

## What to remember

1. **Look at the target before anything else.** Skew, outliers, range — these determine half your modelling choices (transform, loss, metric).
2. **Residuals must look like noise.** Pattern in residuals = model is missing something. Diagnostics aren't optional.
3. **A prediction without uncertainty is half a result.** Use quantile regression or conformal intervals — point predictions are rarely enough for a real decision.
