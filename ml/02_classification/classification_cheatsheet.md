# Classification — End-to-End Cheat Sheet

Operational checklist for any classification problem. Read top to bottom before starting; jump to a section when you need the *why*.

---

## Overall process

```
   ┌────────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │ FRAME      │ ─►│ EDA      │ ─►│ FEATURE │ ─►│ MODEL    │ ─►│ EVALUATE │
   │ + SPLIT    │   │ + LABEL  │   │ PIPELINE│   │ + TUNE   │   │ + CALIB. │
   │            │   │ AUDIT    │   │         │   │          │   │          │
   └────────────┘   └──────────┘   └─────────┘   └──────────┘   └──────────┘
                                                                       │
                                                                       ▼
                                                    ┌──────────────────────────┐
                                                    │ THRESHOLD + DEPLOY +     │
                                                    │ DRIFT MONITORING         │
                                                    └──────────────────────────┘
```

---

## 1. Frame the problem

Before any code, answer:

| Question | Why it matters |
|---|---|
| **Binary or multi-class?** | Multi-class changes metric choice + loss + class-imbalance handling |
| **What's the business cost of FP vs FN?** | Determines the threshold (NOT the model). E.g. fraud detection — FN is catastrophic, FP is annoying |
| **Hard or soft prediction needed?** | Calibrated probabilities require extra step (Platt / isotonic) |
| **Online or batch scoring?** | Latency budget affects model family (XGBoost vs neural net vs logistic) |
| **Label quality?** | If labels are noisy, label review matters more than fancy models |

---

## 2. EDA + label audit

**Check label balance:**
```python
y.value_counts(normalize=True)
```

| Imbalance level | Strategy |
|---|---|
| 50/50 — 70/30 | None needed |
| 70/30 — 90/10 | `class_weight="balanced"` in sklearn; `scale_pos_weight` in XGBoost |
| 90/10 — 99/1 | SMOTE / undersampling + threshold tuning + PR-AUC metric |
| Extreme (>99/1) | Treat as anomaly detection, not classification |

**Why class_weight before SMOTE:** SMOTE creates synthetic minority points — risk of data leakage if done before the train/test split. `class_weight` just re-weights the loss — no new data, no leakage risk.

**Other EDA musts:**
- Missing-value pattern per feature (MCAR vs MAR vs MNAR — fix differently)
- High-cardinality categoricals (group rare levels or use target encoding with CV)
- Correlated features (drop or use a model robust to it — trees don't care, linear does)
- Look at the labels by hand for 10–20 examples — catch labelling bugs early

---

## 3. Train/val/test split — BEFORE any preprocessing

```python
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
```

**Why stratify=y:** preserves class distribution in each split. Critical if imbalanced.

**Why split BEFORE preprocessing:** any scaler, imputer, target encoder, or feature selector fitted on the full dataset leaks test info into training. Fit on train only, apply to val/test.

**Use a `Pipeline`** — it enforces the discipline automatically:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
])
pipe.fit(X_tr, y_tr)
```

---

## 4. Feature pipeline

| Feature type | Treatment |
|---|---|
| Numeric, ~Gaussian | StandardScaler (mean 0, std 1) |
| Numeric, skewed | log1p / Box-Cox / RobustScaler |
| Numeric, tree model | **No scaling needed** (trees are scale-invariant) |
| Low-card categorical (≤10) | OneHotEncoder |
| High-card categorical (>50) | Target encoding **inside CV**, or hashing trick |
| Ordinal | OrdinalEncoder (preserve order) |
| Date/time | Decompose: dayofweek, month, hour; cyclical encoding (sin/cos) |
| Text | TF-IDF, hashing vectorizer, or embeddings |
| Missing | Median impute + missingness indicator column |

**Target encoding trap:** encoding a category by mean target value leaks the label into the feature. Always use leave-one-out / k-fold target encoding, fitted on train only.

---

## 5. Model selection — start simple

| Family | When to use | Strengths | Weaknesses |
|---|---|---|---|
| **Logistic regression** | Baseline, low n, interpretability | Fast, calibrated, regularised | Linear only |
| **Linear SVM** | High-dim sparse (text) | Margin-maximising | Slow on big n |
| **Tree (single)** | Need interpretability | Easy to explain | Overfits, high variance |
| **Random Forest** | Tabular medium-n | Robust, no scaling, handles missing | Slow at scoring time |
| **Gradient Boosting (XGBoost/LightGBM/CatBoost)** | Tabular default | Usually best on tabular | Sensitive to hyperparams |
| **Neural net** | High-dim, images, text, n>100k | Flexible, end-to-end learning | Hungry for data, less interpretable |
| **Naive Bayes** | Spam-like text | Fast baseline | Strong assumptions |

**Rule of thumb:** always train a **logistic regression baseline first**. If your fancy model can't beat it convincingly, the fancy model is wrong somewhere.

---

## 6. Hyperparameter tuning

**With CV — never on a held-out test set:**

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
gs.fit(X_tr, y_tr)
```

**Why StratifiedKFold:** maintains class balance in every fold. Plain KFold can give a fold with zero positives at high imbalance.

**Bayesian search (Optuna) beats grid/random** once your search space is larger than ~3 dimensions. Especially for XGBoost.

**Nest the CV if you also tune feature selection.** Outer CV = generalisation estimate; inner CV = hyperparameter selection.

---

## 7. Evaluation metrics

**Pick the metric BEFORE training**, based on business cost:

| Metric | Formula intuition | When to use |
|---|---|---|
| **Accuracy** | Right / total | Balanced classes only. **Misleading** if imbalanced. |
| **Precision** | TP / (TP+FP) | False positives are costly (spam classifier) |
| **Recall (sensitivity)** | TP / (TP+FN) | False negatives are costly (cancer screening) |
| **F1** | Harmonic mean of P/R | Balanced concern, single number summary |
| **ROC-AUC** | Ranking quality across all thresholds | Threshold-independent; can be misleading on heavy imbalance |
| **PR-AUC** | Precision-recall area | **Default for imbalanced** — directly reflects rare-class performance |
| **Log-loss** | Punishes confident wrong predictions | When you need calibrated probabilities |
| **Brier score** | Mean squared error of probabilities | Calibration assessment |
| **Cohen's kappa** | Accuracy adjusted for chance | Multi-class with imbalance |

**Why ROC-AUC isn't enough for imbalance:** ROC counts the (huge) number of true negatives so AUC can stay high even when the minority class is wrong. PR-AUC ignores true negatives — directly reflects how well you find the rare class.

---

## 8. Threshold tuning (separate from model)

A classifier outputs probabilities. The threshold (default 0.5) is a **business decision**, not a model decision.

```python
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_va, proba_va)
# Pick threshold to hit target precision (e.g. 0.9) or target recall
```

Tune the threshold on the **validation set**, not the test set. Test set evaluates the final pipeline (model + threshold).

---

## 9. Probability calibration

Most models output uncalibrated scores. If you need probabilities to mean what they say (e.g. risk scoring, expected loss):

| Model | Calibration |
|---|---|
| Logistic regression | Already calibrated (usually) |
| Random Forest, GBM | Often overconfident — Platt or isotonic on a held-out set |
| Neural net | Often overconfident — temperature scaling |

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(base_estimator=gbm, method="isotonic", cv=5)
```

**Why this matters:** if you're using probabilities for downstream decisions (expected loss = prob × cost), uncalibrated probabilities give wrong answers.

---

## 10. Final test evaluation (once!)

```python
y_pred = pipe.predict(X_te)
y_proba = pipe.predict_proba(X_te)[:, 1]

print(classification_report(y_te, y_pred))
print(f"ROC-AUC : {roc_auc_score(y_te, y_proba):.4f}")
print(f"PR-AUC  : {average_precision_score(y_te, y_proba):.4f}")
```

**One look at the test set is the rule.** If you tune anything after seeing test results, the test set is no longer a generalisation estimate.

---

## 11. Diagnose errors

- **Confusion matrix** — which class confuses which?
- **Examine wrong predictions** — patterns? Specific feature ranges?
- **Calibration plot** — does P(model says 0.7) really equal 70% positives?
- **Feature importance** — does it match domain intuition? If not, suspect leakage
- **Permutation importance** — more reliable than tree-builtin importance for correlated features

---

## 12. Deployment + monitoring

| Concern | Tactic |
|---|---|
| Train-serve skew | Persist the **entire pipeline** (preprocessing + model), not just the model |
| Schema drift | Validate input schema at serve time |
| Concept drift | Monitor live metrics; PSI / KL on feature distributions vs training |
| Latency | XGBoost > sklearn pipeline in ~10× speed; ONNX export for prod |
| A/B test | Champion vs challenger before full cut-over |

---

## Common pitfalls

- **Data leakage** through target encoding, scaler-on-full-data, time-leaked features
- **Optimising the wrong metric** (accuracy on imbalanced data → trivial all-zero classifier wins)
- **No baseline** → no way to tell if your model is actually any good
- **Forgetting calibration** when downstream needs probabilities
- **Tuning the threshold on the test set** → overfit threshold
- **Train-serve schema mismatch** — column order, missing-value codes, categorical levels not in train
- **Trusting feature importance from one model alone** — use permutation or SHAP

---

## Quick-reference

| Decision | Default | Switch when |
|---|---|---|
| Train/val/test split | 70 / 10 / 20, stratified | Time series → chronological |
| Imbalance fix | `class_weight="balanced"` | Extreme imbalance → resampling + PR-AUC |
| Scaling | StandardScaler | Outliers → RobustScaler; trees → none |
| Baseline | Logistic regression | Always |
| Default tabular model | XGBoost / LightGBM | n < 1k → simpler |
| Tuning | Optuna (Bayesian) | Small space → GridSearchCV |
| CV | StratifiedKFold (k=5) | Imbalanced → k=10 + stratify |
| Primary metric | PR-AUC + log-loss | Balanced classes → ROC-AUC |
| Threshold | Tuned on val for business target | Default 0.5 only when costs equal |
| Calibration | Isotonic / Platt on val | Skip if model gives calibrated proba (LR) |

---

## What to remember

1. **Pick the metric, split, and baseline before you train.** Most modelling mistakes happen because these three decisions were made *after* seeing results.
2. **The model returns probabilities; the threshold is a business choice.** Two separate decisions, two separate tuning steps.
3. **Leakage kills.** Fit preprocessing on train only; encode target inside CV; don't peek at the test set until once at the end.
