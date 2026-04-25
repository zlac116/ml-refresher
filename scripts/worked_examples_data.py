"""Hand-written worked-example cell pairs, keyed by section header.

Each entry is `{header: {'intro': markdown_str, 'code': python_str}}`.

The intro is a 2-4 sentence markdown explanation; the code is a self-contained
demonstration the student can run, modify, and learn from before tackling the
section's exercises. Code uses small synthetic data wherever practical so that
behaviour is obvious and runs fast.

Inserted into each notebook between the section's hint block and its first
exercise.
"""

WORKED_EXAMPLES: dict[str, dict[str, str]] = {}


def _add(header: str, intro: str, code: str) -> None:
    WORKED_EXAMPLES[header] = {"intro": intro.strip("\n"), "code": code.strip("\n")}


# ======================================================================
# classification/classification.ipynb
# ======================================================================

_add(
    "### Exercises — Section 1",
    """
**Worked example — data hygiene on a tiny synthetic frame**

Before the exercises, here's the four idioms (groupby-diff for gaps, `isna().mean()` for
missingness, `pivot` for long→wide, `reindex` for continuity checks) all in one runnable
demo on a 10-row toy frame. Read each block, run the cell, then adapt the same idioms
to the real `df` in the exercises.
""",
    """
import pandas as pd
import numpy as np

# Toy: 3 symbols, hourly except for two deliberately missing bars.
ts = pd.to_datetime([
    '2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 03:00',  # gap at 02:00
    '2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 02:00',
    '2024-01-01 00:00', '2024-01-01 02:00', '2024-01-01 03:00',  # gap at 01:00
], utc=True)
toy = pd.DataFrame({
    'ts': ts,
    'symbol': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'close': [100, 101, np.nan, 50, 51, 52, 30, 31, 32],
})

# 1) Largest per-symbol time gap, in hours (groupby + diff + dt.total_seconds).
gaps = (toy.sort_values(['symbol', 'ts'])
            .groupby('symbol')['ts']
            .apply(lambda s: s.diff().dt.total_seconds().div(3600).max()))
print('Largest gap per symbol (hours):')
print(gaps, '\\n')

# 2) Percent missing per column.
print('Missing %:')
print((toy.isna().mean() * 100).round(2), '\\n')

# 3) Long → wide pivot.
wide = toy.pivot(index='ts', columns='symbol', values='close').sort_index()
print('Wide pivot:')
print(wide, '\\n')

# 4) Continuity check via reindex against an expected hourly grid.
expected = pd.date_range(toy['ts'].min(), toy['ts'].max(), freq='1h', tz='UTC')
n_missing = wide.reindex(expected).isna().sum()
print('Missing bars per symbol after reindex:')
print(n_missing)
""",
)

_add(
    "### Exercises — Section 2",
    """
**Worked example — rolling stats, moments, time-of-day patterns**

Demonstrates: `.rolling().corr()`, `scipy.stats.skew/kurtosis`, plotting a rolling mean
of a binary class to show drift, and bucketing by `index.hour` for an intraday view.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=24*30, freq='1h', tz='UTC')

# Two correlated synthetic returns: B = 0.6 * A + noise.
ret_a = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
ret_b = 0.6 * ret_a + pd.Series(rng.normal(0, 0.5, len(idx)), index=idx)

# 1) 24h rolling correlation.
corr = ret_a.rolling(24).corr(ret_b)
print(f'rolling 24h corr — last 3 values: {corr.dropna().tail(3).round(3).tolist()}')

# 2) Skew + excess kurtosis.
print(f'skew(A): {skew(ret_a):+.3f}    excess kurt(A): {kurtosis(ret_a, fisher=True):+.3f}')

# 3) Drift in a binary "up day" indicator over time.
up = (ret_a > 0).astype(int)
roll = up.rolling(24*7).mean()
fig, ax = plt.subplots(figsize=(9, 2.5))
roll.plot(ax=ax); ax.axhline(0.5, ls='--', color='k', alpha=0.5)
ax.set_title('rolling 7d fraction of up bars')
plt.tight_layout(); plt.show()

# 4) Intraday seasonality via groupby on index.hour.
by_hour = ret_a.groupby(ret_a.index.hour).mean()
print(f'\\nMean return by hour-of-day (first 6 hours): {by_hour.head(6).round(4).to_dict()}')
""",
)

_add(
    "### Exercises — Section 3",
    """
**Worked example — feature engineering with strict past-only data**

Builds three features the exercises will reuse: a manual recursive Wilder average (the
heart of RSI), a cross-asset return via `.shift()`, and a rolling z-score regime feature.
The leakage spot-check pattern is shown last.
""",
    """
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=300, freq='1h', tz='UTC')
close_a = pd.Series(100 + np.cumsum(rng.normal(0, 1, len(idx))), index=idx)
close_b = pd.Series(50  + np.cumsum(rng.normal(0, 0.5, len(idx))), index=idx)

# 1) Manual Wilder smoothing (simplified — same recursion as RSI uses).
def wilder(x: pd.Series, period: int) -> pd.Series:
    out = np.full(len(x), np.nan)
    out[period - 1] = x.iloc[:period].mean()
    for i in range(period, len(x)):
        out[i] = (out[i-1] * (period - 1) + x.iloc[i]) / period
    return pd.Series(out, index=x.index)

deltas = close_a.diff().fillna(0)
gains = wilder(deltas.clip(lower=0), 14)
print(f'wilder gain — last value: {gains.iloc[-1]:.4f}')

# 2) Cross-asset 24h return (always shift first; never look at future).
ret_b_24h = np.log(close_b / close_b.shift(24))
print(f'B 24h log-ret — last value: {ret_b_24h.iloc[-1]:+.4f}')

# 3) Rolling 7-day z-score (volatility-of-volatility regime).
vol = close_a.diff().abs().rolling(24).mean()
vol_z = (vol - vol.rolling(168).mean()) / vol.rolling(168).std()
print(f'vol z-score — last value: {vol_z.iloc[-1]:+.3f}')

# 4) Leakage spot-check: corrupt the future, regenerate features, assert past unchanged.
t_cut = idx[200]
mut = close_a.copy(); mut.loc[mut.index > t_cut] = 0.0
mut_vol = mut.diff().abs().rolling(24).mean()
assert vol.loc[:t_cut].equals(mut_vol.loc[:t_cut]), 'LEAK!'
print(f'\\nleakage check at {t_cut}: OK — past values are bit-identical')
""",
)

_add(
    "### Exercises — Section 4",
    """
**Worked example — splits, leakage, and the toy that exposes them**

Three building blocks the four exercises reuse: a walk-forward split function, a
disjointness assertion, and the smoking-gun comparison of shuffled K-fold vs a
chronological hold-out on a deliberately auto-correlated synthetic series.
""",
    """
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

# 1) Walk-forward splits: expanding train, fixed-size val.
def wf_splits(n, k=3, min_train=10):
    fold = (n - min_train) // k
    for i in range(k):
        tr_end = min_train + i * fold
        yield np.arange(0, tr_end), np.arange(tr_end, tr_end + fold)

print('walk-forward folds (n=40, k=3, min_train=10):')
for i, (tr, va) in enumerate(wf_splits(40, k=3, min_train=10)):
    print(f'  fold {i}: train rows 0..{tr[-1]}  val rows {va[0]}..{va[-1]}')

# 2) Disjointness check using sets — fast and reads clearly in a test.
train_idx, val_idx = set(range(0, 30)), set(range(30, 40))
assert train_idx.isdisjoint(val_idx); print('\\ntrain/val disjoint OK')

# 3) Why shuffled K-fold leaks on autocorrelated data.
#    Build a tiny toy where y_t is highly correlated with y_{t-1}.
rng = np.random.default_rng(42)
n_demo = 80
y_demo = pd.Series(rng.choice([0, 1], size=n_demo))
X_demo = pd.DataFrame({'y_lag1': y_demo.shift(1).fillna(0).astype(int)})

shuffled = cross_val_score(LogisticRegression(), X_demo, y_demo,
                           cv=KFold(5, shuffle=True, random_state=0),
                           scoring='accuracy').mean()
cut = int(n_demo * 0.7)
chrono = LogisticRegression().fit(X_demo.iloc[:cut], y_demo.iloc[:cut]).score(
    X_demo.iloc[cut:], y_demo.iloc[cut:])
print(f'\\nshuffled k-fold:        {shuffled:.3f}   <- inflated by leakage')
print(f'chronological holdout:  {chrono:.3f}   <- realistic')
""",
)

_add(
    "### Exercises — Section 5",
    """
**Worked example — baselines you must beat**

Demonstrates four baseline patterns: a manual majority predictor, a directional
sign-of-last-return baseline, building a tidy comparison table, and a bootstrap CI
on AUC. Same idioms work for any binary task.
""",
    """
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

rng = np.random.default_rng(0)
n = 400
y = pd.Series(rng.choice([0, 1], size=n, p=[0.45, 0.55]))   # slightly biased
last_sign = pd.Series(rng.choice([0, 1], size=n))           # noisy proxy for "last return sign"

# 1) Majority baseline (no sklearn).
maj = int(y.mode().iloc[0])
y_maj = np.full(len(y), maj)
print(f'majority predicts class {maj}, accuracy={accuracy_score(y, y_maj):.3f}')

# 2) "Predict last sign" baseline as a probability.
proba_last = last_sign.astype(float).values
print(f'predict-last-sign AUC: {roc_auc_score(y, proba_last):.3f}')

# 3) Combined metrics table (rows of dicts is the cleanest pattern here).
def evaluate(name, y_true, proba):
    return {'name': name,
            'acc': accuracy_score(y_true, (proba >= 0.5).astype(int)),
            'auc': roc_auc_score(y_true, proba)}
table = pd.DataFrame([
    evaluate('majority',          y, np.full(len(y), maj, dtype=float)),
    evaluate('predict_last_sign', y, proba_last),
])
print('\\n', table.round(3), sep='')

# 4) Bootstrap 95% CI on the predict-last-sign AUC.
boot = []
for _ in range(500):
    idx = rng.integers(0, len(y), len(y))
    boot.append(roc_auc_score(y.iloc[idx], proba_last[idx]))
lo, hi = np.quantile(boot, [0.025, 0.975])
print(f'\\nbootstrap 95% CI on AUC: [{lo:.3f}, {hi:.3f}]')
""",
)

_add(
    "### Exercises — Section 6",
    """
**Worked example — calibrate, expand features, ensemble, ablate**

Four model-building idioms on a tiny synthetic dataset: Platt-style calibration,
adding interaction features via `PolynomialFeatures`, a simple stacking ensemble,
and a feature-subset ablation that quantifies a feature group's contribution.
""",
    """
import numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import log_loss
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=400, n_features=6, random_state=0)
X = pd.DataFrame(X, columns=[f'f{i}' for i in range(6)])
X_tr, X_va, y_tr, y_va = X[:300], X[300:], y[:300], y[300:]

# 1) Calibration of an already-fit model.
# In sklearn ≥1.6 the old `cv='prefit'` was replaced by FrozenEstimator + a small CV.
gbm = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
calibrated = CalibratedClassifierCV(FrozenEstimator(gbm),
                                     method='isotonic', cv=5).fit(X_va, y_va)
print(f'raw  log-loss: {log_loss(y_va, gbm.predict_proba(X_va)[:, 1]):.3f}')
print(f'cal. log-loss: {log_loss(y_va, calibrated.predict_proba(X_va)[:, 1]):.3f}')

# 2) Polynomial / interaction features inside a pipeline.
poly = Pipeline([
    ('sc',   StandardScaler()),
    ('poly', PolynomialFeatures(2, interaction_only=True, include_bias=False)),
    ('lr',   LogisticRegression(max_iter=2000)),
]).fit(X_tr, y_tr)
print(f'\\npoly-LR val acc: {poly.score(X_va, y_va):.3f}')

# 3) Stacking — combine GBM + poly-LR with a meta-LR.
stack = StackingClassifier(
    estimators=[('gbm', GradientBoostingClassifier(random_state=0)),
                ('poly', poly)],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=3,
).fit(X_tr, y_tr)
print(f'stack    val acc: {stack.score(X_va, y_va):.3f}')

# 4) Ablation: drop a feature subset and refit GBM.
drop = ['f4', 'f5']
abl = GradientBoostingClassifier(random_state=0).fit(X_tr.drop(columns=drop), y_tr)
print(f'\\nGBM without {drop}: val acc {abl.score(X_va.drop(columns=drop), y_va):.3f}')
""",
)

_add(
    "### Exercises — Section 7",
    """
**Worked example — Optuna mechanics on a 1-D problem**

A self-contained Optuna study where you can see pruning, search-space widening, and a
storage-backed study in <20 lines. Once these mechanics click, applying them to a real
model objective is mostly bookkeeping.
""",
    """
import optuna
import os, tempfile
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1) Plain optuna study minimising (x-3)**2 — best x is 3.
def obj_basic(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 3) ** 2

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(obj_basic, n_trials=20, show_progress_bar=False)
print(f'best x: {study.best_params["x"]:.3f}  (target=3)  best value: {study.best_value:.4f}')

# 2) Pruning: report intermediate scores and prune unpromising trials.
def obj_pruned(trial):
    a = trial.suggest_float('a', -5, 5)
    b = trial.suggest_float('b', -5, 5)
    cumulative = 0.0
    for step in range(5):
        cumulative += (a - 1) ** 2 + (b - 2) ** 2  # simulate "fold scores"
        trial.report(cumulative, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return cumulative

study2 = optuna.create_study(direction='minimize',
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
study2.optimize(obj_pruned, n_trials=20, show_progress_bar=False)
n_pruned = sum(1 for t in study2.trials if t.state.name == 'PRUNED')
print(f'\\npruning study: {len(study2.trials)} trials, {n_pruned} pruned early')

# 3) SQLite-backed study you can resume across processes.
db = os.path.join(tempfile.gettempdir(), 'demo_study.db')
if os.path.exists(db): os.remove(db)
storage = f'sqlite:///{db}'
optuna.create_study(direction='minimize', study_name='demo',
                    storage=storage).optimize(obj_basic, n_trials=5)
reloaded = optuna.load_study(study_name='demo', storage=storage)
print(f'reloaded study: {len(reloaded.trials)} trials, best={reloaded.best_value:.4f}')
""",
)

_add(
    "### Exercises — Section 8",
    """
**Worked example — explain a model with permutation importance and SHAP**

Four interpretability primitives demonstrated end-to-end on a small classifier:
permutation importance (model-agnostic), SHAP TreeExplainer, ranking by mean |SHAP|,
and a single-row waterfall.
""",
    """
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
import shap

X, y = make_classification(n_samples=300, n_features=6, n_informative=3, random_state=0)
X = pd.DataFrame(X, columns=[f'f{i}' for i in range(6)])
X_tr, X_va, y_tr, y_va = X[:200], X[200:], y[:200], y[200:]
model = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)

# 1) Permutation importance.
perm = permutation_importance(model, X_va, y_va, n_repeats=5, random_state=0)
perm_imp = pd.Series(perm.importances_mean, index=X_va.columns).sort_values(ascending=False)
print('permutation importance (val):')
print(perm_imp.round(4), '\\n')

# 2) SHAP values via TreeExplainer.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_va)
mean_abs = np.abs(shap_values).mean(axis=0)
shap_imp = pd.Series(mean_abs, index=X_va.columns).sort_values(ascending=False)
print('SHAP mean |value| (val):')
print(shap_imp.round(4))

# 3) Top-3 features only — refit and compare val score.
top3 = shap_imp.head(3).index.tolist()
small = GradientBoostingClassifier(random_state=0).fit(X_tr[top3], y_tr)
print(f'\\ntop-3 features only ({top3}): val acc {small.score(X_va[top3], y_va):.3f}')
print(f'all features:              val acc {model.score(X_va, y_va):.3f}')

# 4) Waterfall for one row (largest predicted prob).
i = int(np.argmax(model.predict_proba(X_va)[:, 1]))
exp = explainer(X_va.iloc[[i]])
shap.plots.waterfall(exp[0], max_display=6, show=True)
""",
)

_add(
    "### Exercises — Section 9",
    """
**Worked example — picking a threshold and quantifying noise**

Demonstrates a threshold sweep for an arbitrary metric, a profit-based threshold,
plotting precision/recall vs threshold, and a bootstrap CI for AUC. The exercises
reuse exactly these patterns on the real model's predictions.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)

rng = np.random.default_rng(0)
n = 400
y_true = rng.choice([0, 1], size=n, p=[0.55, 0.45])
proba  = np.clip(0.5 + 0.4 * (y_true - 0.5) + rng.normal(0, 0.2, n), 0, 1)

# 1) F1-optimal threshold via sweep.
ts = np.linspace(0.30, 0.70, 41)
best_t = float(ts[np.argmax([f1_score(y_true, (proba >= t).astype(int)) for t in ts])])
print(f'F1-optimal threshold: {best_t:.3f}')

# 2) Profit-based threshold (+1 for tp/tn, -1 for fp/fn).
def profit(yt, yp):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return tp - fp + tn - fn
profit_t = float(ts[np.argmax([profit(y_true, (proba >= t).astype(int)) for t in ts])])
print(f'profit-optimal threshold: {profit_t:.3f}')

# 3) Precision and recall vs threshold.
prec = [precision_score(y_true, (proba >= t).astype(int), zero_division=0) for t in ts]
rec  = [recall_score(y_true, (proba >= t).astype(int))                       for t in ts]
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(ts, prec, label='precision'); ax.plot(ts, rec, label='recall')
ax.set_xlabel('threshold'); ax.legend(); plt.tight_layout(); plt.show()

# 4) Bootstrap 95% CI on AUC.
boot = []
for _ in range(500):
    idx = rng.integers(0, n, n)
    boot.append(roc_auc_score(y_true[idx], proba[idx]))
lo, hi = np.quantile(boot, [0.025, 0.975])
print(f'AUC bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]')
""",
)

_add(
    "### Exercises — Section 10",
    """
**Worked example — package a model for inference**

Demonstrates the four production-shaped helpers: a stateless `predict_one`, a
pytest-style assertion, a Pydantic schema for the input row, and a `/health`
payload. Same shapes as you'd ship to a FastAPI service.
""",
    """
import joblib, tempfile, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from pydantic import create_model, ValidationError

# Pretend we trained and saved an artifact bundle.
X, y = make_classification(n_samples=200, n_features=4, random_state=0)
FEATS = [f'f{i}' for i in range(4)]
X = pd.DataFrame(X, columns=FEATS)
m = LogisticRegression().fit(X, y)
art_path = os.path.join(tempfile.gettempdir(), 'demo_model.joblib')
joblib.dump({'model': m, 'feature_names': FEATS, 'trained_through': '2024-01-01'}, art_path)

# 1) Stateless predict_one — loads the artifact, validates feature order.
def predict_one(row: dict, path: str = art_path, threshold: float = 0.5) -> dict:
    art = joblib.load(path)
    feats = art['feature_names']
    df = pd.DataFrame([row])[feats]
    prob = float(art['model'].predict_proba(df)[0, 1])
    return {'prob_up': prob, 'label': int(prob >= threshold)}

sample = X.iloc[0].to_dict()
print('predict_one sample:', predict_one(sample))

# 2) Pytest-style assertion.
def test_predict_one_in_range():
    out = predict_one(sample)
    assert 0.0 <= out['prob_up'] <= 1.0, out
    assert out['label'] in (0, 1)
test_predict_one_in_range(); print('pytest-style test passed')

# 3) Pydantic input schema generated from the feature list.
RowSchema = create_model('RowSchema', **{f: (float, ...) for f in FEATS})
RowSchema(**sample)  # valid → no exception
try:
    RowSchema(**{**sample, 'f0': 'not_a_float'})
except ValidationError as e:
    print('pydantic rejects bad type — OK')

# 4) /health payload.
def health_payload(path=art_path):
    art = joblib.load(path)
    return {'status': 'ok',
            'trained_through': art['trained_through'],
            'n_features': len(art['feature_names'])}
print('health:', health_payload())
""",
)


# ======================================================================
# regression/regression.ipynb
# ======================================================================

_add(
    "### Exercises (Section 2)",
    """
**Worked example — per-symbol hygiene on a tiny multi-asset frame**

Shows the four idioms you'll use: per-symbol missing %, the longest gap location,
expected hourly grid via `pd.date_range`, and rows-per-day via `dt.floor('D')`.
""",
    """
import numpy as np, pandas as pd

ts = pd.to_datetime([
    '2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 04:00',  # 3h gap on A
    '2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 02:00',
], utc=True)
toy = pd.DataFrame({'ts': ts, 'symbol': ['A','A','A','B','B','B'],
                    'close': [100, np.nan, 103, 50, 51, 52]})

# 1) Per-symbol % missing.
miss = (toy.set_index('ts').groupby('symbol').apply(lambda g: g.isna().mean() * 100))
print('% missing per symbol:'); print(miss, '\\n')

# 2) Longest gap location for symbol A.
a_ts = toy[toy.symbol == 'A']['ts'].sort_values().reset_index(drop=True)
diffs = a_ts.diff()
i = int(diffs.idxmax())
print(f'A longest gap: {diffs.iloc[i]} between {a_ts.iloc[i-1]} and {a_ts.iloc[i]}')

# 3) Expected hourly grid vs actual — surfaces missing timestamps.
expected = pd.date_range(a_ts.min(), a_ts.max(), freq='1h', tz='UTC')
print(f'\\nA missing timestamps: {expected.difference(a_ts).tolist()}')

# 4) Rows per calendar day.
per_day = toy.groupby(toy['ts'].dt.floor('D')).size()
print('\\nrows per day:'); print(per_day)
""",
)

_add(
    "### Exercises (Section 3)",
    """
**Worked example — returns, vol, and seasonality**

Demonstrates rolling realised vol, skew/kurtosis, weekday seasonality boxplot, and a
lag-scatter that exposes vol-of-vol persistence. Same idioms as the four exercises.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=24*120, freq='1h', tz='UTC')
log_ret = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)

# 1) Rolling 24h realised vol.
rv = np.sqrt((log_ret ** 2).rolling(24).sum()) * np.sqrt(24)
print(f'rolling 24h RV — mean: {rv.mean():.4f}')

# 2) Skew / excess kurt of the return distribution.
print(f'skew: {stats.skew(log_ret):+.3f}    excess kurt: {stats.kurtosis(log_ret):+.3f}')

# 3) Weekday boxplot of realised vol.
df = pd.DataFrame({'rv': rv, 'day': rv.index.day_name()}).dropna()
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
fig, ax = plt.subplots(figsize=(8, 3))
sns.boxplot(data=df, x='day', y='rv', order=order, showfliers=False, ax=ax)
plt.tight_layout(); plt.show()

# 4) Lag scatter: rv(t) vs rv(t-24).
lag = pd.DataFrame({'rv': rv, 'rv_lag24': rv.shift(24)}).dropna()
print(f'\\ncorr(rv(t), rv(t-24)) = {lag.corr().iloc[0,1]:+.3f}  (positive → persistence)')
""",
)

_add(
    "### Exercises (Section 4)",
    """
**Worked example — vol estimators and HAR features**

Implements Parkinson, Garman-Klass, and a small HAR-style lag stack on synthetic OHLC.
Always shift before rolling so each row's features depend strictly on past data.
""",
    """
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=24*30, freq='1h', tz='UTC')
ret = rng.normal(0, 0.01, len(idx))
close = 100 * np.exp(np.cumsum(ret))
hi = close * (1 + np.abs(rng.normal(0, 0.005, len(idx))))
lo = close * (1 - np.abs(rng.normal(0, 0.005, len(idx))))
op = np.r_[close[0], close[:-1]]
df = pd.DataFrame({'open': op, 'high': hi, 'low': lo, 'close': close}, index=idx)

# 1) Parkinson estimator.
def parkinson(high, low, w):
    return np.sqrt((np.log(high/low) ** 2).rolling(w).sum() / (4 * np.log(2)))
p_vol = parkinson(df['high'], df['low'], 24)
print(f'Parkinson 24h vol — last: {p_vol.iloc[-1]:.4f}')

# 2) Garman-Klass estimator.
def garman_klass(o, h, l, c, w):
    hl = np.log(h / l); co = np.log(c / o)
    var = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    return np.sqrt(var.rolling(w).sum())
gk = garman_klass(df['open'], df['high'], df['low'], df['close'], 24)
print(f'Garman-Klass 24h    — last: {gk.iloc[-1]:.4f}')

# 3) HAR-style lag features (always shift first to keep things past-only).
log_ret = np.log(df['close']).diff()
rv1 = (log_ret ** 2).rolling(1).sum() ** 0.5
har = pd.DataFrame({
    'har_lag1':   rv1.shift(1),
    'har_24havg': rv1.shift(1).rolling(24).mean(),
    'har_168havg': rv1.shift(1).rolling(168).mean(),
}).dropna()
print('\\nHAR features tail:'); print(har.tail(3).round(5))
""",
)

_add(
    "### Exercises (Section 5)",
    """
**Worked example — temporal splits and overlap asserts**

Demonstrates an expanding walk-forward generator, a chronological train/val/test split
with a strict overlap assertion, and the purging idiom that drops train rows close to
the val window's start.
""",
    """
import numpy as np, pandas as pd

n = 1000

# 1) Expanding walk-forward.
def expanding_wf(n, n_splits=4, val_size=100):
    initial = n - n_splits * val_size
    for k in range(n_splits):
        tr_end = initial + k * val_size
        yield np.arange(0, tr_end), np.arange(tr_end, tr_end + val_size)

print('expanding walk-forward:')
for k, (tr, va) in enumerate(expanding_wf(n)):
    print(f'  fold {k}: train=[0,{tr[-1]}]  val=[{va[0]},{va[-1]}]')

# 2) Chronological 70/15/15 split with overlap assert.
idx = pd.RangeIndex(n)
tr_end = int(n * 0.7); va_end = int(n * 0.85)
train, val, test = idx[:tr_end], idx[tr_end:va_end], idx[va_end:]
assert train.max() < val.min() < test.min(), 'temporal overlap!'
print(f'\\ntrain {len(train)}, val {len(val)}, test {len(test)} — no overlap')

# 3) Purging: drop training rows within `horizon` bars of fold_start.
def purged_train(train_idx, fold_start, horizon=24):
    return train_idx[train_idx < fold_start - horizon]
print(f'purged train (h=24, fold@500) keeps {len(purged_train(train, 500, 24))} rows '
      f'(was {len(train)})')
""",
)

_add(
    "### Exercises (Section 6)",
    """
**Worked example — vol-aware baselines and bootstrap CIs**

A persistence baseline, the QLIKE volatility loss, a tidy comparison table, and a
bootstrap CI on MAE — all on a few hundred synthetic samples.
""",
    """
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

rng = np.random.default_rng(0)
n = 500
y_true = np.clip(rng.normal(0.02, 0.005, n), 1e-4, None)        # realised vol
y_pers = np.r_[y_true[0], y_true[:-1]]                           # predict yesterday's
y_mean = np.full(n, y_true.mean())                               # predict mean

# 1) Persistence vs mean baseline metrics.
def metrics(yt, yp): return {'MAE': mean_absolute_error(yt, yp),
                              'RMSE': math.sqrt(mean_squared_error(yt, yp))}
print('persistence:', {k: round(v, 5) for k, v in metrics(y_true, y_pers).items()})
print('mean       :', {k: round(v, 5) for k, v in metrics(y_true, y_mean).items()})

# 2) QLIKE — penalises under-prediction more than over-prediction.
def qlike(yt_vol, yp_vol, eps=1e-12):
    yt, yp = yt_vol**2 + eps, yp_vol**2 + eps
    return float(np.mean(yt/yp - np.log(yt/yp) - 1.0))
print(f'\\nQLIKE persistence: {qlike(y_true, y_pers):.5f}')
print(f'QLIKE mean       : {qlike(y_true, y_mean):.5f}')

# 3) Tidy table sorted by MAE.
table = pd.DataFrame([{'model': 'persistence', **metrics(y_true, y_pers)},
                      {'model': 'mean',        **metrics(y_true, y_mean)}]).sort_values('MAE')
print('\\n', table.round(5), sep='')

# 4) Bootstrap 95% CI on persistence MAE.
err = np.abs(y_true - y_pers)
boot = [err[rng.integers(0, n, n)].mean() for _ in range(500)]
print(f'\\npersistence MAE 95% CI: [{np.quantile(boot, 0.025):.5f}, {np.quantile(boot, 0.975):.5f}]')
""",
)

_add(
    "### Exercises (Section 7)",
    """
**Worked example — Pipelines, tradeoff plots, perm importance, ensembles**

Wrap a model in a scaling pipeline, plot fit-time vs MAE for several models,
permutation-rank features for a tree, and average two models' predictions.
""",
    """
import numpy as np, pandas as pd, time
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=600, n_features=8, noise=5, random_state=0)
X = pd.DataFrame(X, columns=[f'f{i}' for i in range(8)])
X_tr, X_va, y_tr, y_va = X[:400], X[400:], y[:400], y[400:]

# 1) Pipeline with scaler + linear model.
ridge_pipe = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]).fit(X_tr, y_tr)
print(f'Ridge pipeline val MAE: {mean_absolute_error(y_va, ridge_pipe.predict(X_va)):.3f}')

# 2) Fit-time vs MAE table.
models = {'Ridge': ridge_pipe,
          'RF':    RandomForestRegressor(n_estimators=100, random_state=0),
          'GBM':   GradientBoostingRegressor(random_state=0)}
rows = []
for name, mdl in models.items():
    t0 = time.time()
    mdl.fit(X_tr, y_tr)
    rows.append({'model': name,
                 'fit_seconds': time.time() - t0,
                 'mae': mean_absolute_error(y_va, mdl.predict(X_va))})
table = pd.DataFrame(rows); print('\\n', table.round(3), sep='')
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(table['fit_seconds'], table['mae'])
for _, r in table.iterrows():
    ax.annotate(r['model'], (r['fit_seconds'], r['mae']))
ax.set_xlabel('fit (s)'); ax.set_ylabel('val MAE'); plt.tight_layout(); plt.show()

# 3) Permutation importance for the RF.
rf = models['RF']
pi = permutation_importance(rf, X_va, y_va, n_repeats=5, random_state=0)
print('\\nperm importance (RF):',
      pd.Series(pi.importances_mean, index=X_va.columns).sort_values(ascending=False).head(3).round(3).to_dict())

# 4) Average top-2 ensemble.
preds = np.column_stack([rf.predict(X_va), models['GBM'].predict(X_va)]).mean(axis=1)
print(f'avg-of-2 val MAE: {mean_absolute_error(y_va, preds):.3f}')
""",
)

_add(
    "### Exercises (Section 8)",
    """
**Worked example — log-target tricks and residual plots**

Demonstrates fitting in log-space (with predict-then-`np.exp` and clipping), comparing
RMSE in the original units, Box-Cox, and a residual-vs-fitted plot to spot
heteroscedasticity.
""",
    """
import numpy as np, pandas as pd, math
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

X, y_raw = make_regression(n_samples=400, n_features=4, noise=20, random_state=0)
y_raw = np.clip(y_raw - y_raw.min() + 1, 1, None)  # strictly positive
X = pd.DataFrame(X, columns=[f'f{i}' for i in range(4)])
X_tr, X_va, y_tr, y_va = X[:300], X[300:], y_raw[:300], y_raw[300:]

# 1) Predict in log space, invert with np.exp + clipping.
def predict_log_space(model, X, lo=-15, hi=15):
    return np.exp(np.clip(model.predict(X), lo, hi))
m_log = GradientBoostingRegressor(random_state=0).fit(X_tr, np.log(y_tr))
m_raw = GradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)

# 2) RMSE comparison on the SAME (raw) units.
rmse_raw = math.sqrt(mean_squared_error(y_va, m_raw.predict(X_va)))
rmse_log = math.sqrt(mean_squared_error(y_va, predict_log_space(m_log, X_va)))
print(f'raw target RMSE: {rmse_raw:.3f}')
print(f'log target RMSE (back to raw units): {rmse_log:.3f}')

# 3) Box-Cox: y_bc, lambda; invert via inverse-transform formula.
y_bc, lam = stats.boxcox(y_tr)
m_bc = GradientBoostingRegressor(random_state=0).fit(X_tr, y_bc)
pred_bc = m_bc.predict(X_va)
y_va_pred = (pred_bc * lam + 1) ** (1 / lam)
print(f'Box-Cox lambda: {lam:.3f}    RMSE: {math.sqrt(mean_squared_error(y_va, y_va_pred)):.3f}')

# 4) Residual-vs-fitted plot (raw vs log) — funnel = heteroscedasticity.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].scatter(m_raw.predict(X_va), y_va - m_raw.predict(X_va), s=8, alpha=0.5)
axes[0].set_title('raw target — residual vs fitted')
axes[1].scatter(predict_log_space(m_log, X_va), y_va - predict_log_space(m_log, X_va), s=8, alpha=0.5)
axes[1].set_title('log target — residual vs fitted')
for ax in axes: ax.axhline(0, color='k', alpha=0.3)
plt.tight_layout(); plt.show()
""",
)

_add(
    "### Exercises (Section 9)",
    """
**Worked example — Optuna mechanics for regression**

Same Optuna primitives as the classification chapter — pruning, extended search space,
SQLite-backed study — wrapped around an LGBM regression objective so you can see what
plugging into a real pipeline looks like.
""",
    """
import optuna, lightgbm as lgb, numpy as np, os, tempfile
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.datasets import make_regression
optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = make_regression(n_samples=600, n_features=5, noise=5, random_state=0)

# 1) Pruned objective: report each fold's MAE so bad trials abort early.
def objective(trial):
    params = dict(
        objective='regression_l1',
        learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int('leaves', 15, 127),
        n_estimators=200, verbosity=-1, random_state=0)
    fold_mae = []
    for k, (tr, va) in enumerate(TimeSeriesSplit(3).split(X)):
        m = lgb.LGBMRegressor(**params).fit(X[tr], y[tr])
        mae = mean_absolute_error(y[va], m.predict(X[va]))
        fold_mae.append(mae)
        trial.report(np.mean(fold_mae), step=k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_mae))

study = optuna.create_study(direction='minimize',
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=8, show_progress_bar=False)
n_pruned = sum(1 for t in study.trials if t.state.name == 'PRUNED')
print(f'study: {len(study.trials)} trials, {n_pruned} pruned, best MAE {study.best_value:.4f}')
print('best params:', study.best_params)

# 2) SQLite-backed study so you can resume across processes.
db = os.path.join(tempfile.gettempdir(), 'demo_reg_study.db')
if os.path.exists(db): os.remove(db)
storage = f'sqlite:///{db}'
optuna.create_study(direction='minimize', study_name='reg_demo',
                    storage=storage).optimize(objective, n_trials=2)
reloaded = optuna.load_study(study_name='reg_demo', storage=storage)
print(f'\\nreloaded study: {len(reloaded.trials)} trials, best={reloaded.best_value:.4f}')
""",
)

_add(
    "### Exercises (Section 10)",
    """
**Worked example — interpret a regression model**

LGBM gain importance, permutation importance, a SHAP dependence plot, and a single-row
waterfall — same toolkit as the classification interpretability section, retargeted at
a regressor.
""",
    """
import numpy as np, pandas as pd, lightgbm as lgb, shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=400, n_features=6, n_informative=3, random_state=0)
X = pd.DataFrame(X, columns=[f'f{i}' for i in range(6)])
X_tr, X_va, y_tr, y_va = X[:300], X[300:], y[:300], y[300:]
model = lgb.LGBMRegressor(n_estimators=200, verbosity=-1, random_state=0).fit(X_tr, y_tr)

# 1) Built-in gain importance.
gain = pd.Series(model.booster_.feature_importance(importance_type='gain'),
                 index=X.columns).sort_values(ascending=False)
print('gain importance:'); print(gain.round(1), '\\n')

# 2) Permutation importance on val (model-agnostic).
pi = permutation_importance(model, X_va, y_va, n_repeats=5, random_state=0)
perm = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
print('permutation importance:'); print(perm.round(3), '\\n')

# 3) SHAP dependence plot for the top feature.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_va)
top = perm.index[0]
shap.dependence_plot(top, shap_values, X_va, show=True)

# 4) Single-row waterfall on the most extreme prediction.
i = int(np.argmax(model.predict(X_va)))
exp = explainer(X_va.iloc[[i]])
shap.plots.waterfall(exp[0], max_display=6, show=True)
""",
)

_add(
    "### Exercises (Section 11)",
    """
**Worked example — residual diagnostics**

Ljung-Box for autocorrelation, rolling residual std for conditional heteroscedasticity,
quantile bucketing for reliability, and a quantile-quantile calibration plot.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

rng = np.random.default_rng(0)
n = 500
y_true = rng.normal(0, 1, n)
# Predictions: well-calibrated on average, but heteroscedastic residuals.
y_pred = y_true + rng.normal(0, 0.5 + 0.5 * (np.arange(n) > 250), n)
resid  = y_true - y_pred

# 1) Ljung-Box on residuals (lags 12, 24).
lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
print('Ljung-Box:'); print(lb.round(4))

# 2) Rolling residual std exposes conditional heteroscedasticity.
rs = pd.Series(resid).rolling(50).std()
fig, ax = plt.subplots(figsize=(8, 2.5))
rs.plot(ax=ax); ax.set_title('rolling 50-row residual std'); plt.tight_layout(); plt.show()

# 3) Reliability buckets via qcut.
df = pd.DataFrame({'pred': y_pred, 'true': y_true})
df['bucket'] = pd.qcut(df['pred'], 5, labels=[f'Q{i}' for i in range(1, 6)])
agg = df.groupby('bucket', observed=True)[['pred', 'true']].mean()
print('\\nbucket-level reliability (mean pred vs mean true):'); print(agg.round(3))

# 4) Quantile-quantile calibration plot.
ts = np.linspace(0, 1, 50)
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(np.quantile(y_pred, ts), np.quantile(y_true, ts), 'o-')
lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5)
ax.set_xlabel('pred quantile'); ax.set_ylabel('true quantile'); plt.tight_layout(); plt.show()
""",
)

_add(
    "### Exercises (Section 12)",
    """
**Worked example — slicing test performance**

Per-period MAE, tail-focused MAE, model-vs-baseline overlay plot, and a simple win-rate
comparison. These idioms turn aggregate metrics into actionable insight.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=24*60, freq='1h', tz='UTC')
y_true = pd.Series(rng.normal(0.02, 0.005, len(idx)), index=idx)
y_pred = y_true + rng.normal(0, 0.001, len(idx))   # decent model
y_pers = y_true.shift(1).fillna(y_true.iloc[0])     # persistence baseline

# 1) Per-month MAE.
df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'pers': y_pers})
df['month'] = df.index.to_period('M')
monthly = df.groupby('month').apply(lambda g: mean_absolute_error(g['true'], g['pred']))
print('per-month MAE:'); print(monthly.round(5))

# 2) Tail-focused MAE (top decile of truth).
thr = np.quantile(y_true, 0.9)
mask = y_true >= thr
print(f'\\nMAE on top decile (n={mask.sum()}): {mean_absolute_error(y_true[mask], y_pred[mask]):.5f}')
print(f'MAE overall:                    {mean_absolute_error(y_true, y_pred):.5f}')

# 3) Model vs persistence overlay on a short window.
window = df.iloc[:7*24]
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(window.index, window['true'], 'k-', lw=1, label='true', alpha=0.6)
ax.plot(window.index, window['pred'], 'C0-', label='model')
ax.plot(window.index, window['pers'], 'C3-', label='persistence', alpha=0.7)
ax.legend(); plt.tight_layout(); plt.show()

# 4) Win-rate vs persistence.
beats = (np.abs(df['true'] - df['pred']) < np.abs(df['true'] - df['pers'])).mean() * 100
print(f'model beats persistence {beats:.1f}% of hours')
""",
)

_add(
    "### Exercises (Section 13)",
    """
**Worked example — quantile regression and coverage**

Train two LGBM quantile models (p10, p90), check empirical coverage of the interval,
plot interval width over time, and apply isotonic recalibration when coverage is off.
""",
    """
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=600, n_features=4, noise=15, random_state=0)
X_tr, X_va, X_te = X[:400], X[400:500], X[500:]
y_tr, y_va, y_te = y[:400], y[400:500], y[500:]

def fit_q(alpha):
    return lgb.LGBMRegressor(objective='quantile', alpha=alpha,
                              n_estimators=200, verbosity=-1, random_state=0).fit(X_tr, y_tr)

# 1) Two-quantile model (p10 / p90) + empirical coverage.
m_lo, m_hi = fit_q(0.1), fit_q(0.9)
p10, p90 = m_lo.predict(X_te), m_hi.predict(X_te)
cov = ((y_te >= p10) & (y_te <= p90)).mean()
print(f'empirical coverage [p10, p90]: {cov*100:.1f}%   (target 80%)')

# 2) Interval width over time.
width = p90 - p10
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.plot(width); ax.set_title('p90 - p10 width'); plt.tight_layout(); plt.show()

# 3) Isotonic recalibration of p90: target alpha=0.9 empirical coverage of "below".
val_p90 = m_hi.predict(X_va)
target = (y_va <= val_p90).astype(float)
ir = IsotonicRegression(out_of_bounds='clip').fit(val_p90, target)
adj = ir.predict(p90)
print(f'\\nisotonic-adjusted p90 mean below-rate on test: {(y_te <= p90).mean():.3f}')

# 4) p05 / p95 — wider band, coverage should be ~90%.
m05, m95 = fit_q(0.05), fit_q(0.95)
p05, p95 = m05.predict(X_te), m95.predict(X_te)
print(f'coverage [p05, p95]: {((y_te >= p05) & (y_te <= p95)).mean()*100:.1f}%   (target 90%)')
""",
)

_add(
    "### Exercises (Section 14)",
    """
**Worked example — package a regression model with quantiles**

Same shape as the classification deployment chapter: a stateless `predict_one`, a
property-based assertion that quantiles are monotonically ordered, a Pydantic schema,
and a `/health` payload.
""",
    """
import numpy as np, pandas as pd, joblib, os, tempfile
import lightgbm as lgb
from pydantic import create_model, ValidationError
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=300, n_features=4, noise=10, random_state=0)
FEATS = [f'f{i}' for i in range(4)]
X = pd.DataFrame(X, columns=FEATS)

# Train a tiny p10/p50/p90 bundle and pickle it.
qs = {a: lgb.LGBMRegressor(objective='quantile', alpha=a,
                            n_estimators=100, verbosity=-1, random_state=0).fit(X, y)
      for a in [0.1, 0.5, 0.9]}
bundle_path = os.path.join(tempfile.gettempdir(), 'demo_quantile_bundle.joblib')
joblib.dump({'features': FEATS, **qs, 'trained_through': '2024-01-01'}, bundle_path)

# 1) Stateless predict_one returning {p10, p50, p90}.
#    Quantile models trained independently can sometimes cross (p10 > p50 etc).
#    Sort the per-row predictions to enforce ordering — the standard fix in production.
def predict_one(row: dict, path: str = bundle_path) -> dict:
    b = joblib.load(path)
    df = pd.DataFrame([row])[b['features']]
    sorted_preds = sorted(float(b[a].predict(df)[0]) for a in [0.1, 0.5, 0.9])
    return {'p10': sorted_preds[0], 'p50': sorted_preds[1], 'p90': sorted_preds[2]}

sample = X.iloc[0].to_dict()
print('predict_one:', {k: round(v, 2) for k, v in predict_one(sample).items()})

# 2) Property test: quantiles never cross (post-sort, this is invariant).
def test_quantile_order(n=20):
    rng = np.random.default_rng(0)
    for _ in range(n):
        row = X.iloc[rng.integers(0, len(X))].to_dict()
        out = predict_one(row)
        assert out['p10'] <= out['p50'] <= out['p90'], out
test_quantile_order(); print('quantile-order test passed')

# 3) Pydantic input schema.
ReqSchema = create_model('ReqSchema', **{f: (float, ...) for f in FEATS})
ReqSchema(**sample)
try: ReqSchema(**{**sample, 'f0': 'oops'})
except ValidationError: print('pydantic rejects bad type — OK')

# 4) /health payload.
def health(path=bundle_path):
    b = joblib.load(path)
    return {'status': 'ok', 'trained_through': b['trained_through'],
            'n_features': len(b['features']), 'quantiles': sorted([k for k in b if isinstance(k, float)])}
print('health:', health())
""",
)


# ======================================================================
# time-series/time_series.ipynb
# ======================================================================

_add(
    "### Exercises — Data Loading",
    """
**Worked example — gap finding and continuity assertions**

Demonstrates the four idioms: dropping duplicate timestamps, finding the longest gap as
a `Timedelta`, asserting a continuous hourly grid, and printing a one-line dataset
summary.
""",
    """
import numpy as np, pandas as pd

ts = pd.to_datetime([
    '2024-01-01 00:00', '2024-01-01 01:00',
    '2024-01-01 01:00',                       # duplicate
    '2024-01-01 04:00',                       # 3-hour gap before this
    '2024-01-01 05:00',
], utc=True)
raw = pd.DataFrame({'ts': ts, 'close': [100, 101, 101, 104, 105]})

# 1) Drop duplicates first.
clean = raw.drop_duplicates(subset='ts').sort_values('ts').reset_index(drop=True)
print(f'after dedup: {len(clean)} rows')

# 2) Longest raw gap.
gaps = clean['ts'].diff()
i = int(gaps.idxmax())
print(f'longest gap: {gaps.iloc[i]} between {clean["ts"].iloc[i-1]} and {clean["ts"].iloc[i]}')

# 3) Reindex onto an expected hourly grid; assert it now matches.
expected = pd.date_range(clean['ts'].min(), clean['ts'].max(), freq='1h', tz='UTC')
df = clean.set_index('ts').reindex(expected)
assert df.index.equals(expected), 'still missing'
print(f'continuous index: {len(df)} hourly rows')

# 4) One-line summary.
n_hours = len(df); n_days = n_hours / 24
print(f'\\nstart {df.index.min()}  end {df.index.max()}  hours {n_hours}  days {n_days:.1f}')
""",
)

_add(
    "### Exercises — EDA",
    """
**Worked example — rolling vol, Hurst, intraday boxplot, manual ACF**

Four self-contained EDA primitives. The Hurst R/S estimator is the only one that's
unfamiliar; read its loop carefully — the exercise asks you to write the same.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=24*60, freq='1h', tz='UTC')
log_ret = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)

# 1) 7-day rolling vol.
vol_7d = log_ret.rolling(24*7).std()
print(f'rolling 7d vol — last: {vol_7d.iloc[-1]:.4f}')

# 2) Hurst (R/S) — H~0.5 random walk, >0.5 persistent, <0.5 anti-persistent.
def hurst_rs(x, n_chunks=5):
    x = np.asarray(x); chunk = len(x) // n_chunks
    rs = []
    for i in range(n_chunks):
        seg = x[i*chunk:(i+1)*chunk]
        seg = seg - seg.mean()
        z = np.cumsum(seg)
        rs.append((z.max() - z.min()) / seg.std())
    return float(np.log(np.mean(rs)) / np.log(chunk))

print(f'Hurst exponent: {hurst_rs(log_ret.values):.3f}')

# 3) Hour-of-day boxplot.
tmp = log_ret.to_frame('r'); tmp['hour'] = tmp.index.hour
fig, ax = plt.subplots(figsize=(8, 2.8))
sns.boxplot(data=tmp, x='hour', y='r', showfliers=False, ax=ax)
plt.tight_layout(); plt.show()

# 4) Manual ACF at lags 1, 24, 168 — np.corrcoef on overlapping slices.
r = log_ret.dropna().values
for k in [1, 24, 168]:
    print(f'lag {k:>3}: {np.corrcoef(r[:-k], r[k:])[0, 1]:+.4f}')
""",
)

_add(
    "### Exercises — Stationarity",
    """
**Worked example — ADF, KPSS, and rolling-window stationarity**

Demonstrates running ADF and KPSS together, interpreting the four reject/not-reject
combinations, applying first-differences to a non-stationary series, and rolling ADF
on a window for time-varying stationarity.
""",
    """
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings; warnings.filterwarnings('ignore')

rng = np.random.default_rng(0)
n = 1000
# Random walk (non-stationary in level, stationary in first-diff).
walk = pd.Series(np.cumsum(rng.normal(0, 1, n)))
ret  = walk.diff().dropna()

def adf_p(s):  return float(adfuller(s)[1])
def kpss_p(s): return float(kpss(s, regression='c', nlags='auto')[1])

# 1) ADF + KPSS on the level.
print(f'walk  ADF p={adf_p(walk):.3f}   KPSS p={kpss_p(walk):.3f}')
print('   ↑ ADF doesn\\'t reject (random walk has unit root); KPSS rejects (non-stationary)\\n')

# 2) First difference is stationary.
print(f'diff  ADF p={adf_p(ret):.5f}    KPSS p={kpss_p(ret):.3f}')
print('   ↑ ADF rejects strongly; KPSS doesn\\'t reject — stationary')

# 3) Rolling ADF p-value over a 200-row window stepping 50.
win, step = 200, 50
rolling = [(i, adf_p(ret.iloc[i:i+win])) for i in range(0, len(ret) - win, step)]
print(f'\\nrolling ADF p-values, first 4: {[round(p, 3) for _, p in rolling[:4]]}')

# 4) Log-squared returns proxies log-vol.
log_sq = np.log(ret ** 2 + 1e-12)
print(f'log(ret^2)  ADF p={adf_p(log_sq):.3f} (vol stationarity check)')
""",
)

_add(
    "### Exercises — Decomposition",
    """
**Worked example — STL and strength-of-seasonality**

A 24-period STL decomposition on a synthetic series with known intraday seasonality,
then computing the F_s strength metric to quantify how much of the variance is seasonal.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=24*30, freq='1h', tz='UTC')
seasonal = 5 * np.sin(2 * np.pi * idx.hour / 24)
trend = np.linspace(100, 110, len(idx))
noise = rng.normal(0, 1, len(idx))
y = pd.Series(trend + seasonal + noise, index=idx)

# 1) STL with period=24.
stl = STL(y, period=24, robust=True).fit()
fig, axes = plt.subplots(3, 1, figsize=(9, 4), sharex=True)
axes[0].plot(stl.trend);    axes[0].set_title('trend')
axes[1].plot(stl.seasonal); axes[1].set_title('seasonal (24h)')
axes[2].plot(stl.resid);    axes[2].set_title('residual')
plt.tight_layout(); plt.show()

# 2) Strength-of-seasonality F_s.
S, R = stl.seasonal.dropna(), stl.resid.dropna()
common = S.index.intersection(R.index)
Fs = max(0.0, 1.0 - R[common].var() / (S[common] + R[common]).var())
print(f'F_s strength of seasonality: {Fs:.3f}   (1 = strong, 0 = none)')

# 3) Compare against classical decomposition (additive).
dec = seasonal_decompose(y, model='additive', period=24)
print(f'classical decomp seasonal std: {dec.seasonal.std():.3f}')
print(f'classical decomp residual std: {dec.resid.std():.3f}')
""",
)

_add(
    "### Exercises — Split",
    """
**Worked example — chronological splits and TimeSeriesSplit**

A walk-forward CV on an expanding window using `TimeSeriesSplit`, an explicit
disjointness assertion, a check for distribution shift between train and test, and the
shuffled-split antipattern as a contrast.
""",
    """
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit

rng = np.random.default_rng(0)
n = 500
y_train_dist = rng.normal(0, 1, int(n * 0.7))
y_test_dist  = rng.normal(0.5, 1, n - int(n * 0.7))   # distribution shift!
y = pd.Series(np.r_[y_train_dist, y_test_dist])

# 1) TimeSeriesSplit folds.
tscv = TimeSeriesSplit(n_splits=4)
print('TimeSeriesSplit folds:')
for i, (tr, va) in enumerate(tscv.split(y)):
    print(f'  fold {i}: train [0..{tr[-1]}]  val [{va[0]}..{va[-1]}]')

# 2) Train/test temporal disjointness assertion.
cut = int(n * 0.7)
train, test = y.index[:cut], y.index[cut:]
assert train.max() < test.min(), 'temporal overlap!'
print('\\ntrain/test disjoint OK')

# 3) Distribution-shift check between train and test.
print('\\ntrain/test summary:')
print(pd.DataFrame({'train': y[train].agg(['mean', 'std']),
                    'test':  y[test].agg(['mean', 'std'])}).round(3))

# 4) Shuffled-split antipattern: train sees rows from the post-shift era.
shuf_test = rng.choice(n, size=len(test), replace=False)
shuf_train = np.setdiff1d(np.arange(n), shuf_test)
print(f'\\nshuffled split mean(train) = {y.iloc[shuf_train].mean():+.3f}, '
      f'mean(test) = {y.iloc[shuf_test].mean():+.3f}  ← shift hidden')
print(f'chronological mean(train) = {y[train].mean():+.3f}, '
      f'mean(test) = {y[test].mean():+.3f}  ← shift visible')
""",
)

_add(
    "### Exercises — Naive Baselines",
    """
**Worked example — seasonal naive, blending, directional accuracy, bootstrap**

Each exercise asks you to implement a different baseline-flavoured idiom; this cell
walks through all four on a tiny series so you can see the moving parts.
""",
    """
import numpy as np, pandas as pd
import math

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=24*14, freq='1h', tz='UTC')
y = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
test_idx = idx[-24:]; y_test = y.loc[test_idx].values

# 1) Seasonal naive at period=24 (manual lookup).
def seasonal_naive(history, target_idx, period):
    return np.array([history.get(t - pd.Timedelta(hours=period), np.nan) for t in target_idx])

pred_sn24 = seasonal_naive(y, test_idx, 24)
print(f'seasonal-naive 24h: {pred_sn24[:3].round(5)} ...')

# 2) Blend two baselines, nan-safe.
pred_zero = np.zeros_like(y_test)
blend = 0.5 * pred_zero + 0.5 * np.nan_to_num(pred_sn24, nan=0.0)
print(f'blend mean abs error: {np.mean(np.abs(y_test - blend)):.5f}')

# 3) Directional accuracy (manual).
mask = ~np.isnan(pred_sn24) & (y_test != 0)
hit = (np.sign(pred_sn24[mask]) == np.sign(y_test[mask])).mean()
print(f'directional acc on seasonal-naive: {hit:.3f}')

# 4) Bootstrap 95% CI on naive-zero RMSE.
resid = y_test - pred_zero
boot = []
for _ in range(500):
    idx_b = rng.integers(0, len(resid), len(resid))
    boot.append(math.sqrt(np.mean(resid[idx_b] ** 2)))
print(f'naive-zero RMSE 95% CI: [{np.quantile(boot, 0.025):.5f}, {np.quantile(boot, 0.975):.5f}]')
""",
)

_add(
    "### Exercises — ETS",
    """
**Worked example — ExponentialSmoothing in 4 modes**

Holt-Winters with seasonality, a damped-trend ETS, the smoothing-alpha parameter, and
the in-sample fitted values plot. Each exercise picks one of these threads to extend.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=24*30, freq='1h', tz='UTC')
trend = np.linspace(100, 105, len(idx))
seasonal = 2 * np.sin(2 * np.pi * idx.hour / 24)
noise = rng.normal(0, 0.5, len(idx))
y = pd.Series(trend + seasonal + noise, index=idx)
hist, true_next = y.iloc[:-24], y.iloc[-24:]

# 1) Holt-Winters with seasonal_periods=24.
hw = ExponentialSmoothing(hist, trend='add', seasonal='add', seasonal_periods=24).fit()
fc = hw.forecast(24)
print(f'HW forecast MAE on next 24h: {np.mean(np.abs(true_next.values - fc.values)):.3f}')

# 2) Damped trend.
damped = ExponentialSmoothing(hist, trend='add', damped_trend=True,
                               initialization_method='estimated').fit()
print(f'damped trend AIC: {damped.aic:.1f}')
print(f'undamped     AIC: {ExponentialSmoothing(hist, trend="add", initialization_method="estimated").fit().aic:.1f}')

# 3) Smoothing alpha — α≈1 = trust last observation only; α≈0 = heavy smoothing.
ses = ExponentialSmoothing(hist, trend=None, initialization_method='estimated').fit()
print(f'\\nSES alpha: {ses.params["smoothing_level"]:.3f}')

# 4) In-sample fit on the last 100 hours.
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.plot(hist.iloc[-100:].values, label='actual')
ax.plot(ses.fittedvalues[-100:].values, label='SES fitted')
ax.legend(); plt.tight_layout(); plt.show()
""",
)

_add(
    "### Exercises — SARIMA",
    """
**Worked example — SARIMA, residual diagnostics, manual AR(1)**

A small SARIMAX fit, the standard residual ACF + Ljung-Box pair, and a manual AR(1)
fit via OLS. The exercises generalise each of these.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

rng = np.random.default_rng(0)
n = 400
phi = 0.4
y = np.zeros(n)
for t in range(1, n):
    y[t] = phi * y[t-1] + rng.normal(0, 1)
y = pd.Series(y)

# 1) SARIMA(1,0,0) fit.
m = SARIMAX(y, order=(1, 0, 0)).fit(disp=False)
print(f'SARIMA AR(1) coef: {m.params[1]:.3f}    AIC: {m.aic:.1f}')

# 2) Residual ACF + Ljung-Box.
resid = pd.Series(m.resid).dropna()
fig, ax = plt.subplots(figsize=(7, 2.5))
plot_acf(resid, lags=24, ax=ax); plt.tight_layout(); plt.show()
print('\\nLjung-Box (lag 12):'); print(acorr_ljungbox(resid, lags=[12], return_df=True).round(4))

# 3) Manual AR(1) via OLS — should match SARIMA's coef closely.
y_lag = y.shift(1).dropna(); y_now = y.iloc[1:]
X = np.column_stack([np.ones(len(y_lag)), y_lag.values])
beta, *_ = np.linalg.lstsq(X, y_now.values, rcond=None)
print(f'manual AR(1) intercept={beta[0]:.3f}  phi={beta[1]:.3f}  (target {phi})')

# 4) In-sample fitted values vs actuals on last 80 obs.
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(y.iloc[-80:].values, label='actual')
ax.plot(m.fittedvalues[-80:].values, label='SARIMA fitted')
ax.legend(); plt.tight_layout(); plt.show()
""",
)

_add(
    "### Exercises — Lag Engineering",
    """
**Worked example — lags, leakage check, interactions, Fourier**

Cross-asset lag features, a leakage spot-check that corrupts the future and verifies
past unchanged, an interaction term, and Fourier features at K=2 harmonics.
""",
    """
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
idx = pd.date_range('2024-01-01', periods=24*30, freq='1h', tz='UTC')
y = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name='y')

X = pd.DataFrame(index=idx)

# 1) Lag features — always shift first.
X['y_lag1']  = y.shift(1)
X['y_lag24'] = y.shift(24)
print('lag features tail:'); print(X.tail(3).round(5))

# 2) Leakage check: corrupt y at t=last, rebuild lags, assert past unchanged.
y_corrupt = y.copy(); y_corrupt.iloc[-1] = 99.0
X_corrupt = pd.DataFrame({'y_lag1': y_corrupt.shift(1), 'y_lag24': y_corrupt.shift(24)}, index=idx)
assert X.iloc[:-1].fillna(0).equals(X_corrupt.iloc[:-1].fillna(0)), 'LEAK!'
print('\\nleakage spot-check OK')

# 3) Interaction feature.
X['rstd_24'] = y.rolling(24).std()
X['lag1_x_rstd'] = X['y_lag1'] * X['rstd_24']
print(f'\\ncorr(interaction, y) = {X["lag1_x_rstd"].corr(y):+.4f}')

# 4) Fourier seasonality at period=24, K=2 harmonics.
h = idx.hour.values
for k in range(1, 3):
    X[f'sin_{k}'] = np.sin(2 * np.pi * k * h / 24)
    X[f'cos_{k}'] = np.cos(2 * np.pi * k * h / 24)
print('\\nFourier features (head):'); print(X[['sin_1', 'cos_1', 'sin_2', 'cos_2']].head(3).round(3))
""",
)

_add(
    "### Exercises — ML + Optuna",
    """
**Worked example — Optuna for time-series ML**

LightGBM tuned with a MedianPruner objective, the same Optuna primitives as the
classification chapter, plus a comparison of LightGBM vs XGBoost at equal trial budget.
""",
    """
import optuna, lightgbm as lgb, xgboost as xgb, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression
optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = make_regression(n_samples=400, n_features=4, noise=5, random_state=0)

def make_obj(model_cls, name):
    def obj(trial):
        params = dict(learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True),
                       n_estimators=200, random_state=0)
        if name == 'lgb':   params.update(verbosity=-1, num_leaves=trial.suggest_int('leaves', 15, 127))
        if name == 'xgb':   params.update(verbosity=0,  max_depth=trial.suggest_int('depth', 3, 9),
                                          objective='reg:absoluteerror')
        scores = []
        for k, (tr, va) in enumerate(TimeSeriesSplit(3).split(X)):
            m = model_cls(**params).fit(X[tr], y[tr])
            scores.append(mean_absolute_error(y[va], m.predict(X[va])))
            trial.report(np.mean(scores), step=k)
            if trial.should_prune(): raise optuna.TrialPruned()
        return float(np.mean(scores))
    return obj

# 1) LightGBM study with pruning.
study_lgb = optuna.create_study(direction='minimize',
                                 pruner=optuna.pruners.MedianPruner(n_startup_trials=2))
study_lgb.optimize(make_obj(lgb.LGBMRegressor, 'lgb'), n_trials=6, show_progress_bar=False)
print(f'LGBM best MAE: {study_lgb.best_value:.4f}    params: {study_lgb.best_params}')

# 2) XGBoost at the same budget.
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(make_obj(xgb.XGBRegressor, 'xgb'), n_trials=6, show_progress_bar=False)
print(f'XGB  best MAE: {study_xgb.best_value:.4f}    params: {study_xgb.best_params}')
""",
)

_add(
    "### Exercises — Recursive Multi-Step",
    """
**Worked example — DIRECT vs recursive multi-step**

Two ways to forecast h steps ahead: train one model per horizon (DIRECT) or train one
model and feed its predictions back as features (recursive). Plot RMSE vs horizon to
see which approach degrades more gracefully.
""",
    """
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

rng = np.random.default_rng(0)
n = 600
y = pd.Series(np.cumsum(rng.normal(0, 0.5, n)) + np.sin(np.arange(n) / 24))
X_full = pd.DataFrame({'lag_1': y.shift(1), 'lag_24': y.shift(24)}).dropna()
y_full = y.loc[X_full.index]
cut = int(len(X_full) * 0.7)
X_tr, X_te = X_full.iloc[:cut], X_full.iloc[cut:]
y_tr, y_te = y_full.iloc[:cut], y_full.iloc[cut:]

# 1) DIRECT: one model per horizon.
horizons = [1, 6, 12, 24]
direct_rmse = {}
for h in horizons:
    target = y_full.shift(-h).dropna()
    Xh = X_full.loc[target.index]
    cut_h = int(len(Xh) * 0.7)
    m = lgb.LGBMRegressor(n_estimators=100, verbosity=-1).fit(Xh.iloc[:cut_h], target.iloc[:cut_h])
    pred = m.predict(Xh.iloc[cut_h:])
    direct_rmse[h] = math.sqrt(mean_squared_error(target.iloc[cut_h:], pred))
print('DIRECT RMSE per horizon:', {k: round(v, 3) for k, v in direct_rmse.items()})

# 2) Recursive: one model, feed predictions back as features.
m_one = lgb.LGBMRegressor(n_estimators=100, verbosity=-1).fit(X_tr, y_tr)
def recursive_forecast(model, X_start, h):
    X_cur = X_start.copy(); preds = []
    for _ in range(h):
        p = model.predict(X_cur)
        preds.append(p[0])
        # shift features by one — naive: re-use the predicted lag_1.
        X_cur = X_cur.copy(); X_cur['lag_1'] = p[0]
    return preds

# Demo on 50 starting points; collect h-step errors.
errs = np.zeros((50, 24))
for i in range(50):
    start = X_te.iloc[[i]]
    truth = y_te.iloc[i:i+24].values
    pred = np.array(recursive_forecast(m_one, start, 24))[:len(truth)]
    errs[i, :len(truth)] = (truth - pred) ** 2

rec_rmse = np.sqrt(errs.mean(axis=0))
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(np.arange(1, 25), rec_rmse, marker='o', label='recursive')
for h, v in direct_rmse.items(): ax.plot(h, v, 'rx', markersize=10)
ax.set_xlabel('horizon (h)'); ax.set_ylabel('RMSE'); ax.legend(['recursive', 'DIRECT'])
plt.tight_layout(); plt.show()
""",
)

_add(
    "### Exercises — Feature Importance",
    """
**Worked example — interpret a time-series model**

Permutation importance, SHAP dependence, top-K refit comparison, and a single-row
waterfall — same toolkit as the classification chapter, applied to a regression model.
""",
    """
import numpy as np, pandas as pd, lightgbm as lgb, shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import math

X, y = make_regression(n_samples=400, n_features=6, n_informative=3, random_state=0)
X = pd.DataFrame(X, columns=[f'f{i}' for i in range(6)])
X_tr, X_va = X[:300], X[300:]
y_tr, y_va = y[:300], y[300:]
m = lgb.LGBMRegressor(n_estimators=200, verbosity=-1, random_state=0).fit(X_tr, y_tr)

# 1) Permutation importance.
pi = permutation_importance(m, X_va, y_va, n_repeats=5, random_state=0)
perm = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
print('permutation importance:'); print(perm.round(3))

# 2) SHAP dependence plot for top feature.
explainer = shap.TreeExplainer(m)
sv = explainer.shap_values(X_va)
shap.dependence_plot(perm.index[0], sv, X_va, show=True)

# 3) Drop bottom 50% of features by importance, refit, compare RMSE.
keep = perm.head(len(perm) // 2).index.tolist()
m_small = lgb.LGBMRegressor(n_estimators=200, verbosity=-1, random_state=0).fit(X_tr[keep], y_tr)
print(f'\\nfull-feature  RMSE: {math.sqrt(mean_squared_error(y_va, m.predict(X_va))):.3f}')
print(f'top-{len(keep)} feature RMSE: {math.sqrt(mean_squared_error(y_va, m_small.predict(X_va[keep]))):.3f}')

# 4) Single-row waterfall.
i = int(np.argmax(m.predict(X_va)))
exp = explainer(X_va.iloc[[i]])
shap.plots.waterfall(exp[0], max_display=6, show=True)
""",
)

_add(
    "### Exercises — Comparison",
    """
**Worked example — Diebold-Mariano, MASE, and cumulative-error plots**

Statistical model-comparison primitives: a DM test for whether two forecast errors
differ significantly, MASE which scales by the in-sample naive MAE, and a cumulative
absolute-error plot to visualise where the gap accrues.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import acovf
import math

rng = np.random.default_rng(0)
n = 400
y_train = rng.normal(0, 1, 200)
y_test  = rng.normal(0, 1, n)
pred_a  = y_test + rng.normal(0, 0.5, n)   # better
pred_b  = y_test + rng.normal(0, 1.0, n)   # worse

# 1) Diebold-Mariano test (squared-error loss, h=1).
e_a = (y_test - pred_a) ** 2
e_b = (y_test - pred_b) ** 2
d = e_a - e_b
gamma = acovf(d, nlag=0)[0]
DM = np.sqrt(n) * d.mean() / np.sqrt(gamma)
print(f'DM stat: {DM:+.3f}  (negative ⇒ A has lower loss)')

# 2) MASE for predictor A.
naive_in_mae = np.mean(np.abs(np.diff(y_train)))
mase_a = mean_absolute_error(y_test, pred_a) / naive_in_mae
print(f'MASE(A): {mase_a:.3f}   (<1 = better than naive)')

# 3) Cumulative |error| comparison.
cum_a = np.cumsum(np.abs(y_test - pred_a))
cum_b = np.cumsum(np.abs(y_test - pred_b))
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.plot(cum_a, label='|err| A'); ax.plot(cum_b, label='|err| B')
ax.legend(); ax.set_xlabel('test step'); plt.tight_layout(); plt.show()
""",
)

_add(
    "### Exercises — Diagnostics",
    """
**Worked example — residual diagnostics for a forecasting model**

Rolling RMSE, a Q-Q plot vs Normal, Ljung-Box for residual autocorrelation, and the
actual-vs-predicted scatter with a y=x reference line.
""",
    """
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

rng = np.random.default_rng(0)
n = 500
y_true = rng.normal(0, 1, n)
y_pred = y_true + rng.normal(0, 0.4, n) + rng.standard_t(df=4, size=n) * 0.1
resid = y_true - y_pred

# 1) Rolling 24-row RMSE.
roll = pd.Series(resid ** 2).rolling(24).mean().pow(0.5)
fig, ax = plt.subplots(figsize=(8, 2.5))
roll.plot(ax=ax); ax.set_title('rolling 24h RMSE'); plt.tight_layout(); plt.show()

# 2) Q-Q plot vs Normal.
fig, ax = plt.subplots(figsize=(4, 4))
stats.probplot(resid, dist='norm', plot=ax); plt.tight_layout(); plt.show()

# 3) Ljung-Box on residuals.
print('Ljung-Box:')
print(acorr_ljungbox(resid, lags=[1, 24], return_df=True).round(4))

# 4) Actual vs predicted with y=x.
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(y_pred, y_true, s=4, alpha=0.4)
lims = [min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())]
ax.plot(lims, lims, 'r--')
ax.set_xlabel('pred'); ax.set_ylabel('true'); plt.tight_layout(); plt.show()
""",
)

_add(
    "### Exercises — Probabilistic",
    """
**Worked example — quantile forecasts and pinball loss**

Quantile LightGBM models, empirical coverage of an interval, the pinball / quantile
loss formula, and how to extend the band to p05/p95.
""",
    """
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=4, noise=10, random_state=0)
X_tr, X_te = X[:350], X[350:]
y_tr, y_te = y[:350], y[350:]

def fit_q(a):
    return lgb.LGBMRegressor(objective='quantile', alpha=a,
                              n_estimators=200, verbosity=-1, random_state=0).fit(X_tr, y_tr)

# 1) Coverage of [p10, p90].
m_lo, m_hi = fit_q(0.1), fit_q(0.9)
p10, p90 = m_lo.predict(X_te), m_hi.predict(X_te)
print(f'coverage [p10, p90]: {((y_te >= p10) & (y_te <= p90)).mean()*100:.1f}%   (target 80%)')

# 2) Pinball loss.
def qloss(yt, yp, alpha):
    e = yt - yp
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))
print(f'pinball(p10): {qloss(y_te, p10, 0.10):.3f}')
print(f'pinball(p90): {qloss(y_te, p90, 0.90):.3f}')

# 3) Width plot.
width = p90 - p10
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(width); ax.set_title('p90 - p10 width'); plt.tight_layout(); plt.show()

# 4) Wider band (p05/p95) — coverage should be ~90%.
p05, p95 = fit_q(0.05).predict(X_te), fit_q(0.95).predict(X_te)
print(f'\\ncoverage [p05, p95]: {((y_te >= p05) & (y_te <= p95)).mean()*100:.1f}%   (target 90%)')
""",
)

_add(
    "### Exercises — Deployment",
    """
**Worked example — package a forecasting bundle**

Recursive `forecast_h` helper, a parametrised pytest assertion that the output is
exactly h rows, a Pydantic schema with `Field` constraints, and a `/health` payload —
the deployment shape mirrors the regression chapter exactly.
""",
    """
import numpy as np, pandas as pd, joblib, os, tempfile
import lightgbm as lgb
from pydantic import BaseModel, Field
from typing import List
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=300, n_features=3, noise=5, random_state=0)
FEATS = [f'f{i}' for i in range(3)]
m = lgb.LGBMRegressor(n_estimators=100, verbosity=-1, random_state=0).fit(X, y)
bundle_path = os.path.join(tempfile.gettempdir(), 'demo_ts_bundle.joblib')
joblib.dump({'features': FEATS, 'model': m, 'trained_through': '2024-01-01'}, bundle_path)

# 1) Recursive forecast_h: returns a DataFrame of (ts, p50) for h steps.
def forecast_h(h: int, model, history_idx_max) -> pd.DataFrame:
    start = history_idx_max + pd.Timedelta(hours=1)
    ts = pd.date_range(start, periods=h, freq='1h', tz='UTC')
    # naive: predict on a row of zeros (toy demo). Real version uses recursive lags.
    preds = model.predict(np.zeros((h, len(FEATS))))
    return pd.DataFrame({'ts': ts, 'p50': preds})

last = pd.Timestamp('2024-01-15 23:00', tz='UTC')
print('forecast 6 steps:'); print(forecast_h(6, m, last))

# 2) Property test: output length == h for any h.
def test_forecast_length():
    for h in [1, 6, 24, 168]:
        out = forecast_h(h, m, last)
        assert len(out) == h, (h, len(out))
test_forecast_length(); print('\\nlength test passed')

# 3) Pydantic request schema with constraints.
class ForecastRequest(BaseModel):
    h: int = Field(default=24, ge=1, le=168)
class ForecastResponse(BaseModel):
    ts: List[str]
    p50: List[float]

req = ForecastRequest(h=12); print(f'\\nvalid request: h={req.h}')
try: ForecastRequest(h=200)
except Exception as e: print('rejected h=200 — OK')

# 4) /health payload.
def health(path=bundle_path):
    b = joblib.load(path)
    return {'status': 'ok', 'trained_through': b['trained_through'], 'n_features': len(b['features'])}
print('health:', health())
""",
)
