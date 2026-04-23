"""Hand-written 'Before you start' hint blocks, keyed by section header.

Each entry is a short idiom bullet list + one micro-example, designed to cue the
techniques the section's exercises use without giving away the solution.

Keyed by the exact text of the `### Exercises ...` markdown cell's first line.
"""

HINTS: dict[str, str] = {}


def _add(header: str, body: str) -> None:
    HINTS[header] = body.strip() + "\n"


# ======================================================================
# classification/classification.ipynb
# ======================================================================

_add("### Exercises — Section 1", r"""
**Before you start — techniques you'll use:**

- **`groupby(col).apply(func)`** for per-group calculations. On a datetime series,
  `.diff()` returns `Timedelta`; convert with `.dt.total_seconds()` then divide by
  3600 / 86400 for hours / days.
- **`df.isna().mean()`** → fraction missing per column. Multiply by 100 for percent;
  `.sort_values(ascending=False)` ranks them.
- **`df.pivot(index=, columns=, values=)`** turns long → wide. The `columns` entries
  become one column per unique value.
- **`pd.date_range(start, end, freq='1h', tz='UTC')`** builds a regular expected
  index. Use `series.reindex(expected)` to expose gaps as `NaN`.

*Mini-example (longest per-group gap):*
```python
# largest daily gap between login events per user
gaps = (events.groupby('user_id')['ts']
              .apply(lambda s: s.diff().dt.total_seconds().div(86400).max()))
```
""")

_add("### Exercises — Section 2", r"""
**Before you start — techniques you'll use:**

- **Rolling statistics**: `series.rolling(window).corr(other)` / `.std()` / `.mean()`.
  Window is in rows, so "7 days hourly" = `24*7 = 168`.
- **Higher moments**: `scipy.stats.skew(x)` and `scipy.stats.kurtosis(x, fisher=True)`
  (fisher=True returns **excess** kurtosis — subtracts 3 from Pearson kurtosis).
- **Time-of-day grouping**: `series.index.hour` yields 0–23 ints. Feed them straight
  into `sns.boxplot(x=hour, y=values)` to surface intraday seasonality.
- **Threshold lines**: `plt.axhline(y, linestyle='--', color=..., alpha=...)` — use
  them to mark a reference (e.g. class-balance 0.5) on top of rolling plots.

*Mini-example (rolling correlation):*
```python
# 7-day rolling corr between two return series, sampled hourly
rolling_corr = returns['BTC'].rolling(168).corr(returns['ETH'])
```
""")

_add("### Exercises — Section 3", r"""
**Before you start — techniques you'll use:**

- **Recursive/Wilder smoothing** for RSI: iterate with `for i in range(period, n):`,
  updating `avg_gain = (prev * (period-1) + gain[i]) / period`. Use `np.where(diff>0,
  diff, 0)` to split gains/losses.
- **Dict-comprehensions building DataFrames**: `pd.DataFrame({f'{sym.lower()}_24h':
  np.log(wide[sym] / wide[sym].shift(24)) for sym in symbols})`.
- **Leakage spot-checks**: corrupt `close` on rows **after** a cutoff `t`, re-run
  feature generation, then assert features at rows `<= t` are unchanged (`.equals()`).
- **Z-score regime features**: `(x - x.rolling(w).mean()) / x.rolling(w).std()` —
  use a multiple of the natural day to capture weekly cycles (e.g. `w=168`).

*Mini-example (z-score regime):*
```python
vol = features['btc_vol_24h']
vol_z = ((vol - vol.rolling(168).mean()) / vol.rolling(168).std()).rename('vol_z')
```
""")

_add("### Exercises — Section 4", r"""
**Before you start — techniques you'll use:**

- **Walk-forward splits** = expanding train, fixed-size val. Pattern:
  `fold_size = (n - min_train) // n_splits`; yield `np.arange(0, train_end),
  np.arange(train_end, val_end)` for each fold.
- **`set(idx)` for disjointness**: `set_a.isdisjoint(set_b)` is your O(n) overlap
  assert. Wrap in `assert` so CI fails loudly if a refactor reintroduces leakage.
- **Split-by-split class balance**: build rows of dicts then `pd.DataFrame(rows)`
  — cleaner than fighting `groupby.agg` for heterogeneous columns.
- **Shuffled-CV leakage demo**: build a toy autoregressive target, compare
  `cross_val_score(KFold(shuffle=True))` vs `TimeSeriesSplit()`; shuffled inflates.

*Mini-example (walk-forward splits):*
```python
def walk_forward(n, n_splits=5, min_train=2000):
    fold = (n - min_train) // n_splits
    for k in range(n_splits):
        tr_end = min_train + k * fold
        yield np.arange(0, tr_end), np.arange(tr_end, tr_end + fold)
```
""")

_add("### Exercises — Section 5", r"""
**Before you start — techniques you'll use:**

- **Majority baseline** without sklearn: `y_train.mode().iloc[0]` → broadcast with
  `np.full(len(y_val), majority)`; accuracy is `(pred == y_val.values).mean()`.
- **Directional baselines**: sign-of-last-return, or its opposite (mean-reversion).
  Convert booleans to floats via `.astype(float)` to feed `predict_proba`-style APIs.
- **Combined metrics table**: build rows as dicts `{'name': ..., 'acc': ..., 'auc':
  ...}`, then `pd.DataFrame(rows).round(4)`.
- **Bootstrap CI**: `rng = np.random.default_rng(seed)`, `idx = rng.integers(0, n,
  n)`, compute metric on `(y[idx], p[idx])`, repeat 1000×, take `np.quantile(scores,
  [0.025, 0.975])`.

*Mini-example (manual majority baseline):*
```python
maj = int(y_train.mode().iloc[0])
acc = (np.full(len(y_val), maj) == y_val.values).mean()
```
""")

_add("### Exercises — Section 6", r"""
**Before you start — techniques you'll use:**

- **Calibration**: wrap your fitted model in `CalibratedClassifierCV(base_estimator,
  method='isotonic', cv='prefit')` to map raw scores → well-calibrated probabilities.
- **Pipelines with feature expansion**: `Pipeline([('scaler', StandardScaler()),
  ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
  ('clf', LogisticRegression(...))])`.
- **Stacking**: `StackingClassifier(estimators=[...], final_estimator=...,
  cv=TimeSeriesSplit(...), passthrough=False)`. Use a time-aware CV for leakage-safe
  out-of-fold scores.
- **Ablations**: clone the best config, `.drop(columns=subset)`, refit, rescore.
  Ensures any delta is from the features, not the search randomness.

*Mini-example (polynomial pipeline):*
```python
poly = Pipeline([('sc', StandardScaler()),
                 ('poly', PolynomialFeatures(2, interaction_only=True, include_bias=False)),
                 ('lr', LogisticRegression(max_iter=2000))]).fit(X_train, y_train)
```
""")

_add("### Exercises — Section 7", r"""
**Before you start — techniques you'll use:**

- **Pruning in Optuna**: `study.optimize(obj, n_trials=...)` where the `objective`
  calls `trial.report(score, step=k)` inside CV folds and `raise
  optuna.TrialPruned()` if `trial.should_prune()`.
- **Widening the search space**: add `trial.suggest_int/float/categorical` calls with
  sensible ranges — keep `log=True` for learning-rate / L1 / L2 hyperparams.
- **Same-budget baseline**: `RandomizedSearchCV(model, param_distributions=..., n_iter=N,
  cv=TimeSeriesSplit(5), scoring='neg_log_loss')` — compare best-ever score at equal N.
- **Persisting studies**: `storage=f'sqlite:///{path}'` + `study_name='...'` on
  `create_study` lets you resume after a crash.

*Mini-example (pruning hook):*
```python
def obj(trial):
    for k, (tr, va) in enumerate(TimeSeriesSplit(5).split(X)):
        score = fit_and_score(trial, tr, va)
        trial.report(score, step=k)
        if trial.should_prune(): raise optuna.TrialPruned()
    return np.mean(scores)
```
""")

_add("### Exercises — Section 8", r"""
**Before you start — techniques you'll use:**

- **Permutation importance** (model-agnostic): `from sklearn.inspection import
  permutation_importance; permutation_importance(model, X_val, y_val, n_repeats=5,
  scoring='neg_log_loss', random_state=seed)`.
- **SHAP values** (tree models): `explainer = shap.TreeExplainer(model)`;
  `shap_values = explainer.shap_values(X_sample)` — for classification with two
  outputs, index class 1 or use the newer `shap.Explanation` API.
- **Top-K refit**: rank features by `np.abs(shap_values).mean(axis=0)`, keep top-K,
  refit with the same model config, compare val metrics to the full model.
- **Single-sample explanations**: `shap.plots.waterfall(explainer(X_row)[0])`.
  Always pass a DataFrame with the **same column order** the model was trained on.

*Mini-example (top-5 by SHAP):*
```python
mean_abs = np.abs(shap_values).mean(axis=0)
top5 = X_val.columns[np.argsort(mean_abs)[::-1][:5]].tolist()
```
""")

_add("### Exercises — Section 9", r"""
**Before you start — techniques you'll use:**

- **Threshold sweep**: `thresholds = np.linspace(0.3, 0.7, 81)`; compute a metric for
  `(proba_val >= t).astype(int)` across them, then `np.argmax` / `np.argmin`.
- **Cost-based thresholds**: define `profit(y_true, y_pred) = tp - fp + tn - fn`
  using `confusion_matrix(...).ravel()` → plug into the same sweep loop.
- **Precision/recall curves**: `from sklearn.metrics import precision_score,
  recall_score`. Use `zero_division=0` to silence the no-positive-predictions warning.
- **Bootstrap AUC**: same pattern as Section 5 bootstraps, with
  `roc_auc_score(y[idx], p[idx])` inside the loop. Report the 2.5 / 97.5 percentiles.

*Mini-example (F1-optimal threshold):*
```python
ts = np.linspace(0.30, 0.70, 81)
best_t = float(ts[np.argmax([f1_score(y_val, (p_val >= t).astype(int)) for t in ts])])
```
""")

_add("### Exercises — Section 10", r"""
**Before you start — techniques you'll use:**

- **Self-contained predict helper**: load artifact once inside the function, pull the
  expected feature order out of the bundle (`feats = art['feature_names']`), then
  `pd.DataFrame([row_dict])[feats]` — never rely on global state.
- **Unit tests**: write `def test_xxx():` with plain `assert` statements; sample a
  known-good row with `.iloc[0].to_dict()` and assert on shape + value ranges.
- **Pydantic validation**: `create_model('Name', **{f: (float, ...) for f in FEATS})`
  turns the feature list into a typed input schema; catches `ValidationError` for
  bad payloads.
- **Health checks**: return `{'status': 'ok', 'model_version': ..., 'trained_through':
  ..., 'n_features': ...}` — make the payload reflect the artifact, not constants.

*Mini-example (predict-one skeleton):*
```python
def predict_one(row: dict, path: str = MODEL_PATH, threshold=0.5) -> dict:
    art = joblib.load(path)
    X = pd.DataFrame([row])[art['feature_names']]
    prob = float(art['model'].predict_proba(X)[0, 1])
    return {'prob_up': prob, 'label': int(prob >= threshold)}
```
""")


# ======================================================================
# regression/regression.ipynb
# ======================================================================

_add("### Exercises (Section 2)", r"""
**Before you start — techniques you'll use:**

- **Per-group missing-rate**: `df.set_index('ts').groupby('symbol').apply(lambda g:
  g.isna().mean() * 100)` — one row per symbol, one column per field.
- **Gap finding**: sort `ts`, `diff()`, `idxmax()` for the position of the largest
  timedelta; pair with `.loc[[i-1, i]]` to show the bracketing timestamps.
- **Expected hourly grid**: `pd.date_range(min, max, freq='1h', tz='UTC')`, then
  `full.difference(btc.index)` surfaces the exact missing timestamps.
- **Floor to day for count plots**: `ts.dt.floor('D')` then
  `.groupby(...).size()` — one integer per day.

*Mini-example (longest gap location):*
```python
diffs = btc['ts'].sort_values().reset_index(drop=True).diff()
i = int(diffs.idxmax())
print('longest gap:', diffs.iloc[i], '@', btc['ts'].iloc[i])
```
""")

_add("### Exercises (Section 3)", r"""
**Before you start — techniques you'll use:**

- **Realized volatility**: `np.sqrt((log_ret ** 2).rolling(w).sum()) * np.sqrt(w)`
  annualises per-hour vol to match the sampling window.
- **Skew / excess kurt**: `scipy.stats.skew(r)`, `scipy.stats.kurtosis(r)` (Fisher
  convention — 0 means Normal).
- **Weekday seasonality**: `rv.index.day_name()` gives strings; pass an explicit
  `order=['Monday', ..., 'Sunday']` to `sns.boxplot` so the axis stays intuitive.
- **Lag scatter**: build `pd.DataFrame({'rv': rv, 'rv_lag24': rv.shift(24)}).dropna()`;
  persistence shows up as a positive slope.

*Mini-example (rolling RV):*
```python
rv_24h = np.sqrt((btc['log_ret'] ** 2).rolling(24).sum()) * np.sqrt(24)
```
""")

_add("### Exercises (Section 4)", r"""
**Before you start — techniques you'll use:**

- **Parkinson**: $\sigma^2 = \mathrm{rolling\_sum}(\log(H/L)^2) / (4 \log 2)$. Code
  as `(np.log(high/low) ** 2).rolling(w).sum() / (4*np.log(2))` then `np.sqrt()`.
- **Garman-Klass** uses both `log(H/L)` and `log(C/O)`: `0.5*(hl**2) -
  (2*np.log(2) - 1)*(co**2)`.
- **HAR lag structure**: include lag-1, 24-mean, 168-mean — all via `shift(1)`
  **before** any rolling window, so features at $t$ depend strictly on info up to
  $t-1$.
- **Cross-asset features**: compute RV on another symbol, `.reindex(target.index)`
  to align on the training index.

*Mini-example (Parkinson):*
```python
def parkinson(high, low, w):
    return np.sqrt((np.log(high/low)**2).rolling(w).sum() / (4*np.log(2)))
```
""")

_add("### Exercises (Section 5)", r"""
**Before you start — techniques you'll use:**

- **Expanding walk-forward**: `initial = n - n_splits * val_size`; each fold expands
  train by `val_size`, val slides forward by the same.
- **Temporal disjointness asserts**: `assert train.index.max() < val.index.min()`
  — faster than building sets and reads more clearly in a test.
- **Split-level target distribution**: list of dicts → `pd.DataFrame` — each row
  carries `{'split', 'mean', 'std', 'n'}`; easier than reshaping a groupby.
- **Purged CV**: after computing a fold's train range, **drop** training rows inside
  `[fold_end - horizon, fold_end]` so the model can't peek at future-aligned targets.

*Mini-example (expanding walk-forward skeleton):*
```python
def wf(n, n_splits, val):
    initial = n - n_splits * val
    for k in range(n_splits):
        tr_end = initial + k * val
        yield np.arange(0, tr_end), np.arange(tr_end, tr_end + val)
```
""")

_add("### Exercises (Section 6)", r"""
**Before you start — techniques you'll use:**

- **Persistence baseline**: predict the trailing 24h realised vol — it's a
  surprisingly tough benchmark for vol forecasting.
- **QLIKE loss** (scale-free, vol-friendly):
  `np.mean(yt/yp - np.log(yt/yp) - 1)` where `yt, yp` are variances (i.e. σ²).
- **Baseline comparison table**: stack predictions in a dict, compute MAE/RMSE/QLIKE
  for each, assemble into a DataFrame, `.sort_values('MAE')`.
- **Bootstrap a metric CI**: draw `n` with replacement from the residuals (`rng =
  default_rng`, `idx = rng.integers(0, n, n)`), recompute MAE, repeat 1000×.

*Mini-example (QLIKE):*
```python
def qlike(yt_vol, yp_vol, eps=1e-12):
    yt, yp = yt_vol**2 + eps, yp_vol**2 + eps
    return float(np.mean(yt/yp - np.log(yt/yp) - 1.0))
```
""")

_add("### Exercises (Section 7)", r"""
**Before you start — techniques you'll use:**

- **`Pipeline([('sc', StandardScaler()), ('m', Ridge(...))])`**: fold scaling into
  the model so CV never leaks val stats into train scaling.
- **Fit-time vs accuracy plot**: `ax.scatter(cv_fit_seconds, cv_mae)`, then
  `ax.annotate(name, (x, y))` for each model — the frontier is where you pick.
- **Permutation importance for RF**: `permutation_importance(rf, X_val, y_val,
  n_repeats=5, n_jobs=-1)` returns `importances_mean`; rank with a `pd.Series`.
- **Averaging ensemble**: predict from top-2 models, take `np.mean(np.column_stack([p1,
  p2]), axis=1)`; compare MAE to each component.

*Mini-example (scaled pipeline):*
```python
models = {'Ridge': Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))])}
```
""")

_add("### Exercises (Section 8)", r"""
**Before you start — techniques you'll use:**

- **Log-target regression**: fit on `np.log(y_train)`, invert predictions with
  `np.exp(pred)`. Clip pre-exp (`np.clip(pred, -15, 5)`) to avoid overflow on
  pathological test rows.
- **Comparing raw vs log** on the **same units**: evaluate RMSE in the original
  target's scale (`mean_squared_error(y_val, np.exp(pred_log))`).
- **Box-Cox** needs strictly positive y: `from scipy.stats import boxcox; y_bc, lam =
  boxcox(y)` → `inv = (y_bc * lam + 1) ** (1/lam)` to invert.
- **Residual vs fitted plots**: side-by-side for raw vs log target surfaces
  heteroscedasticity — look for a visible funnel shape shrinking.

*Mini-example (log-space predict helper):*
```python
def predict_in_log_space(model, X, lo=-15, hi=5):
    return np.exp(np.clip(model.predict(X), lo, hi))
```
""")

_add("### Exercises (Section 9)", r"""
**Before you start — techniques you'll use:**

- **Pruning = MedianPruner**: `optuna.create_study(pruner=optuna.pruners.MedianPruner(
  n_startup_trials=3))`; inside objective, call `trial.report(fold_score, step=k)`
  and `if trial.should_prune(): raise TrialPruned()`.
- **Extended search spaces**: `trial.suggest_int('num_leaves', 15, 127)`,
  `trial.suggest_int('min_child_samples', 5, 100)` — LGBM's regularisers.
- **SQLite-backed study**: `storage=f'sqlite:///{path}/study.db'` and a stable
  `study_name=` let you reload: `optuna.load_study(study_name=..., storage=...)`.
- **Multi-objective**: `optuna.create_study(directions=['minimize', 'minimize'])`;
  objective returns a **tuple** (e.g. CV MAE and fit time).

*Mini-example (pruning hook):*
```python
for k, (tr, va) in enumerate(TimeSeriesSplit(5).split(X)):
    fold_mae = ...
    trial.report(fold_mae, step=k)
    if trial.should_prune(): raise optuna.TrialPruned()
```
""")

_add("### Exercises (Section 10)", r"""
**Before you start — techniques you'll use:**

- **Permutation importance on val**: model-agnostic, expensive but honest.
  Combine with the model's built-in gain importance for a 2-column table.
- **SHAP dependence plot**: `shap.dependence_plot('feature_name', shap_values,
  sample, show=True)` — shows feature value vs its SHAP contribution, colored by
  interaction.
- **Feature-pruning ablation**: sort by gain, keep top-50%, refit with the SAME
  hyperparameters — any MAE change comes from the feature cut, not the model.
- **Waterfall for one row**: use the new SHAP API: `explainer(sample_row)` returns
  an `Explanation`; `shap.plots.waterfall(exp[0])` plots a single prediction.

*Mini-example (keep top-K by gain, refit):*
```python
keep = imp.sort_values('gain', ascending=False).head(len(FEATS)//2)['feature'].tolist()
m_small = lgb.LGBMRegressor(**best_params).fit(X_train[keep], y_train)
```
""")

_add("### Exercises (Section 11)", r"""
**Before you start — techniques you'll use:**

- **Ljung-Box on residuals**: `from statsmodels.stats.diagnostic import
  acorr_ljungbox; acorr_ljungbox(resid, lags=[12, 24], return_df=True)`. p<0.05
  means residuals still have autocorrelation.
- **Rolling residual std**: `pd.Series(resid, index=val.index).rolling(24).std()`
  exposes conditional heteroscedasticity — a classic missing-GARCH signal.
- **Reliability buckets**: `pd.qcut(pred, 5, labels=[f'Q{i}' for i in range(1,6)])`
  then `.groupby('bucket').apply(lambda g: pd.Series({'n': len(g), 'mean_true':
  g['true'].mean(), 'mean_pred': g['pred'].mean()}))`.
- **Q-Q-style calibration plot**: `np.quantile(pred, ts)` vs `np.quantile(y, ts)`
  on the **same** `ts = np.linspace(0, 1, 50)` — a 45° line means well-calibrated.

*Mini-example (bucket-level reliability):*
```python
df_q['bucket'] = pd.qcut(df_q['pred'], 5, labels=[f'Q{i}' for i in range(1,6)])
agg = df_q.groupby('bucket')[['pred', 'true']].mean()
```
""")

_add("### Exercises (Section 12)", r"""
**Before you start — techniques you'll use:**

- **Per-calendar-month MAE**: `df.index.to_period('M')` makes a hashable month
  label; `.groupby(month).apply(lambda g: mean_absolute_error(g['true'], g['pred']))`.
- **Tail-focused error**: mask with `y >= np.quantile(y, 0.9)` to score the right
  tail only — often the interesting regime for vol / risk models.
- **Model-vs-baseline time series**: plot both on the same axis over a narrow
  window (e.g. `test.iloc[:14*24]` = 2 weeks); overlay truth as a thick grey line.
- **Win-rate**: `win_rate = (np.abs(y - pred_model) < np.abs(y - pred_baseline)).mean()`
  — not a replacement for MAE, but tells you *how often* you're better.

*Mini-example (monthly MAE):*
```python
df = pd.DataFrame({'true': y_test, 'pred': pred}, index=test.index)
df['month'] = df.index.to_period('M')
monthly_mae = df.groupby('month').apply(lambda g: mean_absolute_error(g['true'], g['pred']))
```
""")

_add("### Exercises (Section 13)", r"""
**Before you start — techniques you'll use:**

- **Empirical coverage**: for a `[pα, p1-α]` interval, compute
  `((y >= p_lo) & (y <= p_hi)).mean()` — should match the nominal coverage (0.8 for
  p10/p90, 0.9 for p05/p95).
- **Interval width over time**: `p_hi - p_lo` plotted against time shows when the
  model is uncertain (wide) vs confident (narrow).
- **Isotonic recalibration**: `IsotonicRegression(out_of_bounds='clip')` fit on
  (val_pred_quantile, empirical_below_rate) pairs — monotone, non-parametric.
- **Fit multiple quantiles**: `lgb.LGBMRegressor(objective='quantile', alpha=q)` —
  one fit per quantile level.

*Mini-example (coverage):*
```python
coverage = ((y_test >= q_preds[0.1]) & (y_test <= q_preds[0.9])).mean()
print(f'{coverage*100:.1f}% inside [p10, p90] (target 80%)')
```
""")

_add("### Exercises (Section 14)", r"""
**Before you start — techniques you'll use:**

- **Predict-one helper**: load bundle → pull `feats` → validate keys with `[f for f
  in feats if f not in row_dict]` → `pd.DataFrame([row_dict])[feats]` → predict.
- **Property-based test for quantiles**: assert `p10 <= p50 <= p90` for every row of
  a random sample — catches regressions where quantile models cross.
- **Pydantic schema from features**: `create_model('PredictRequest', **{f: (float,
  ...) for f in FEATURES})`; validates types and presence in one call.
- **FastAPI `/health`**: return `{status, trained_through, n_features}` — useful for
  load balancers and monitoring dashboards.

*Mini-example (predict_one skeleton):*
```python
def predict_one(row_dict, bundle_path=BUNDLE_PATH):
    b = joblib.load(bundle_path)
    X = pd.DataFrame([row_dict])[b['features']]
    return {q: float(b[q].predict(X)[0]) for q in [0.1, 0.5, 0.9]}
```
""")


# ======================================================================
# time-series/time_series.ipynb
# ======================================================================

_add("### Exercises — Data Loading", r"""
**Before you start — techniques you'll use:**

- **Drop dup timestamps first**: `raw.drop_duplicates(subset='ts')` — upstream
  providers emit duplicates on venue failovers.
- **Longest raw gap**: `diff = raw_btc['ts'].sort_values().diff()`, then
  `diff.max()` (a `Timedelta`) or `diff.idxmax()` for position. Divide by
  `pd.Timedelta(hours=1)` for a float.
- **Continuity assertions**: `idx.is_monotonic_increasing`, `not idx.has_duplicates`,
  and `len(idx) == len(pd.date_range(min, max, freq='H', tz='UTC'))`.
- **Dataset summary**: `n_hours = len(df); n_days = n_hours / 24` — put start, end,
  hours, and days in a single `print(f'...')` block.

*Mini-example (continuity asserts):*
```python
expected = pd.date_range(btc.index.min(), btc.index.max(), freq='H', tz='UTC')
assert btc.index.equals(expected), 'index is not a continuous hourly grid'
```
""")

_add("### Exercises — EDA", r"""
**Before you start — techniques you'll use:**

- **Rolling volatility**: `btc['log_return'].rolling(24*7).std()` — window is rows,
  so scale by the sampling rate.
- **Hurst (R/S)**: chunk the series, for each chunk compute range / std, then
  regress `log(RS)` on `log(chunk_size)`. H ≈ 0.5 = random walk; >0.5 persistent.
- **Intraday boxplot**: `tmp['hour'] = tmp.index.hour` then
  `sns.boxplot(data=tmp, x='hour', y='log_return', showfliers=False)`.
- **Manual autocorrelation**: at lag k, `np.corrcoef(r[:-k], r[k:])[0, 1]`.
  Cross-check with `statsmodels.tsa.stattools.acf` — they should agree to ~1e-10.

*Mini-example (manual ACF):*
```python
r = btc['log_return'].dropna().values
for k in [1, 24, 168]:
    print(f'lag {k:3d}: {np.corrcoef(r[:-k], r[k:])[0, 1]:+.4f}')
```
""")

_add("### Exercises — Stationarity", r"""
**Before you start — techniques you'll use:**

- **ADF / KPSS together**: ADF null = "unit root"; KPSS null = "stationary". The
  four combinations: both reject, neither rejects, or each disagrees (often means
  trend-stationary vs difference-stationary).
- **Rolling ADF**: slide a 30-day window, record `adfuller(win).pvalue` — long
  stretches with p > 0.05 hint at regime-dependent stationarity.
- **First differences**: `close.diff()` is usually stationary even when `close`
  isn't; compare ADF p-values to confirm.
- **Log-squared returns**: proxy for log volatility —
  `np.log(r**2 + 1e-12)` — ADF on this surfaces whether volatility itself is
  stationary.

*Mini-example (rolling ADF p-values):*
```python
win, step = 24*30, 24
p_vals = [adfuller(r.iloc[i:i+win]).pvalue for i in range(0, len(r)-win, step)]
```
""")

_add("### Exercises — Decomposition", r"""
**Before you start — techniques you'll use:**

- **STL** (seasonal-trend-loess): `STL(series, period=24, robust=True).fit()` → has
  `.trend`, `.seasonal`, `.resid`. Plot the three on shared x-axes.
- **Classical decomposition**: `seasonal_decompose(series, model='additive',
  period=24)` — simpler, struggles with regime shifts.
- **Twin axis overlay**: `ax1.plot(price); ax2 = ax1.twinx(); ax2.plot(seasonal,
  color='C3')` to visualise a small seasonal pattern on top of a dominant trend.
- **Strength-of-seasonality**: $F_s = \max(0, 1 - \mathrm{Var}(R)/\mathrm{Var}(S+R))$.
  Close to 1 = strong seasonality; 0 = none.

*Mini-example (strength-of-seasonality):*
```python
S, R = decomp.seasonal.dropna(), decomp.resid.dropna()
common = S.index.intersection(R.index)
Fs = max(0.0, 1.0 - R[common].var() / (S[common] + R[common]).var())
```
""")

_add("### Exercises — Split", r"""
**Before you start — techniques you'll use:**

- **Walk-forward CV**: `TimeSeriesSplit(n_splits=5).split(train)` — expanding train,
  fixed-size val, chronological order preserved.
- **Distribution shift check**: `train['log_return'].agg(['mean', 'std', 'min',
  'max'])` vs same on test — big shifts predict worse test performance.
- **Disjointness + gap**: `train.index.intersection(test.index)` should be empty;
  `test.index.min() - train.index.max()` should be ≥ 1 bar.
- **Shuffled-split antipattern demo**: `rng.choice(n, TEST_HOURS, replace=False)`
  mixes eras — test mean/std match train mean/std, masking distribution shift.

*Mini-example (fold enumeration):*
```python
for i, (tr, va) in enumerate(TimeSeriesSplit(5).split(train)):
    print(f'Fold {i}: train={len(tr)}, val={len(va)}, val_start={train.index[va[0]]}')
```
""")

_add("### Exercises — Naive Baselines", r"""
**Before you start — techniques you'll use:**

- **Seasonal naive**: predict y at time t = y at time `t - period`. For hourly data
  and a weekly cycle, period = 168. Handle missing lookups with `get(ts, np.nan)`.
- **Blending baselines**: `0.5 * pred_a + 0.5 * np.nan_to_num(pred_b, 0)` — nan-safe
  combination keeps one model from dropping its turn.
- **Directional accuracy (manual)**: `mask = ~np.isnan(p) & (y != 0)`, then
  `(np.sign(p[mask]) == np.sign(y[mask])).mean()`.
- **Bootstrap RMSE CI**: resample residuals (not the raw series — residuals are
  roughly iid after demeaning), recompute RMSE, take 2.5 / 97.5 percentiles.

*Mini-example (seasonal-naive manual):*
```python
preds = [btc['log_return'].get(ts - pd.Timedelta(hours=168), np.nan) for ts in test.index]
```
""")

_add("### Exercises — ETS", r"""
**Before you start — techniques you'll use:**

- **Holt-Winters**: `ExponentialSmoothing(series, trend='add', seasonal='add',
  seasonal_periods=24).fit()` — captures level, slope, and 24h seasonality.
- **Damped trend** (for forecasts that shouldn't explode):
  `trend='add', damped_trend=True`. Compare AIC vs the undamped fit.
- **Smoothing parameter**: `fit.params['smoothing_level']` is α. α ≈ 0 → heavy
  smoothing (forecast = long mean); α ≈ 1 → no smoothing (forecast = last value).
- **In-sample fit check**: `model.fittedvalues` — plot over actual; used to sanity-
  check that the fit didn't collapse to the mean.

*Mini-example (HW on last 14 days):*
```python
hist, future = btc['close'].iloc[-24*14:-24], btc['close'].iloc[-24:]
hw = ExponentialSmoothing(hist, trend='add', seasonal='add', seasonal_periods=24).fit()
pred = hw.forecast(24)
```
""")

_add("### Exercises — SARIMA", r"""
**Before you start — techniques you'll use:**

- **Residual diagnostics**: `plot_acf(resid, lags=48)` and
  `acorr_ljungbox(resid, lags=[24], return_df=True)`. p > 0.05 means residuals look
  like white noise — what you want.
- **Small grid search**: loop `(p, q, P, Q)` in small ranges, fit with
  `SARIMAX(...).fit(disp=False)`, collect `(params, aic)` rows, sort by AIC.
- **AR(1) by hand**: stack `X = [1, y_{t-1}]`, solve `beta = np.linalg.lstsq(X, y,
  rcond=None)[0]`. Compare `phi` to `SARIMAX(order=(1,0,0))` output.
- **In-sample fitted**: `fitted_sarima.fittedvalues[-200:]` vs actual —
  tight overlap on returns is suspicious (means model is trading the noise).

*Mini-example (manual AR(1)):*
```python
y = r[1:]; X = np.column_stack([np.ones_like(r[:-1]), r[:-1]])
c, phi = np.linalg.lstsq(X, y, rcond=None)[0]
```
""")

_add("### Exercises — Lag Engineering", r"""
**Before you start — techniques you'll use:**

- **Cross-asset lag features**: reindex each symbol onto the BTC hourly grid, take
  `.shift(lag)` — always past-only.
- **Leakage spot-check**: corrupt a single y value deep in train and rebuild
  features; any feature row **before** the corruption must be bit-identical.
- **Interactions**: `X['feat1_x_feat2'] = X['feat1'] * X['feat2']`. Check
  univariate correlation with y to see if it carries any signal.
- **Fourier seasonality**: for period P, K harmonics, add columns
  `sin(2πk·t/P), cos(2πk·t/P)` for k in 1..K. Let trees pick which harmonics matter.

*Mini-example (Fourier features at period=24, K=3):*
```python
h = X_full.index.hour.values
for k in range(1, 4):
    X_full[f'sin_{k}'] = np.sin(2*np.pi*k*h/24)
    X_full[f'cos_{k}'] = np.cos(2*np.pi*k*h/24)
```
""")

_add("### Exercises — ML + Optuna", r"""
**Before you start — techniques you'll use:**

- **MedianPruner in an Optuna objective**: inside each CV fold call
  `trial.report(fold_mae, step=k); if trial.should_prune(): raise TrialPruned()`.
- **Extended space for LGBM**: add `trial.suggest_int('min_child_samples', 20, 200)`,
  `trial.suggest_float('feature_fraction', 0.5, 1.0)` — common regularisers.
- **XGBoost counterpart**: `xgb.XGBRegressor(objective='reg:absoluteerror',
  tree_method='hist', ...)`. Keep the same CV + number of trials for a fair compare.
- **SQLite-backed study**: `storage=f'sqlite:///{path}'`, `study_name='...'`,
  `load_if_exists=True` lets you resume.

*Mini-example (pruning loop):*
```python
for k, (tr, va) in enumerate(TimeSeriesSplit(5).split(X)):
    score = fit_and_score(trial, tr, va)
    trial.report(score, step=k)
    if trial.should_prune(): raise optuna.TrialPruned()
```
""")

_add("### Exercises — Recursive Multi-Step", r"""
**Before you start — techniques you'll use:**

- **DIRECT multi-step**: one separate model per horizon h, target = `y.shift(-h)`.
  Contrast with **recursive**: one model, feed predictions back as features.
- **DirRec hybrid**: at horizon h, augment features with the **predicted** y at
  h-1 from the previous model — chains predictions while keeping one model per step.
- **RMSE per horizon from many starts**: pick 50 random starts, run a 24-step
  forecast from each, stack errors in a 50×24 array, `np.sqrt((err**2).mean(axis=0))`.
- **RMSE-vs-horizon plot**: `ax.plot(np.arange(1, 25), rmse_h, marker='o')` — the
  growth rate tells you when the model stops beating naive.

*Mini-example (direct multi-step):*
```python
for h in [1, 6, 24]:
    y_h = y_full.shift(-h).dropna()
    X_h = X_full.loc[y_h.index]
    model_h = lgb.LGBMRegressor().fit(X_h.iloc[:cutoff], y_h.iloc[:cutoff])
```
""")

_add("### Exercises — Feature Importance", r"""
**Before you start — techniques you'll use:**

- **Permutation importance on a val slice**: `permutation_importance(model,
  X.iloc[:200], y.iloc[:200], n_repeats=5)` — small samples are fine because the
  repeats average out.
- **SHAP dependence**: `shap.dependence_plot('lag_1', shap_values, sample)` shows
  the feature value vs its marginal contribution, coloured by an auto-chosen
  interaction.
- **Feature pruning by gain**: sort `booster_.feature_importance('gain')`,
  keep top-50%, refit with default hyperparams; compare RMSE.
- **Waterfall for one row**: `explainer(row)` → `shap.plots.waterfall(exp[0])` —
  pick the most extreme positive prediction (`np.argmax(pred)`) to interpret.

*Mini-example (waterfall for the biggest prediction):*
```python
i = int(np.argmax(pred_test))
exp = explainer(X_test.iloc[[i]])
shap.plots.waterfall(exp[0])
```
""")

_add("### Exercises — Comparison", r"""
**Before you start — techniques you'll use:**

- **Diebold-Mariano**: loss differential `d = e_a**2 - e_b**2`, HAC-adjust the
  variance, `DM = mean(d) / sqrt(var_HAC(d) / n)`. Use `acovf(d, nlag=h-1)` for
  the HAC kernel with equal weights.
- **Directional accuracy ranking**: `comp.sort_values('dir_acc',
  ascending=False)[['model', 'dir_acc']]`.
- **MASE** (Mean Absolute Scaled Error): `mae_model / mean(|diff(y_train)|)`.
  < 1 means you beat the naive random walk.
- **Cumulative |error| plot**: `np.cumsum(np.abs(err_model))` vs
  `np.cumsum(np.abs(err_baseline))` — lets you see when divergence happens.

*Mini-example (MASE):*
```python
naive_mae = np.mean(np.abs(np.diff(y_train.values)))
mase = mean_absolute_error(y_test, pred_model) / naive_mae
```
""")

_add("### Exercises — Diagnostics", r"""
**Before you start — techniques you'll use:**

- **Rolling RMSE**: `pd.Series(resid**2, index=test.index).rolling(24).mean().pow(0.5)`
  — surfaces regime shifts that average metrics hide.
- **Q-Q plot vs Normal**: `scipy.stats.probplot(resid, dist='norm', plot=ax)` — fat
  tails show as departures from the diagonal in the corners.
- **Ljung-Box on residuals**: `acorr_ljungbox(resid, lags=[1, 24], return_df=True)`
  — p > 0.05 at every lag you care about ⇒ residuals look like white noise.
- **Actual vs predicted scatter with y=x**: `ax.plot([lo, hi], [lo, hi], 'r--')`
  after the scatter; scale both axes identically.

*Mini-example (rolling RMSE):*
```python
roll_rmse = (pd.Series(resid**2, index=test.index).rolling(24).mean()) ** 0.5
```
""")

_add("### Exercises — Probabilistic", r"""
**Before you start — techniques you'll use:**

- **Coverage on test**: `((y >= p10) & (y <= p90)).mean()` should be ≈ 0.80.
  Miscalibration is common — tightening loss via more data rarely fixes it alone.
- **Quantile (pinball) loss**: `np.mean(np.maximum(α*(y - ŷ), (α-1)*(y - ŷ)))`.
  Compare against `lgb_model.best_score_['valid_0']['quantile']` as a sanity check.
- **Interval-width plot**: `p90 - p10` vs time — narrow stretches indicate
  confident intervals; widening precedes breakouts.
- **Extreme quantiles (p05/p95)**: train additional quantile models with
  `objective='quantile', alpha=0.05/0.95`; verify coverage ≈ 0.90.

*Mini-example (pinball loss):*
```python
def qloss(y, yhat, alpha):
    e = y - yhat
    return np.mean(np.maximum(alpha * e, (alpha - 1) * e))
```
""")

_add("### Exercises — Deployment", r"""
**Before you start — techniques you'll use:**

- **Horizon-parameterised forecast helper**: `forecast_h(h, model, history)` runs
  the recursive forecaster, returns a DataFrame with `ts, p50` (and optional q-cols).
- **Pytest shape assertion**: `assert len(forecast_h(h, ...)) == h` — use a parametrised
  loop over `[1, 6, 24]` to catch off-by-one errors.
- **Pydantic request/response models**: `Field(ge=1, le=168)` constrains horizon,
  `List[float]` types the prediction array; FastAPI uses these for docs + validation.
- **`/health` payload**: `{'status': 'ok', 'trained_through': ..., 'n_features': ...}`
  — keeps metadata close to the artifact so pipelines can verify compatibility.

*Mini-example (forecast endpoint skeleton):*
```python
class ForecastRequest(BaseModel):
    h: int = Field(default=24, ge=1, le=168)

def forecast_h(h, model, history):
    fc = recursive_forecast(model, history, history.index.max() + pd.Timedelta(hours=1), h)
    return fc.rename(columns={'pred_log_return': 'p50'}).reset_index()
```
""")
