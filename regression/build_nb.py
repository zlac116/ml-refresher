"""Build the regression refresher notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = nb.cells

def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

# helper to wrap solution
def solution(code_str, expl):
    return (
        "<details>\n<summary>\U0001F4A1 Click to reveal solution</summary>\n\n"
        "```python\n" + code_str.strip() + "\n```\n\n"
        f"**Explanation**: {expl}\n\n</details>"
    )

# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------
md("""# Regression Refresher: Predicting BTC 24h Realized Volatility

End-to-end regression workflow on a real crypto dataset. Each major section ends
with **exercises** (problem -> empty scaffold -> hidden solution) so you can
practise implementing every step yourself.

**Target**: 24h forward realized volatility of BTC, defined as

$$
RV_t \\;=\\; \\sqrt{\\sum_{h=1}^{24} r_{t+h}^{2}} \\cdot \\sqrt{24}
$$

where $r_{t+h} = \\ln(C_{t+h}/C_{t+h-1})$ is the hourly log return.
We treat this as a tabular regression problem and benchmark classical ML against
strong vol baselines (persistence, HAR-RV).
""")

# ---------------------------------------------------------------------------
# SECTION: Setup & imports (minor, no exercises)
# ---------------------------------------------------------------------------
md("""## 0. Setup & imports

CPU-only stack: numpy / pandas / sklearn / xgboost / lightgbm / optuna / shap.
Set seeds so the notebook is reproducible.""")

code("""import warnings, os, json, time, math
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
import joblib

from scipy import stats

SEED = 42
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = (10, 4)

DATA_PATH = '/home/zlac116/Code/learning/ml-revision/data/crypto_hourly.parquet'
ARTIFACT_DIR = '/home/zlac116/Code/learning/ml-revision/regression/artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)
print('imports ok')""")

# ---------------------------------------------------------------------------
# SECTION: Problem framing (minor, no exercises)
# ---------------------------------------------------------------------------
md("""## 1. Problem framing

We are predicting **a positive, fat-tailed, highly auto-correlated quantity**
(realized vol). That has consequences for the entire pipeline:

- **Loss**: MAE/RMSE are fine for a refresher, but in vol-forecasting literature
  the **QLIKE** loss is preferred because it penalises under-prediction more
  heavily and is robust to outliers:
  $$ \\mathrm{QLIKE}(\\hat\\sigma^2, \\sigma^2) = \\frac{\\sigma^2}{\\hat\\sigma^2} - \\ln\\frac{\\sigma^2}{\\hat\\sigma^2} - 1 $$
- **Persistence baseline is strong**: vol clusters, so "tomorrow's vol = today's
  vol" is a brutal benchmark. If you crush it by 50%, you are leaking.
- **Time-aware splitting only**: no shuffled CV, no group-K-fold. Rolling /
  expanding windows.
- **Stationarity**: vol regimes change, so we will check OOS robustness, not
  just an aggregate test number.""")

# ---------------------------------------------------------------------------
# SECTION 2: Data loading & sanity (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 2. Data loading & sanity

Load the shared parquet, check schema, look for gaps. Crypto data is *usually*
clean on majors but small drops happen (exchange downtime, ingestion gaps).
Catching these early prevents NaNs from poisoning rolling features later.""")

code("""raw = pd.read_parquet(DATA_PATH)
raw['ts'] = pd.to_datetime(raw['ts'], utc=True)
raw = raw.sort_values(['symbol', 'ts']).reset_index(drop=True)
print('shape:', raw.shape)
print('symbols:', sorted(raw['symbol'].unique()))
print('range :', raw['ts'].min(), '->', raw['ts'].max())
raw.head()""")

code("""# basic sanity: per-symbol coverage and duplicates
sanity = (raw.groupby('symbol')
              .agg(rows=('ts', 'size'),
                   first=('ts', 'min'),
                   last=('ts', 'max'),
                   dupes=('ts', lambda s: int(s.duplicated().sum())))
              .reset_index())
sanity['days'] = (sanity['last'] - sanity['first']).dt.total_seconds() / 86400
sanity""")

md("""### Exercises (Section 2)""")

# ex 2.1
md("""**Exercise 2.1** - Compute the percentage of missing values for each
column, broken down per symbol. Return a tidy DataFrame with one row per
(symbol, column). Expected: very small or zero values for this dataset.""")
code("# Your answer here\n")
md(solution(
"""miss = (raw.set_index('ts')
            .groupby('symbol')
            .apply(lambda g: g.isna().mean() * 100)
            .reset_index()
            .melt(id_vars='symbol', var_name='column', value_name='pct_missing'))
miss.sort_values('pct_missing', ascending=False).head(10)""",
"`groupby(symbol).apply(isna().mean())` gives proportions per column; multiply by 100 and melt to long form. For this clean dataset all values are 0%."
))

# ex 2.2
md("""**Exercise 2.2** - Find the **longest gap** in BTC's series. Return the
gap duration (Timedelta), start ts, and end ts. Hint: `diff()` on a sorted
timestamp index.""")
code("# Your answer here\n")
md(solution(
"""btc_ts = raw[raw.symbol == 'BTC']['ts'].sort_values().reset_index(drop=True)
gaps = btc_ts.diff()
i = int(gaps.idxmax())
print('longest gap:', gaps.iloc[i],
      ' between', btc_ts.iloc[i-1], 'and', btc_ts.iloc[i])""",
"Sort, diff, take the argmax. For a clean hourly dataset this should be 1 hour exactly."
))

# ex 2.3
md("""**Exercise 2.3** - Verify hourly continuity for BTC by reindexing on the
full hourly range and counting how many hours are missing. Expected: 0 or close
to 0.""")
code("# Your answer here\n")
md(solution(
"""btc = raw[raw.symbol == 'BTC'].set_index('ts').sort_index()
full = pd.date_range(btc.index.min(), btc.index.max(), freq='1h', tz='UTC')
missing = full.difference(btc.index)
print('expected hours:', len(full),
      ' actual:', len(btc),
      ' missing:', len(missing))""",
"`pd.date_range` builds the canonical hourly grid; `difference` against the actual index pinpoints missing hours -- the standard reindex-and-check pattern."
))

# ex 2.4
md("""**Exercise 2.4** - Plot the number of rows per calendar day for BTC. Days
with fewer than 24 rows indicate dropped hours. Use a line plot with day on x
and row count on y.""")
code("# Your answer here\n")
md(solution(
"""btc = raw[raw.symbol == 'BTC'].copy()
per_day = btc.groupby(btc['ts'].dt.floor('D')).size()
fig, ax = plt.subplots(figsize=(11, 3))
per_day.plot(ax=ax, lw=0.8)
ax.axhline(24, color='k', ls='--', alpha=0.5, label='24 rows expected')
ax.set_title('BTC rows per day'); ax.set_xlabel('date'); ax.set_ylabel('rows')
ax.legend(); plt.tight_layout()""",
"Floor timestamps to day, group, count. Dips below 24 are exchange or ingestion outages."
))

# ---------------------------------------------------------------------------
# SECTION 3: EDA (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 3. Exploratory data analysis

Three things we MUST verify visually for any vol-forecasting problem:

1. **Fat-tailed return distribution** -> motivates robust losses, log targets.
2. **Vol clustering**: ACF of squared returns decays slowly -> persistence
   baseline will be strong.
3. **Seasonality**: weekly / hourly patterns in vol can drive feature design.
""")

code("""# build a wide BTC frame for EDA
btc = (raw[raw.symbol == 'BTC']
       .set_index('ts').sort_index()[['open','high','low','close','volume']]
       .copy())
btc['log_ret'] = np.log(btc['close'] / btc['close'].shift(1))
btc.head()""")

code("""fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
btc['close'].plot(ax=axes[0], color='C0')
axes[0].set_title('BTC close (USD)'); axes[0].set_ylabel('price')
btc['log_ret'].plot(ax=axes[1], color='C1', lw=0.5)
axes[1].set_title('BTC hourly log returns'); axes[1].set_ylabel('log return')
plt.tight_layout()""")

code("""# return distribution: clear fat tails
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(btc['log_ret'].dropna(), bins=120, kde=True, ax=axes[0])
axes[0].set_title('log return histogram')
stats.probplot(btc['log_ret'].dropna(), dist='norm', plot=axes[1])
axes[1].set_title('QQ plot vs Normal')
plt.tight_layout()""")

code("""# ACF of squared returns -> volatility clustering
from pandas.plotting import autocorrelation_plot
sq = (btc['log_ret']**2).dropna()
lags = np.arange(1, 73)
acf_vals = [sq.autocorr(lag=l) for l in lags]
fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(lags, acf_vals, color='C2')
ax.axhline(0, color='k', lw=0.5)
ax.set_title('ACF of squared hourly returns (BTC) - vol clustering')
ax.set_xlabel('lag (hours)'); ax.set_ylabel('autocorrelation')
plt.tight_layout()""")

md("""### Exercises (Section 3)""")

md("""**Exercise 3.1** - Plot the rolling 24h realized volatility of BTC over
time (the same quantity we will predict). Use the formula
`sqrt(rolling_24h(log_ret^2)) * sqrt(24)`. Title and axis labels mandatory.""")
code("# Your answer here\n")
md(solution(
"""rv = np.sqrt((btc['log_ret']**2).rolling(24).sum()) * np.sqrt(24)
fig, ax = plt.subplots(figsize=(11, 3.5))
rv.plot(ax=ax, color='C3', lw=0.7)
ax.set_title('BTC 24h rolling realized volatility')
ax.set_xlabel('date'); ax.set_ylabel('annualised RV')
plt.tight_layout()""",
"Sum squared returns over a 24h window, sqrt and scale by sqrt(24) to annualise from hourly. This is the *backward-looking* twin of our target."
))

md("""**Exercise 3.2** - Compute the skew and excess kurtosis of BTC hourly log
returns. A normal distribution has skew=0 and excess kurt=0. What do you
observe?""")
code("# Your answer here\n")
md(solution(
"""r = btc['log_ret'].dropna()
print(f'skew         : {stats.skew(r):+.3f}')
print(f'excess kurt  : {stats.kurtosis(r):+.3f}')""",
"Crypto returns typically show modest skew and *very* high kurtosis (fat tails). This is why squared returns are so informative for vol forecasting and why we may want a log-transformed target."
))

md("""**Exercise 3.3** - Boxplot of realized 24h vol by **weekday**. Is there a
visually obvious weekday effect? (Use `dt.day_name()`.)""")
code("# Your answer here\n")
md(solution(
"""rv = np.sqrt((btc['log_ret']**2).rolling(24).sum()) * np.sqrt(24)
df_w = pd.DataFrame({'rv': rv, 'weekday': rv.index.day_name()}).dropna()
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
fig, ax = plt.subplots(figsize=(9, 3.5))
sns.boxplot(data=df_w, x='weekday', y='rv', order=order, showfliers=False, ax=ax)
ax.set_title('BTC 24h realized vol by weekday')
plt.tight_layout()""",
"Weekend liquidity tends to be lower in crypto so we sometimes see slightly elevated vol; the effect is usually weak but visible. Day-of-week is therefore a worthwhile feature."
))

md("""**Exercise 3.4** - Scatter realized vol at time t vs realized vol at time
t-24 (one-day-lagged). This visualises the autocorrelation that makes
persistence so hard to beat. Add a y=x line for reference.""")
code("# Your answer here\n")
md(solution(
"""rv = np.sqrt((btc['log_ret']**2).rolling(24).sum()) * np.sqrt(24)
df = pd.DataFrame({'rv': rv, 'rv_lag24': rv.shift(24)}).dropna()
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(df['rv_lag24'], df['rv'], s=4, alpha=0.3)
lim = [0, df.max().max()]
ax.plot(lim, lim, 'k--', lw=1)
ax.set_xlabel('RV(t-24)'); ax.set_ylabel('RV(t)')
ax.set_title(f"RV autocorrelation rho = {df.corr().iloc[0,1]:.3f}")
plt.tight_layout()""",
"Strong positive correlation around the y=x line is the visual signature of vol persistence. Any model must beat this trivial relationship."
))

# ---------------------------------------------------------------------------
# SECTION 4: Feature engineering (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 4. Feature engineering (leakage-free)

Every feature is **strictly backward-looking** with respect to time `t`. The
target uses returns over `(t, t+24]`. To stay safe:

- All rolling windows end *at or before* `t` (use `.shift(1)` AFTER the rolling
  if the rolling closes on the last bar by default in pandas).
- Cross-asset features at time `t` use only data up to `t` for the OTHER asset.
- Drop NaNs from rolling **and** target windows together.

**HAR-RV** (Corsi 2009) is a very strong vol baseline: model RV as a linear
combination of yesterday's RV, last week's average RV, and last month's average
RV. We will include those three features explicitly.

**Range estimators** use the high/low/open/close to form variance estimates that
are statistically more efficient than squared close-to-close returns:

- **Parkinson**: $\\sigma^2 = \\frac{1}{4 \\ln 2} (\\ln H/L)^2$
- **Garman-Klass**: $\\sigma^2 = 0.5 (\\ln H/L)^2 - (2\\ln 2 - 1)(\\ln C/O)^2$
""")

code("""# build per-symbol feature frames, target is 24h FORWARD RV of BTC
def hourly_log_ret(df):
    return np.log(df['close'] / df['close'].shift(1))

def realized_vol_fwd(log_ret, window=24):
    # forward 24h RV starting at t+1: shift(-1) so window aligns to (t, t+24]
    fwd_sq = (log_ret.shift(-1)**2).rolling(window).sum().shift(-(window-1))
    return np.sqrt(fwd_sq) * np.sqrt(window)

def realized_vol_back(log_ret, window=24):
    return np.sqrt((log_ret**2).rolling(window).sum()) * np.sqrt(window)

# pivot to wide frames per field for cross-asset features
piv_close = raw.pivot(index='ts', columns='symbol', values='close').sort_index()
piv_high  = raw.pivot(index='ts', columns='symbol', values='high').sort_index()
piv_low   = raw.pivot(index='ts', columns='symbol', values='low').sort_index()
piv_open  = raw.pivot(index='ts', columns='symbol', values='open').sort_index()
piv_vol   = raw.pivot(index='ts', columns='symbol', values='volume').sort_index()
log_ret   = np.log(piv_close / piv_close.shift(1))

print('pivots built:', piv_close.shape, '(ts x symbol)')""")

code("""# --- BTC features (the model's input table) ---
feat = pd.DataFrame(index=piv_close.index)

# past RV at multiple horizons (HAR-style)
for w in [6, 12, 24, 72, 168]:
    feat[f'rv_back_{w}h'] = realized_vol_back(log_ret['BTC'], window=w)

# HAR-RV components: lag1d, week avg, month avg of *daily* RV
rv_24 = realized_vol_back(log_ret['BTC'], window=24)
feat['har_d'] = rv_24                       # last 24h RV
feat['har_w'] = rv_24.rolling(24*7).mean()  # weekly avg
feat['har_m'] = rv_24.rolling(24*30).mean() # monthly avg

# past returns at multiple horizons (sign + magnitude info)
for w in [1, 6, 24, 72]:
    feat[f'ret_{w}h'] = log_ret['BTC'].rolling(w).sum()

# range estimators (Parkinson, Garman-Klass)
hl = np.log(piv_high['BTC'] / piv_low['BTC'])
co = np.log(piv_close['BTC'] / piv_open['BTC'])
park_var = (hl**2) / (4 * np.log(2))
gk_var   = 0.5 * (hl**2) - (2*np.log(2) - 1) * (co**2)
feat['parkinson_24h']    = np.sqrt(park_var.rolling(24).sum()) * np.sqrt(24)
feat['garman_klass_24h'] = np.sqrt(gk_var.clip(lower=0).rolling(24).sum()) * np.sqrt(24)

# volume features
feat['log_dollar_vol_24h'] = np.log1p((piv_close['BTC'] * piv_vol['BTC']).rolling(24).sum())
v = piv_vol['BTC']
feat['volume_z_24h'] = (v - v.rolling(168).mean()) / v.rolling(168).std()

# cross-asset RV features
for sym in ['ETH', 'SOL', 'BNB']:
    feat[f'{sym}_rv_24h']  = realized_vol_back(log_ret[sym], window=24)
    feat[f'{sym}_ret_24h'] = log_ret[sym].rolling(24).sum()

# time encodings (cyclical)
hours = feat.index.hour
days  = feat.index.dayofweek
feat['hour_sin'] = np.sin(2 * np.pi * hours / 24)
feat['hour_cos'] = np.cos(2 * np.pi * hours / 24)
feat['dow_sin']  = np.sin(2 * np.pi * days / 7)
feat['dow_cos']  = np.cos(2 * np.pi * days / 7)

# target
feat['target_rv_24h_fwd'] = realized_vol_fwd(log_ret['BTC'], window=24)

# drop rows with NaNs from rolling and forward target
data = feat.dropna().copy()
print('feature matrix:', data.shape)
data.tail(3)""")

code("""# --- leakage check: target at time t MUST be deterministic from data after t ---
# verify that target_rv_24h_fwd at time t uses returns from t+1 .. t+24
sample_t = data.index[1000]
expected = np.sqrt(((log_ret['BTC'].loc[sample_t:].iloc[1:25])**2).sum()) * np.sqrt(24)
print(f"t            = {sample_t}")
print(f"stored target= {data.loc[sample_t, 'target_rv_24h_fwd']:.6f}")
print(f"recomputed   = {expected:.6f}")
assert np.isclose(data.loc[sample_t, 'target_rv_24h_fwd'], expected, rtol=1e-9)
print('leakage check passed: target uses ONLY future returns')""")

md("""### Exercises (Section 4)""")

md("""**Exercise 4.1** - Implement the **Parkinson** volatility estimator over a
rolling window of `n` hours, returning an annualised vol series. The formula
is $\\sigma^2_t = \\frac{1}{4\\ln 2}(\\ln H_t/L_t)^2$; sum over the window, sqrt
and scale by $\\sqrt{n}$.""")
code("# Your answer here\n# def parkinson(high, low, window): ...\n")
md(solution(
"""def parkinson(high, low, window):
    hl = np.log(high / low)
    var = (hl**2) / (4 * np.log(2))
    return np.sqrt(var.rolling(window).sum()) * np.sqrt(window)

p = parkinson(piv_high['BTC'], piv_low['BTC'], 24)
print(p.tail())""",
"Parkinson uses only H and L so it is robust to opening jumps. Statistically ~5x more efficient than squared close-to-close for a Brownian process."
))

md("""**Exercise 4.2** - Implement the **Garman-Klass** estimator. Formula:
$\\sigma^2_t = 0.5(\\ln H/L)^2 - (2\\ln 2 - 1)(\\ln C/O)^2$. Clip negative
variance to zero before summing.""")
code("# Your answer here\n# def garman_klass(open_, high, low, close, window): ...\n")
md(solution(
"""def garman_klass(open_, high, low, close, window):
    hl = np.log(high / low)
    co = np.log(close / open_)
    var = 0.5 * (hl**2) - (2*np.log(2) - 1) * (co**2)
    var = var.clip(lower=0)
    return np.sqrt(var.rolling(window).sum()) * np.sqrt(window)

gk = garman_klass(piv_open['BTC'], piv_high['BTC'],
                  piv_low['BTC'],  piv_close['BTC'], 24)
print(gk.tail())""",
"Garman-Klass adds the open-close term, gaining further efficiency over Parkinson when opening jumps carry information. Variance can go negative for noisy bars; clip to zero."
))

md("""**Exercise 4.3** - Build the full **HAR-RV** feature set (lag-1h,
trailing-24h average, trailing-168h average of hourly RV). Then assert there is
no leakage: row at time t must not depend on any return from t+1 onward.
Hint: rebuild from `log_ret['BTC']` and check a sample timestamp.""")
code("# Your answer here\n")
md(solution(
"""rv1 = (log_ret['BTC']**2).rolling(1).sum().pipe(np.sqrt) * np.sqrt(1)
har = pd.DataFrame({
    'har_lag1':   rv1.shift(1),                 # strictly past
    'har_24havg': rv1.shift(1).rolling(24).mean(),
    'har_168havg':rv1.shift(1).rolling(168).mean(),
}).dropna()

t = har.index[5000]
# the rolling/shift means HAR at t uses returns up to t-1 only
assert har.loc[t, 'har_lag1'] == rv1.shift(1).loc[t]
print(har.tail())""",
"`shift(1)` BEFORE the rolling guarantees the window ends at t-1. Without that shift, pandas' default 'window closes at current bar' would silently include r_t."
))

md("""**Exercise 4.4** - Add cross-asset RV features (ETH, SOL, BNB at 24h
window) and report their absolute Pearson correlation with the BTC 24h forward
target. Which is most informative?""")
code("# Your answer here\n")
md(solution(
"""tgt = data['target_rv_24h_fwd']
rows = []
for sym in ['ETH', 'SOL', 'BNB']:
    rv_other = realized_vol_back(log_ret[sym], 24).reindex(tgt.index)
    rows.append({'symbol': sym, 'corr_with_target': rv_other.corr(tgt)})
pd.DataFrame(rows).sort_values('corr_with_target', ascending=False)""",
"Crypto majors share systemic risk so cross-asset RV is highly correlated with BTC RV. ETH typically tops the table; this kind of feature is cheap alpha for vol models."
))

# ---------------------------------------------------------------------------
# SECTION 5: Train/val/test split (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 5. Train / val / test split (time-aware)

Chronological 70/15/15. **No shuffling**. Validation is used for early stopping
and tuning; test is touched **once** at the very end.""")

code("""data = data.sort_index()
n = len(data)
i_train = int(n * 0.70)
i_val   = int(n * 0.85)

train = data.iloc[:i_train]
val   = data.iloc[i_train:i_val]
test  = data.iloc[i_val:]

FEATURES = [c for c in data.columns if c != 'target_rv_24h_fwd']
TARGET   = 'target_rv_24h_fwd'

print(f"train: {len(train):>5}  {train.index.min()} -> {train.index.max()}")
print(f"val  : {len(val):>5}  {val.index.min()} -> {val.index.max()}")
print(f"test : {len(test):>5}  {test.index.min()} -> {test.index.max()}")
print(f"features: {len(FEATURES)}")""")

md("""### Exercises (Section 5)""")

md("""**Exercise 5.1** - Implement an **expanding-window walk-forward split**
generator that yields `(train_idx, val_idx)` index arrays for `n_splits` folds
on the full `data` frame. Each fold must extend the training window forward and
hold out a fixed-size validation block immediately after.""")
code("# Your answer here\n# def expanding_walk_forward(n_rows, n_splits, val_size): ...\n")
md(solution(
"""def expanding_walk_forward(n_rows, n_splits, val_size):
    # initial train = first chunk, then expand by val_size each fold
    initial = n_rows - n_splits * val_size
    if initial <= 0:
        raise ValueError('not enough rows for that config')
    for k in range(n_splits):
        end_train = initial + k * val_size
        end_val   = end_train + val_size
        yield (np.arange(0, end_train), np.arange(end_train, end_val))

for tr, va in expanding_walk_forward(len(data), n_splits=5, val_size=500):
    print(f'train [0:{tr[-1]+1}]  val [{va[0]}:{va[-1]+1}]')""",
"Expanding-window respects time order while letting later folds learn from more data. Compared to a fixed-window walk-forward it trades off a bit of regime-adaptation for more signal."
))

md("""**Exercise 5.2** - Verify there is **no temporal overlap** between train,
val and test by asserting the max(ts) of each split is strictly less than the
min(ts) of the next.""")
code("# Your answer here\n")
md(solution(
"""assert train.index.max() < val.index.min(), 'train/val overlap!'
assert val.index.max()   < test.index.min(), 'val/test overlap!'
print('no temporal overlap')""",
"A trivial but essential sanity check. Run it after every split-construction change."
))

md("""**Exercise 5.3** - Compare the **distribution of the target** across
splits (mean, std, p95). Different distributions = different vol regimes =
expect OOS degradation.""")
code("# Your answer here\n")
md(solution(
"""rows = []
for name, df in [('train', train), ('val', val), ('test', test)]:
    rows.append({'split': name,
                 'mean': df[TARGET].mean(),
                 'std':  df[TARGET].std(),
                 'p95':  df[TARGET].quantile(0.95),
                 'n':    len(df)})
pd.DataFrame(rows)""",
"If test has materially different stats than train (e.g. test in a calm regime, train in a turbulent one) you should expect the model's absolute errors to scale with the new regime."
))

md("""**Exercise 5.4** - Build a **purged CV** generator that, for a given
forward target horizon `H` (here 24 hours), drops samples from the training fold
whose target window crosses into the validation fold. This is Lopez de Prado's
recipe for preventing target overlap.""")
code("# Your answer here\n# def purged_cv_split(n_rows, n_splits, val_size, horizon): ...\n")
md(solution(
"""def purged_cv_split(n_rows, n_splits, val_size, horizon):
    initial = n_rows - n_splits * val_size
    for k in range(n_splits):
        end_train = initial + k * val_size
        purge_from = max(0, end_train - horizon)
        train_idx = np.arange(0, purge_from)            # drop last `horizon` rows
        val_idx   = np.arange(end_train, end_train + val_size)
        yield train_idx, val_idx

for tr, va in purged_cv_split(len(data), 4, 500, horizon=24):
    print(f'train rows {len(tr)}  val [{va[0]}:{va[-1]+1}]')""",
"Targets at the tail of the train block use returns that fall inside the val block. Purging drops `horizon` rows of train so this overlap cannot leak."
))

# ---------------------------------------------------------------------------
# SECTION 6: Baselines (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 6. Baselines

Three baselines, in order of difficulty to beat:

1. **Historical mean** of train RV.
2. **Persistence**: predict yesterday's realized vol (`har_d` in our features).
3. **HAR-RV** linear regression on the three HAR features.

For vol forecasting, persistence is the workhorse benchmark. If a fancy model
fails to beat HAR by a meaningful margin, the alpha is illusory.""")

code("""def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return dict(MAE=mae, RMSE=rmse, R2=r2, MAPE=mape)

# baseline 1: historical mean
mean_pred = np.full(len(val), train[TARGET].mean())
m_mean = metrics(val[TARGET].values, mean_pred)

# baseline 2: persistence (yesterday's realized 24h vol = har_d feature)
m_pers = metrics(val[TARGET].values, val['har_d'].values)

# baseline 3: HAR-RV linear regression
har_cols = ['har_d', 'har_w', 'har_m']
har_lr = LinearRegression().fit(train[har_cols], train[TARGET])
har_pred = har_lr.predict(val[har_cols])
m_har = metrics(val[TARGET].values, har_pred)

baseline_df = pd.DataFrame([m_mean, m_pers, m_har],
                           index=['historical_mean', 'persistence', 'HAR-RV']).round(5)
baseline_df""")

md("""### Exercises (Section 6)""")

md("""**Exercise 6.1** - Implement the persistence baseline manually (without
using the `har_d` feature column). Use `log_ret['BTC']` directly to compute
the trailing 24h realized vol at each val timestamp.""")
code("# Your answer here\n")
md(solution(
"""rv24_back = realized_vol_back(log_ret['BTC'], 24)
y_pred = rv24_back.reindex(val.index).values
y_true = val[TARGET].values
print(metrics(y_true, y_pred))""",
"This re-derives `har_d` from raw returns, confirming the feature column is just the trailing 24h RV. Useful to know if you ever need to reproduce features in production from cold-start data."
))

md("""**Exercise 6.2** - Compute the **QLIKE** loss for the persistence
baseline. Formula:
$\\mathrm{QLIKE} = \\sigma^2/\\hat\\sigma^2 - \\ln(\\sigma^2/\\hat\\sigma^2) - 1$
on the variance scale (square the vol predictions first). Lower is better.""")
code("# Your answer here\n")
md(solution(
"""def qlike(y_true_vol, y_pred_vol, eps=1e-12):
    yt = (y_true_vol**2) + eps
    yp = (y_pred_vol**2) + eps
    return float(np.mean(yt/yp - np.log(yt/yp) - 1.0))

print('QLIKE persistence:', qlike(val[TARGET].values, val['har_d'].values))""",
"QLIKE is asymmetric: under-prediction blows up the ratio yt/yp -> log term cannot save it. That property is exactly what a vol forecaster wants since under-predicting vol is dangerous."
))

md("""**Exercise 6.3** - Combine all three baselines into a tidy DataFrame
sorted by validation MAE ascending.""")
code("# Your answer here\n")
md(solution(
"""baseline_df.sort_values('MAE')""",
"In practice HAR usually beats persistence by a small but real margin, and historical mean is far behind because it ignores the regime."
))

md("""**Exercise 6.4** - Bootstrap a 95% CI on the persistence MAE on val
(1000 resamples with replacement of the row indices).""")
code("# Your answer here\n")
md(solution(
"""rng = np.random.default_rng(SEED)
y_true = val[TARGET].values
y_pred = val['har_d'].values
err    = np.abs(y_true - y_pred)
maes   = [err[rng.integers(0, len(err), len(err))].mean() for _ in range(1000)]
lo, hi = np.quantile(maes, [0.025, 0.975])
print(f'persistence MAE 95% CI: [{lo:.5f}, {hi:.5f}]')""",
"Bootstrap CI on residual magnitudes gives a quick uncertainty band on point metrics. If your fancy model's MAE falls inside this CI, you have not actually beaten persistence."
))

# ---------------------------------------------------------------------------
# SECTION 7: Model selection (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 7. Model selection

Five candidates: Ridge, ElasticNet, RandomForest, XGBoost, LightGBM. Evaluate
with `TimeSeriesSplit` on the train+val rows (NOT touching test).""")

code("""# combine train+val for the CV experiment
cv_pool = pd.concat([train, val]).sort_index()
X_cv = cv_pool[FEATURES].values
y_cv = cv_pool[TARGET].values

models = {
    'Ridge':       Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]),
    'ElasticNet':  Pipeline([('sc', StandardScaler()), ('m', ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000))]),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10,
                                          n_jobs=-1, random_state=SEED),
    'XGBoost':     xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=SEED, n_jobs=-1, verbosity=0),
    'LightGBM':    lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=SEED, n_jobs=-1, verbose=-1),
}

tscv = TimeSeriesSplit(n_splits=4)
rows = []
for name, mdl in models.items():
    fold_mae = []
    t0 = time.time()
    for tr, va in tscv.split(X_cv):
        mdl.fit(X_cv[tr], y_cv[tr])
        pred = mdl.predict(X_cv[va])
        fold_mae.append(mean_absolute_error(y_cv[va], pred))
    rows.append({'model': name,
                 'cv_mae_mean': np.mean(fold_mae),
                 'cv_mae_std':  np.std(fold_mae),
                 'fit_seconds': time.time() - t0})
cv_results = pd.DataFrame(rows).sort_values('cv_mae_mean').reset_index(drop=True)
cv_results.round(5)""")

md("""### Exercises (Section 7)""")

md("""**Exercise 7.1** - Wrap each model in a `Pipeline` with a `StandardScaler`
where appropriate (linear models definitely; tree-based models do not need it).
Explain in one line why scaling matters for Ridge but not RF.""")
code("# Your answer here\n")
md(solution(
"""scaled = {
    'Ridge':      Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]),
    'ElasticNet': Pipeline([('sc', StandardScaler()), ('m', ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000))]),
}
unscaled = {
    'RF':       RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED),
    'XGB':      xgb.XGBRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, verbosity=0),
    'LGBM':     lgb.LGBMRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1),
}
print('Ridge minimises ||y - Xw||^2 + a||w||^2 -- the L2 penalty hits coefficients,')
print('so feature scale directly distorts the regularisation.')
print('Trees split on thresholds, scale-invariant.')""",
"Linear models with L1/L2 penalties are scale-sensitive because the penalty is on raw weight magnitude. Trees split on thresholds and ignore scale entirely."
))

md("""**Exercise 7.2** - From the `cv_results` table, plot fit time vs CV MAE
to visualise the speed/accuracy tradeoff. Annotate each point with the model
name.""")
code("# Your answer here\n")
md(solution(
"""fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(cv_results['fit_seconds'], cv_results['cv_mae_mean'], s=80)
for _, r in cv_results.iterrows():
    ax.annotate(r['model'], (r['fit_seconds'], r['cv_mae_mean']),
                xytext=(5,5), textcoords='offset points')
ax.set_xlabel('CV fit time (s)'); ax.set_ylabel('CV MAE')
ax.set_title('Speed / accuracy tradeoff')
plt.tight_layout()""",
"GBMs typically win accuracy. RF is slow and only marginally better than GBMs. Linear models are blazing fast and surprisingly competitive on this kind of HAR-rich feature set."
))

md("""**Exercise 7.3** - Run `permutation_importance` on the RandomForest fit on
train, evaluated on val. Show the top 10 features.""")
code("# Your answer here\n")
md(solution(
"""rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=SEED)
rf.fit(train[FEATURES], train[TARGET])
pi = permutation_importance(rf, val[FEATURES], val[TARGET],
                            n_repeats=5, random_state=SEED, n_jobs=-1)
pi_df = (pd.DataFrame({'feature': FEATURES, 'importance': pi.importances_mean})
           .sort_values('importance', ascending=False).head(10))
pi_df""",
"Permutation importance is model-agnostic and uses HELD-OUT data, unlike gain-based RF importance which can be biased toward high-cardinality features."
))

md("""**Exercise 7.4** - Build a simple averaging ensemble of the top-2 models
by CV MAE. Compare its val MAE against each individual top-2 model.""")
code("# Your answer here\n")
md(solution(
"""top2 = cv_results.head(2)['model'].tolist()
preds = {}
for name in top2:
    m = models[name]
    m.fit(train[FEATURES], train[TARGET])
    preds[name] = m.predict(val[FEATURES])
ens = np.mean(list(preds.values()), axis=0)

rows = [{'model': n, 'val_MAE': mean_absolute_error(val[TARGET], p)}
        for n, p in preds.items()]
rows.append({'model': '+ensemble_avg',
             'val_MAE': mean_absolute_error(val[TARGET], ens)})
pd.DataFrame(rows)""",
"Averaging models with weakly-correlated errors usually improves MAE. If the ensemble is *worse*, the two models are too similar (e.g. both GBMs on identical features)."
))

# ---------------------------------------------------------------------------
# SECTION 8: Target transformation (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 8. Target transformation experiment

RV is positive and right-skewed -> a log transform may stabilise variance and
help linear/tree models alike. We compare `predict(y)` vs `predict(log y)`,
**always reporting the metric in the original units** (otherwise the comparison
is meaningless because log MAE and raw MAE are on different scales).""")

code("""best_name = cv_results.iloc[0]['model']
print('best model from CV:', best_name)

def fit_predict_raw(model_factory, X_tr, y_tr, X_va):
    m = model_factory()
    m.fit(X_tr, y_tr)
    return m.predict(X_va)

def fit_predict_log(model_factory, X_tr, y_tr, X_va):
    m = model_factory()
    m.fit(X_tr, np.log(y_tr))
    return np.exp(m.predict(X_va))

best_factory = lambda: lgb.LGBMRegressor(n_estimators=400, num_leaves=31,
                                         learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8,
                                         random_state=SEED, n_jobs=-1, verbose=-1)

p_raw = fit_predict_raw(best_factory, train[FEATURES].values, train[TARGET].values, val[FEATURES].values)
p_log = fit_predict_log(best_factory, train[FEATURES].values, train[TARGET].values, val[FEATURES].values)

cmp = pd.DataFrame([
    {'target_form': 'raw',  **metrics(val[TARGET].values, p_raw)},
    {'target_form': 'log',  **metrics(val[TARGET].values, p_log)},
]).round(5)
cmp""")

md("""### Exercises (Section 8)""")

md("""**Exercise 8.1** - Implement a helper `predict_in_log_space(model, X) ->
np.ndarray` that takes a model trained on `log(y)` and returns predictions
back-transformed into the original units. Apply a small numerical safeguard
against extreme predictions (e.g., clip to reasonable range).""")
code("# Your answer here\n")
md(solution(
"""def predict_in_log_space(model, X, lo=-15, hi=5):
    log_pred = np.clip(model.predict(X), lo, hi)
    return np.exp(log_pred)

m = best_factory().fit(train[FEATURES], np.log(train[TARGET]))
pred = predict_in_log_space(m, val[FEATURES])
print('val MAE:', mean_absolute_error(val[TARGET], pred))""",
"Always clip in log space before exponentiating: a single huge log prediction blows up to inf and contaminates downstream metrics. The bounds [-15, 5] are generous for vol on the [1e-4, 100] scale."
))

md("""**Exercise 8.2** - Compare RMSE on raw target vs RMSE on log target.
Explain in one line which comparison is meaningful.""")
code("# Your answer here\n")
md(solution(
"""m_raw = best_factory().fit(train[FEATURES], train[TARGET])
m_log = best_factory().fit(train[FEATURES], np.log(train[TARGET]))

rmse_raw_origunits = math.sqrt(mean_squared_error(val[TARGET], m_raw.predict(val[FEATURES])))
rmse_log_origunits = math.sqrt(mean_squared_error(val[TARGET], np.exp(m_log.predict(val[FEATURES]))))
rmse_log_logunits  = math.sqrt(mean_squared_error(np.log(val[TARGET]), m_log.predict(val[FEATURES])))

print(f'RMSE raw target,        eval in raw units: {rmse_raw_origunits:.5f}')
print(f'RMSE log target,        eval in raw units: {rmse_log_origunits:.5f}')
print(f'RMSE log target,        eval in LOG units: {rmse_log_logunits:.5f}  <- not comparable')""",
"Only the first two are comparable because they are on the same scale. The log-units RMSE is meaningless for cross-target comparison."
))

md("""**Exercise 8.3** - Try Box-Cox on the target via `scipy.stats.boxcox` and
compare val RMSE in original units against raw and log. Use the lambda fitted
on the train set ONLY.""")
code("# Your answer here\n")
md(solution(
"""y_tr_bc, lam = stats.boxcox(train[TARGET].values)
m = best_factory().fit(train[FEATURES], y_tr_bc)
pred_bc = m.predict(val[FEATURES])
# inverse Box-Cox
pred_inv = (pred_bc * lam + 1) ** (1/lam) if lam != 0 else np.exp(pred_bc)
print(f'lambda = {lam:.4f}')
print('Box-Cox RMSE (orig units):', math.sqrt(mean_squared_error(val[TARGET], pred_inv)))""",
"Box-Cox generalises log transformation. lambda close to 0 -> log; close to 1 -> identity. For vol it usually lands around 0-0.3, validating the log instinct."
))

md("""**Exercise 8.4** - Plot residuals vs predicted for raw-target vs
log-target models. Does the variance stabilise visually?""")
code("# Your answer here\n")
md(solution(
"""m_raw = best_factory().fit(train[FEATURES], train[TARGET])
m_log = best_factory().fit(train[FEATURES], np.log(train[TARGET]))
p_raw = m_raw.predict(val[FEATURES])
p_log = np.exp(m_log.predict(val[FEATURES]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
axes[0].scatter(p_raw, val[TARGET]-p_raw, s=4, alpha=0.4)
axes[0].axhline(0, color='k', ls='--'); axes[0].set_title('raw target residuals')
axes[0].set_xlabel('predicted'); axes[0].set_ylabel('residual')
axes[1].scatter(p_log, val[TARGET]-p_log, s=4, alpha=0.4)
axes[1].axhline(0, color='k', ls='--'); axes[1].set_title('log-target residuals')
axes[1].set_xlabel('predicted')
plt.tight_layout()""",
"For heteroskedastic targets, the log-trained model usually shows tighter residual cones at high predicted values. If the residuals widen with prediction in the raw model and not in the log model, log wins."
))

# ---------------------------------------------------------------------------
# SECTION 9: Hyperparameter tuning (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 9. Hyperparameter tuning with Optuna

We tune the best model from Section 7 with ~30 trials, optimising MAE on a
TimeSeriesSplit using train+val rows. Test stays untouched.""")

code("""def objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 200, 800),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        num_leaves       = trial.suggest_int('num_leaves', 16, 127),
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 200),
        subsample        = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha        = trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        reg_lambda       = trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
    )
    tscv = TimeSeriesSplit(n_splits=3)
    fold_mae = []
    X = np.concatenate([train[FEATURES].values, val[FEATURES].values])
    y = np.concatenate([train[TARGET].values, val[TARGET].values])
    for tr, va in tscv.split(X):
        m = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr])
        fold_mae.append(mean_absolute_error(y[va], m.predict(X[va])))
    return float(np.mean(fold_mae))

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=30, show_progress_bar=False)
print('best CV MAE:', study.best_value)
print('best params:', study.best_params)""")

code("""# trajectory of best-so-far value
trial_df = study.trials_dataframe()
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(trial_df['number'], trial_df['value'], 'o-', alpha=0.5, label='trial MAE')
ax.plot(trial_df['number'], trial_df['value'].cummin(), 'r-', lw=2, label='best so far')
ax.set_xlabel('trial'); ax.set_ylabel('CV MAE'); ax.set_title('Optuna trajectory')
ax.legend(); plt.tight_layout()""")

md("""### Exercises (Section 9)""")

md("""**Exercise 9.1** - Recreate the study with a `MedianPruner` so unpromising
trials are stopped early. Run only 10 trials to see pruning in action.""")
code("# Your answer here\n")
md(solution(
"""study2 = optuna.create_study(direction='minimize',
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0),
                             sampler=optuna.samplers.TPESampler(seed=SEED))

def obj_pruning(trial):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 200, 800),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int('num_leaves', 16, 127),
    )
    tscv = TimeSeriesSplit(n_splits=3)
    X = np.concatenate([train[FEATURES].values, val[FEATURES].values])
    y = np.concatenate([train[TARGET].values, val[TARGET].values])
    fold_mae = []
    for fold_idx, (tr, va) in enumerate(tscv.split(X)):
        m = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr])
        fold_mae.append(mean_absolute_error(y[va], m.predict(X[va])))
        trial.report(np.mean(fold_mae), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_mae))

study2.optimize(obj_pruning, n_trials=10, show_progress_bar=False)
print('best:', study2.best_value)
print('pruned trials:', sum(t.state == optuna.trial.TrialState.PRUNED for t in study2.trials))""",
"`MedianPruner` kills trials whose intermediate score is worse than the median at the same step. Combined with `trial.report()` inside the CV loop, it can give 2-5x speedup on deeper search spaces."
))

md("""**Exercise 9.2** - Extend the search space to include `num_leaves` and
`min_data_in_leaf` (the two LightGBM knobs that most directly control model
complexity) and rerun for 10 trials.""")
code("# Your answer here\n")
md(solution(
"""def obj_extra(trial):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 200, 800),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int('num_leaves', 8, 255),         # extended
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 5, 500),  # extended
    )
    X = np.concatenate([train[FEATURES].values, val[FEATURES].values])
    y = np.concatenate([train[TARGET].values, val[TARGET].values])
    fold_mae = []
    for tr, va in TimeSeriesSplit(n_splits=3).split(X):
        m = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr])
        fold_mae.append(mean_absolute_error(y[va], m.predict(X[va])))
    return float(np.mean(fold_mae))

study3 = optuna.create_study(direction='minimize')
study3.optimize(obj_extra, n_trials=10, show_progress_bar=False)
print('best:', study3.best_params)""",
"`num_leaves` and `min_data_in_leaf` jointly determine tree depth and overfitting. Wider ranges let the optimiser find the right complexity for your dataset size."
))

md("""**Exercise 9.3** - Save the original study to a SQLite store at
`artifacts/study.db` and reload it.""")
code("# Your answer here\n")
md(solution(
"""storage = f'sqlite:///{ARTIFACT_DIR}/study.db'
saved = optuna.create_study(direction='minimize',
                            study_name='lgbm_vol',
                            storage=storage,
                            load_if_exists=True)
# copy in completed trials from the original study
for t in study.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        saved.add_trial(t)
reloaded = optuna.load_study(study_name='lgbm_vol', storage=storage)
print('reloaded trials:', len(reloaded.trials), ' best:', reloaded.best_value)""",
"SQLite storage gives you persistence and lets multiple processes contribute trials in parallel. The `load_if_exists=True` pattern avoids duplicate-name errors on reruns."
))

md("""**Exercise 9.4** - Run a tiny multi-objective study (5 trials) optimising
(MAE, model complexity proxy = `n_estimators * num_leaves`) and print the
Pareto front.""")
code("# Your answer here\n")
md(solution(
"""def obj_mo(trial):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 100, 400),
        num_leaves=trial.suggest_int('num_leaves', 16, 127),
        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
    )
    X = np.concatenate([train[FEATURES].values, val[FEATURES].values])
    y = np.concatenate([train[TARGET].values, val[TARGET].values])
    fold_mae = []
    for tr, va in TimeSeriesSplit(n_splits=3).split(X):
        m = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr])
        fold_mae.append(mean_absolute_error(y[va], m.predict(X[va])))
    complexity = params['n_estimators'] * params['num_leaves']
    return float(np.mean(fold_mae)), complexity

study_mo = optuna.create_study(directions=['minimize', 'minimize'])
study_mo.optimize(obj_mo, n_trials=5, show_progress_bar=False)
front = [(t.values[0], t.values[1], t.params) for t in study_mo.best_trials]
for v in front:
    print(f'MAE={v[0]:.5f}  complexity={v[1]}  {v[2]}')""",
"Multi-objective optimisation returns a Pareto set rather than a single best -- you decide which point trades off accuracy and inference cost best for your deployment."
))

# ---------------------------------------------------------------------------
# SECTION 10: Feature importance (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 10. Feature importance (gain + SHAP)

We refit the tuned LGBM on train+val and inspect both:

- **Gain importance**: how much each feature reduced training loss.
- **SHAP values**: per-sample contribution to the prediction; gives a proper
  signed-magnitude view per feature.""")

code("""best_params = study.best_params
final_model = lgb.LGBMRegressor(**best_params, random_state=SEED, n_jobs=-1, verbose=-1)
X_fit = pd.concat([train[FEATURES], val[FEATURES]])
y_fit = pd.concat([train[TARGET], val[TARGET]])
final_model.fit(X_fit, y_fit)

gain = pd.DataFrame({'feature': FEATURES,
                     'gain': final_model.booster_.feature_importance(importance_type='gain')})
gain = gain.sort_values('gain', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(gain['feature'][::-1], gain['gain'][::-1])
ax.set_title('Top 15 features by gain (LightGBM)')
ax.set_xlabel('gain'); plt.tight_layout()""")

code("""# SHAP on a val sample for speed
sample = val[FEATURES].sample(min(500, len(val)), random_state=SEED)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(sample)

shap.summary_plot(shap_values, sample, plot_size=(9, 5), show=True, max_display=15)""")

md("""### Exercises (Section 10)""")

md("""**Exercise 10.1** - Compute permutation importance on the val set for the
tuned LGBM and merge it into a single table next to gain importance for the
top 10 by gain. Are the two rankings consistent?""")
code("# Your answer here\n")
md(solution(
"""pi = permutation_importance(final_model, val[FEATURES], val[TARGET],
                            n_repeats=5, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({'feature': FEATURES,
                    'gain': final_model.booster_.feature_importance(importance_type='gain'),
                    'perm': pi.importances_mean})
imp.sort_values('gain', ascending=False).head(10)""",
"Gain measures train-time impact; permutation measures held-out impact. Big gain but tiny permutation = the feature memorised noise. Aligned rankings = trustworthy importance."
))

md("""**Exercise 10.2** - Plot a SHAP **dependence plot** for the single most
important feature. Look at how the model uses it across its range.""")
code("# Your answer here\n")
md(solution(
"""top = gain.iloc[0]['feature']
shap.dependence_plot(top, shap_values, sample, show=True)""",
"Dependence plots show non-linearity (curve shape) and interactions (color encodes another feature). A monotone curve confirms the feature acts the way intuition predicts."
))

md("""**Exercise 10.3** - Drop the bottom 50% of features by gain importance,
refit on train+val, evaluate on val, and compare MAE to the full model.""")
code("# Your answer here\n")
md(solution(
"""imp_full = pd.DataFrame({'feature': FEATURES,
                         'gain': final_model.booster_.feature_importance(importance_type='gain')})
keep = imp_full.sort_values('gain', ascending=False).head(len(FEATURES)//2)['feature'].tolist()
m_small = lgb.LGBMRegressor(**best_params, random_state=SEED, n_jobs=-1, verbose=-1)
m_small.fit(train[keep], train[TARGET])
mae_small = mean_absolute_error(val[TARGET], m_small.predict(val[keep]))
mae_full  = mean_absolute_error(val[TARGET], final_model.predict(val[FEATURES]))
print(f'MAE full ({len(FEATURES)} feats): {mae_full:.5f}')
print(f'MAE half ({len(keep)} feats): {mae_small:.5f}')""",
"Dropping half the features and recovering similar MAE suggests strong feature redundancy. In production this means cheaper inference and lower overfit risk."
))

md("""**Exercise 10.4** - Pick a single test row, compute its SHAP values, and
plot a **waterfall** showing how each feature contributes to the prediction.""")
code("# Your answer here\n")
md(solution(
"""row_idx = 0
sample_row = test[FEATURES].iloc[[row_idx]]
exp = explainer(sample_row)
shap.plots.waterfall(exp[0], max_display=12, show=True)""",
"Waterfall plots are the per-prediction view: each bar pushes the base rate up or down, ending at the model's actual output. Critical for deployment-time explanations."
))

# ---------------------------------------------------------------------------
# SECTION 11: Diagnostics (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 11. Diagnostics

Four standard plots for a regression model:

1. Residuals vs predicted (homoskedasticity).
2. Residual ACF (autocorrelation = leakage or missing dynamics).
3. QQ plot (residual normality, less critical for MAE-driven training).
4. Predicted vs actual scatter with y=x reference.""")

code("""val_pred = final_model.predict(val[FEATURES])
resid = val[TARGET].values - val_pred

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0,0].scatter(val_pred, resid, s=4, alpha=0.4)
axes[0,0].axhline(0, color='k', ls='--')
axes[0,0].set_title('Residuals vs predicted')
axes[0,0].set_xlabel('predicted'); axes[0,0].set_ylabel('residual')

lags = np.arange(1, 49)
acf_r = [pd.Series(resid).autocorr(lag=l) for l in lags]
axes[0,1].bar(lags, acf_r, color='C2')
axes[0,1].axhline(0, color='k', lw=0.5)
axes[0,1].set_title('ACF of residuals')
axes[0,1].set_xlabel('lag (hours)')

stats.probplot(resid, dist='norm', plot=axes[1,0])
axes[1,0].set_title('QQ plot of residuals')

axes[1,1].scatter(val[TARGET], val_pred, s=4, alpha=0.4)
lim = [min(val[TARGET].min(), val_pred.min()), max(val[TARGET].max(), val_pred.max())]
axes[1,1].plot(lim, lim, 'k--')
axes[1,1].set_xlabel('actual'); axes[1,1].set_ylabel('predicted')
axes[1,1].set_title('Predicted vs actual')

plt.tight_layout()""")

md("""### Exercises (Section 11)""")

md("""**Exercise 11.1** - Run a **Ljung-Box** test on residuals (lags 12 and 24)
to formally check for residual autocorrelation. p-value < 0.05 = residuals are
correlated = model is missing dynamics.""")
code("# Your answer here\n")
md(solution(
"""from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
print(lb)""",
"Ljung-Box jointly tests whether the first k autocorrelations are zero. For vol forecasting we usually still see some structure -- if the p-value is tiny, you have an under-fitted model and should consider GARCH or a more flexible feature set."
))

md("""**Exercise 11.2** - Plot the rolling 24-row standard deviation of
residuals over time. If the line trends or jumps, residual variance is
non-stationary (heteroskedastic).""")
code("# Your answer here\n")
md(solution(
"""rs = pd.Series(resid, index=val.index).rolling(24).std()
fig, ax = plt.subplots(figsize=(11, 3))
rs.plot(ax=ax)
ax.set_title('Rolling 24h std of residuals')
ax.set_ylabel('residual std')
plt.tight_layout()""",
"Rising residual std during high-vol periods is normal for vol models. If it correlates with the absolute prediction, the variance scales with the level -- a hint that log-target (or quantile loss) might fit better."
))

md("""**Exercise 11.3** - Bucket val rows by **quintile of prediction** and
compute MAE per bucket. The model is well-calibrated by magnitude if MAE grows
proportionally with prediction.""")
code("# Your answer here\n")
md(solution(
"""df_q = pd.DataFrame({'pred': val_pred, 'true': val[TARGET].values})
df_q['bucket'] = pd.qcut(df_q['pred'], 5, labels=[f'Q{i}' for i in range(1,6)])
df_q.groupby('bucket').apply(
    lambda g: pd.Series({'n': len(g),
                          'mae': mean_absolute_error(g['true'], g['pred']),
                          'mean_pred': g['pred'].mean()})
)""",
"Bucketed MAE exposes whether the model fails uniformly or only in extreme regimes. High-vol buckets typically have larger absolute MAE but similar relative MAE."
))

md("""**Exercise 11.4** - Build a calibration plot: rank val predictions, sort
by rank, and plot empirical CDF of predicted vs empirical CDF of actual. A
well-calibrated model lies on the diagonal.""")
code("# Your answer here\n")
md(solution(
"""pred_q = np.quantile(val_pred, np.linspace(0, 1, 50))
true_q = np.quantile(val[TARGET].values, np.linspace(0, 1, 50))
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(pred_q, true_q, 'o-')
lim = [min(pred_q.min(), true_q.min()), max(pred_q.max(), true_q.max())]
ax.plot(lim, lim, 'k--')
ax.set_xlabel('predicted quantile'); ax.set_ylabel('actual quantile')
ax.set_title('Quantile-quantile calibration')
plt.tight_layout()""",
"Marginal QQ calibration ignores order but checks whether the predicted distribution matches the actual. Off-diagonal points at the tails = the model under/over-estimates extremes."
))

# ---------------------------------------------------------------------------
# SECTION 12: Final eval on TEST (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 12. Final evaluation on TEST

This is the **single** time we touch test. Refit the tuned model on train+val
and evaluate.""")

code("""test_pred = final_model.predict(test[FEATURES])
test_persist = test['har_d'].values

m_final  = metrics(test[TARGET].values, test_pred)
m_persist= metrics(test[TARGET].values, test_persist)

final_table = pd.DataFrame([
    {'model': 'tuned_LGBM',  **m_final},
    {'model': 'persistence', **m_persist},
]).round(5)
final_table""")

code("""fig, ax = plt.subplots(figsize=(11, 4))
test[TARGET].plot(ax=ax, label='actual', color='k', lw=1)
pd.Series(test_pred, index=test.index).plot(ax=ax, label='LGBM', color='C0', lw=1, alpha=0.8)
pd.Series(test_persist, index=test.index).plot(ax=ax, label='persistence', color='C3', lw=0.8, alpha=0.6)
ax.set_title('Test set: actual vs predicted 24h forward RV')
ax.set_ylabel('RV'); ax.legend()
plt.tight_layout()""")

md("""### Exercises (Section 12)""")

md("""**Exercise 12.1** - Compute test MAE per **calendar month**. Is performance
stable, or are some months much worse?""")
code("# Your answer here\n")
md(solution(
"""df_t = pd.DataFrame({'true': test[TARGET].values, 'pred': test_pred}, index=test.index)
df_t['month'] = df_t.index.to_period('M')
monthly = df_t.groupby('month').apply(
    lambda g: pd.Series({'n': len(g),
                          'mae': mean_absolute_error(g['true'], g['pred'])}))
monthly""",
"Per-month metrics expose regime-dependent failure. A model with stable MAE across months is far more deployable than one that is great on average and terrible during a single shock."
))

md("""**Exercise 12.2** - Compute the test MAE **only on rows in the top decile
of actual vol** (the tail). How much worse is tail MAE vs overall MAE?""")
code("# Your answer here\n")
md(solution(
"""thr = np.quantile(test[TARGET].values, 0.9)
mask = test[TARGET].values >= thr
mae_tail = mean_absolute_error(test[TARGET].values[mask], test_pred[mask])
mae_all  = mean_absolute_error(test[TARGET].values, test_pred)
print(f'overall MAE   : {mae_all:.5f}')
print(f'top-decile MAE: {mae_tail:.5f}  (n={mask.sum()})')""",
"Tail performance is what matters in risk applications -- a model that nails calm markets but blows up during stress is useless. Tail MAE is typically 3-5x overall MAE."
))

md("""**Exercise 12.3** - Build a side-by-side line plot of model vs persistence
predictions over the first 14 days of test, with actual overlaid.""")
code("# Your answer here\n")
md(solution(
"""window = test.iloc[:14*24]
pred_w  = pd.Series(final_model.predict(window[FEATURES]), index=window.index)
pers_w  = window['har_d']
fig, ax = plt.subplots(figsize=(11, 4))
window[TARGET].plot(ax=ax, color='k', label='actual', lw=1.5)
pred_w.plot(ax=ax, color='C0', label='LGBM', lw=1)
pers_w.plot(ax=ax, color='C3', label='persistence', lw=1, alpha=0.7)
ax.set_title('First 14 test days: actual vs predictions')
ax.legend(); plt.tight_layout()""",
"Visual inspection of a short window often reveals whether the model adds value at the right turning points or just trails persistence with a small offset."
))

md("""**Exercise 12.4** - Compute the percentage of test hours where the LGBM
absolute error is smaller than persistence's absolute error.""")
code("# Your answer here\n")
md(solution(
"""err_lgbm = np.abs(test[TARGET].values - test_pred)
err_pers = np.abs(test[TARGET].values - test_persist)
win_rate = (err_lgbm < err_pers).mean() * 100
print(f'LGBM beats persistence {win_rate:.1f}% of hours')""",
"A win rate just above 50% is realistic for vol forecasting. If you see 70%+ on val/test, double-check feature leakage."
))

# ---------------------------------------------------------------------------
# SECTION 13: Prediction intervals (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 13. Prediction intervals (quantile LightGBM)

Point estimates are not enough for risk. Train three LightGBM models with the
**quantile objective** at p10, p50, p90 to produce a prediction interval.
LightGBM's quantile loss:

$$
L_\\tau(y, \\hat y) = \\sum_i \\tau \\max(0, y_i - \\hat y_i) + (1-\\tau) \\max(0, \\hat y_i - y_i)
$$
""")

code("""def fit_quantile(alpha):
    m = lgb.LGBMRegressor(objective='quantile', alpha=alpha,
                          n_estimators=400, learning_rate=0.05,
                          num_leaves=31, random_state=SEED,
                          n_jobs=-1, verbose=-1)
    m.fit(X_fit, y_fit)
    return m

q_models = {a: fit_quantile(a) for a in [0.1, 0.5, 0.9]}
q_preds  = {a: m.predict(test[FEATURES]) for a, m in q_models.items()}

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(test.index, test[TARGET].values, 'k', lw=0.8, label='actual')
ax.plot(test.index, q_preds[0.5], 'C0', lw=0.8, label='p50')
ax.fill_between(test.index, q_preds[0.1], q_preds[0.9], color='C0', alpha=0.25, label='[p10, p90]')
ax.set_title('Test set fan chart')
ax.set_ylabel('RV'); ax.legend(); plt.tight_layout()""")

md("""### Exercises (Section 13)""")

md("""**Exercise 13.1** - Compute **empirical coverage** of the [p10, p90]
interval on test. Ideal coverage is 80%.""")
code("# Your answer here\n")
md(solution(
"""inside = (test[TARGET].values >= q_preds[0.1]) & (test[TARGET].values <= q_preds[0.9])
print(f'empirical coverage of [p10,p90]: {inside.mean()*100:.1f}% (target 80%)')""",
"Quantile regressors trained on train+val often under-cover on test because the test distribution is shifted. If coverage is 60% you need recalibration; if 95% your intervals are too wide."
))

md("""**Exercise 13.2** - Plot the **interval width** (p90 - p10) over time on
test. Does it widen during high-vol periods?""")
code("# Your answer here\n")
md(solution(
"""width = q_preds[0.9] - q_preds[0.1]
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(test.index, width, color='C2')
ax.set_title('Prediction interval width [p90-p10] over time')
ax.set_ylabel('width'); plt.tight_layout()""",
"A good vol model should have widening intervals when the regime is uncertain. Constant-width intervals suggest the model is not propagating regime information into uncertainty."
))

md("""**Exercise 13.3** - Implement **isotonic recalibration** of the p90
quantile on val: fit `IsotonicRegression` mapping val p90 predictions to
empirical CDF of val targets, then apply to test.""")
code("# Your answer here\n")
md(solution(
"""from sklearn.isotonic import IsotonicRegression
val_p90 = q_models[0.9].predict(val[FEATURES])
# empirical "alpha at this prediction" target = whether val truth is below it (we want 90% of mass below p90)
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(val_p90, (val[TARGET].values <= val_p90).astype(float))
adj_factor = ir.predict(q_preds[0.9])
# scale predicted p90 so we hit the desired coverage proportion
recal_p90 = q_preds[0.9] * (0.9 / np.clip(adj_factor.mean(), 0.5, 1.5))
print('original coverage:', ((test[TARGET].values <= q_preds[0.9]).mean()))
print('recal    coverage:', ((test[TARGET].values <= recal_p90).mean()))""",
"Isotonic recalibration is a non-parametric way to fix systematic mis-coverage. In production you re-fit the isotonic map on a rolling window of recent residuals."
))

md("""**Exercise 13.4** - Train p5 and p95 quantile models and compute coverage
of [p5, p95] (target 90%).""")
code("# Your answer here\n")
md(solution(
"""m05 = fit_quantile(0.05)
m95 = fit_quantile(0.95)
p05 = m05.predict(test[FEATURES])
p95 = m95.predict(test[FEATURES])
cov = ((test[TARGET].values >= p05) & (test[TARGET].values <= p95)).mean()
print(f'[p5, p95] empirical coverage: {cov*100:.1f}%  (target 90%)')""",
"Wider intervals will usually under-cover by more than narrow ones because the tails of the target distribution are harder to model. Always check both narrow and wide PIs."
))

# ---------------------------------------------------------------------------
# SECTION 14: Deployment (MAJOR, exercises)
# ---------------------------------------------------------------------------
md("""## 14. Deployment (joblib + FastAPI snippet)

Persist the **point-estimate model and the three quantile models** plus the
feature spec. Then a minimal FastAPI app exposes `/predict`.""")

code("""bundle = {
    'point': final_model,
    'quantiles': {a: m for a, m in q_models.items()},
    'features': FEATURES,
    'target': TARGET,
    'meta': {
        'trained_on': f'{train.index.min()} -> {val.index.max()}',
        'best_params': best_params,
    }
}
bundle_path = os.path.join(ARTIFACT_DIR, 'rv_model_bundle.joblib')
joblib.dump(bundle, bundle_path)
print('saved:', bundle_path, '->', os.path.getsize(bundle_path), 'bytes')""")

code("""# FastAPI snippet (do NOT launch here)
fastapi_code = '''
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

bundle = joblib.load("artifacts/rv_model_bundle.joblib")
features = bundle["features"]
point_model = bundle["point"]
q_models = bundle["quantiles"]

app = FastAPI()

class FeatureRow(BaseModel):
    values: dict   # {feature_name: float}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(row: FeatureRow):
    x = np.array([[row.values[f] for f in features]])
    return {
        "mean": float(point_model.predict(x)[0]),
        "p10":  float(q_models[0.1].predict(x)[0]),
        "p50":  float(q_models[0.5].predict(x)[0]),
        "p90":  float(q_models[0.9].predict(x)[0]),
    }
'''
print(fastapi_code)
print('# curl example:')
print('# curl -X POST localhost:8000/predict -H "Content-Type: application/json" \\\\')
print('#   -d \\'{"values": {"har_d": 0.4, "har_w": 0.42, ...}}\\'')""")

md("""### Exercises (Section 14)""")

md("""**Exercise 14.1** - Implement `predict_one(row_dict: dict) -> dict` that
loads the bundle, validates that all expected features are present, and returns
`{mean, p10, p50, p90}`.""")
code("# Your answer here\n")
md(solution(
"""def predict_one(row_dict, bundle_path=bundle_path):
    b = joblib.load(bundle_path)
    feats = b['features']
    missing = [f for f in feats if f not in row_dict]
    if missing:
        raise ValueError(f'missing features: {missing[:5]}...')
    x = np.array([[row_dict[f] for f in feats]])
    return {
        'mean': float(b['point'].predict(x)[0]),
        'p10':  float(b['quantiles'][0.1].predict(x)[0]),
        'p50':  float(b['quantiles'][0.5].predict(x)[0]),
        'p90':  float(b['quantiles'][0.9].predict(x)[0]),
    }

example = test[FEATURES].iloc[0].to_dict()
predict_one(example)""",
"This is the function your serving handler will call. Always validate the feature list explicitly so a missing feature raises early instead of silently being filled with 0."
))

md("""**Exercise 14.2** - Write a `pytest`-style assertion function that
verifies `p10 <= p50 <= p90` for a sample of test rows.""")
code("# Your answer here\n")
md(solution(
"""def test_quantile_ordering(n=50):
    rng = np.random.default_rng(SEED)
    sample = test[FEATURES].sample(n, random_state=SEED)
    for i in range(len(sample)):
        out = predict_one(sample.iloc[i].to_dict())
        assert out['p10'] <= out['p50'] <= out['p90'], (i, out)
    print(f'all {n} rows have monotone quantiles')

test_quantile_ordering()""",
"Independently-trained quantile models can cross. A monotonicity assertion is a cheap CI-time guard; if it fails you can sort the predictions or use a constrained quantile model."
))

md("""**Exercise 14.3** - Wrap the input schema in a Pydantic model that
validates each feature is a float.""")
code("# Your answer here\n")
md(solution(
"""from pydantic import BaseModel, create_model
fields = {f: (float, ...) for f in FEATURES}
PredictRequest = create_model('PredictRequest', **fields)
sample = test[FEATURES].iloc[0].to_dict()
req = PredictRequest(**sample)
print(req.dict())""",
"`pydantic.create_model` builds a schema dynamically from the feature list. In production you'd freeze the schema once after model training and ship it with the bundle."
))

md("""**Exercise 14.4** - Sketch a FastAPI `/health` endpoint that returns the
model's training window and number of features. Print the route definition --
do not launch the server.""")
code("# Your answer here\n")
md(solution(
"""health_code = '''
@app.get("/health")
def health():
    return {
        "status": "ok",
        "trained_on": bundle["meta"]["trained_on"],
        "n_features": len(bundle["features"]),
    }
'''
print(health_code)""",
"`/health` should be cheap and informative -- include enough metadata for an operator to spot a stale model without parsing logs."
))

# ---------------------------------------------------------------------------
# SECTION: Caveats / What's next (minor, no exercises)
# ---------------------------------------------------------------------------
md("""## 15. Caveats and what's next

What this notebook deliberately glossed over:

- **GARCH baselines**: the right next benchmark for hourly RV. `arch` package.
- **HAR-RV with realised semi-variances**: split positive/negative returns to
  capture leverage effects.
- **Microstructure noise**: hourly bars hide intra-hour vol; subsampling noise
  is not modelled here.
- **Regime drift**: vol regimes shift fast in crypto; for production you would
  retrain weekly with an expanding window.
- **Multi-step / multi-asset**: we predict 24h ahead for BTC only; jointly
  modelling all four symbols and longer horizons opens richer architectures
  (TFT, iTransformer).
- **Loss choice**: MAE is robust but symmetric; QLIKE or asymmetric loss is
  often more aligned with the cost of vol mis-prediction.

You now have the scaffolding for each step -- swap in any model, target, or
horizon and the structure stays the same.""")

# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
out = '/home/zlac116/Code/learning/ml-revision/regression/regression.ipynb'
nbf.write(nb, out)
print('written:', out, ' cells:', len(C))
