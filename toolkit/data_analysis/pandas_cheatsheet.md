# pandas Cheatsheet (2026)

Companion to `pandas_card.pdf`. This markdown is the **narrative** — mental
model + why-it-matters + worked I/O toy examples. The card is the **reference
lookup**.

Stack: **pandas ≥2.2 · Python ≥3.10 · matplotlib ≥3.8**.
Modern conventions: Copy-on-Write, PyArrow backend, method chaining, no `inplace=`.

---

## §0 — The general pattern

Every pandas script has the same 6 steps:

```
1. LOAD          read_parquet / read_csv / read_sql            (typed at read-time)
2. ENABLE CoW    pd.options.mode.copy_on_write = True          (pd 2.2+; default 3.x)
3. INSPECT       .info() .describe() .head() .isna().sum()     (know your data)
4. CLEAN         dtypes / index / missing / duplicates          (fail fast on errors)
5. TRANSFORM     .assign / .query / .groupby / .agg / .merge   (method chain)
6. PERSIST       to_parquet                                     (columnar > CSV)
```

The **method-chain habit** is the single biggest productivity gain in pandas.
Read chained code top-down, no intermediate variables, no `inplace=`. Every step
returns a new frame; you can `.pipe(debug_fn)` mid-chain to inspect.

---

## §1 — What Series and DataFrame really are

Both are **labelled numpy arrays** with an **index**. `Series` = 1D; `DataFrame`
= 2D (rows × columns). The index is what makes alignment automatic across
operations.

```python
import pandas as pd, numpy as np
pd.options.mode.copy_on_write = True

s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="score")
# a    10
# b    20
# c    30
# Name: score, dtype: int64

df = pd.DataFrame({
    "score":  [10, 20, 30],
    "region": ["EU", "US", "EU"],
}, index=["a", "b", "c"])
```

**Alignment**: any operation between two Series aligns on index (like a SQL
join). Missing labels → NaN.

```python
a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20], index=["y", "z"])
a + b
# x     NaN     ← no match
# y    12.0
# z    23.0
# dtype: float64
```

---

## §2 — Selection: `.loc` vs `.iloc` vs `.query`

The **three access patterns** you must internalise:

```python
df.loc["row_a", "col_x"]         # LABEL-based (inclusive slicing)
df.iloc[0, 2]                     # POSITION-based (Python semantics; end-exclusive)
df.query("score > 15")            # STRING expression (returns filtered view)
```

**Toy I/O**:

```python
df = pd.DataFrame({
    "score":  [10, 20, 30, 40],
    "region": ["EU", "US", "EU", "APAC"],
}, index=["a", "b", "c", "d"])

df.loc["b":"c"]                   # 2 rows (label slice INCLUDES "c")
#    score region
# b     20     US
# c     30     EU

df.iloc[1:3]                       # 2 rows (position slice EXCLUDES 3)
# same result as above

df.query("region in ['EU', 'US'] and score > 15")
#    score region
# b     20     US
# c     30     EU
```

**The chained-assignment trap** (pandas 3.x makes it silent):

```python
# ❌ Fails silently in pandas 3.x
df["score"][df["region"] == "EU"] = 0

# ✅ Correct
df.loc[df["region"] == "EU", "score"] = 0
```

---

## §3 — groupby: the single most useful operation

**Mental model**: split → apply → combine. Split rows by key(s), apply a
function to each group, combine back into a single result.

```python
sales = pd.DataFrame({
    "region":   ["EU", "US", "EU", "US", "EU"],
    "product":  ["A",  "B",  "A",  "A",  "B"],
    "revenue":  [100, 200, 150, 300, 120],
    "qty":      [10,  15,  12,  20,  8],
})

# Named aggregations (canonical — clear output column names)
sales.groupby("region").agg(
    total_rev=("revenue", "sum"),
    avg_qty=("qty", "mean"),
    n=("revenue", "size"),
)
#         total_rev  avg_qty  n
# region
# EU            370  10.0     3
# US            500  17.5     2
```

**transform** — return a same-length series aligned back to the ORIGINAL index:

```python
sales["rev_share_of_region"] = (
    sales["revenue"] / sales.groupby("region")["revenue"].transform("sum")
)
#    region product  revenue  qty  rev_share_of_region
# 0     EU       A      100   10             0.270270
# 1     US       B      200   15             0.400000
# 2     EU       A      150   12             0.405405
# 3     US       A      300   20             0.600000
# 4     EU       B      120    8             0.324324
```

**Key setting**: always pass `observed=True` when grouping by a **category**
dtype (default becomes True in pandas 3.x; without it you get rows for
unused-category × unused-category combinations, potentially exploding memory).

---

## §4 — Reshape: pivot / melt / stack / unstack

Wide (one row per entity, many columns) vs Long (one row per observation).
Every dataset is one or the other; you'll convert between them constantly.

```python
# Long → Wide (pivot_table handles duplicates; pivot doesn't)
long = pd.DataFrame({
    "date":   ["2026-01", "2026-01", "2026-02", "2026-02"],
    "ticker": ["AAPL",    "MSFT",    "AAPL",    "MSFT"],
    "close":  [180,       310,       190,       320],
})

wide = long.pivot_table(index="date", columns="ticker", values="close",
                         aggfunc="mean", observed=True)
# ticker    AAPL  MSFT
# date
# 2026-01    180   310
# 2026-02    190   320

# Wide → Long
back = wide.reset_index().melt(id_vars="date", var_name="ticker", value_name="close")
```

---

## §5 — merge / concat / merge_asof

Same as SQL joins.

```python
orders = pd.DataFrame({"cust_id": [1, 2, 3], "amount": [100, 200, 300]})
customers = pd.DataFrame({"cust_id": [1, 2, 4], "name": ["Alice", "Bob", "Carol"]})

pd.merge(orders, customers, on="cust_id", how="left")
#    cust_id  amount   name
# 0        1     100  Alice
# 1        2     200    Bob
# 2        3     300    NaN     ← left join keeps orders row 3

pd.merge(orders, customers, on="cust_id", how="outer", indicator=True)
#    cust_id  amount   name      _merge
# 0        1   100.0  Alice        both
# 1        2   200.0    Bob        both
# 2        3   300.0    NaN   left_only
# 3        4     NaN  Carol  right_only
```

**merge_asof** — the killer feature for time-series (match each trade to the
most recent quote):

```python
trades = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01 09:00:00",
                                              "2026-01-01 09:00:10"]),
                        "ticker": ["AAPL", "AAPL"], "px": [180.0, 180.5]})
quotes = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01 08:59:55",
                                              "2026-01-01 09:00:05"]),
                        "ticker": ["AAPL", "AAPL"], "bid": [179.9, 180.4]})

pd.merge_asof(trades.sort_values("ts"), quotes.sort_values("ts"),
               on="ts", by="ticker", direction="backward",
               tolerance=pd.Timedelta("30s"))
# Matches each trade to the most-recent quote ≤ trade time within 30s
```

---

## §6 — Time-series

**Always** set a `DatetimeIndex`, sort, and declare frequency:

```python
df = pd.read_csv("prices.csv", parse_dates=["dt"])
df = df.set_index("dt").sort_index().asfreq("D")
```

**Resample** (change frequency):

```python
df["px"].resample("D").last()                # daily close (last observation)
df["px"].resample("1W-FRI").agg(["first","max","min","last"])  # weekly OHLC (Fri close)
```

**Rolling** (windowed statistics — beware leakage):

```python
# For descriptive stats (use current row):
df["ma7"] = df["px"].rolling(7).mean()

# For features (predicting future — MUST shift first):
df["ma7_lag"] = df["px"].shift(1).rolling(7).mean()   # ← the leakage-safe form
```

---

## §7 — Missing data

pandas encodes missingness three ways:

| Type | Missing marker | Where |
|---|---|---|
| Numeric (numpy) | `NaN` | float columns |
| Numeric (nullable) | `pd.NA` | `Int64`, `Float64`, `boolean` dtypes |
| Datetime | `NaT` | `datetime64[ns]` |
| String (Arrow-backed) | `pd.NA` | `string[pyarrow]` |

```python
df.isna().sum()                    # per-column count
df["x"] = df["x"].astype("Int64")  # capital-I nullable int (keeps NA)
df.ffill(limit=3); df.bfill()      # method= is deprecated; use direct methods
df.interpolate(method="time")      # for DatetimeIndex
```

---

## §8 — IO: parquet first, always

**Parquet** stores dtypes, is columnar (column-pruning speeds up reads
massively), compressed by default, and much faster than CSV:

```python
df.to_parquet("out.parquet", engine="pyarrow", compression="snappy")
# Read only two columns from a 100-column file:
pd.read_parquet("out.parquet", columns=["date", "close"])   # ~100× faster than CSV
```

CSV is fine for interchange, human review, small files. Always specify dtypes
and dates at read-time — inference is slow and error-prone:

```python
pd.read_csv("data.csv",
    parse_dates=["dt"],
    dtype={"id": "int32", "region": "category", "score": "float32"},
    dtype_backend="pyarrow")        # modern: PyArrow-backed dtypes
```

---

## §9 — Method chaining (the daily productivity habit)

A pandas expression that would need 6 intermediate variables becomes one
readable pipeline:

```python
top_regions_2026 = (
    df
    .query("year == 2026 and status == 'settled'")
    .assign(quarter=lambda d: d["dt"].dt.quarter,
             gross=lambda d: d["amount"] * d["qty"])
    .groupby(["quarter", "region"], observed=True)
    .agg(total=("gross", "sum"), n=("gross", "size"))
    .reset_index()
    .sort_values(["quarter", "total"], ascending=[True, False])
    .groupby("quarter", observed=True).head(3)
)
```

Debug mid-chain with `.pipe()`:

```python
def peek(d, label):
    print(f"{label}: shape={d.shape}, cols={list(d.columns)}")
    return d

out = (df
    .query("...")
    .pipe(peek, "after query")
    .groupby("g", observed=True).sum()
    .pipe(peek, "after agg"))
```

---

## §10 — matplotlib: the fig/ax pattern

The **only** matplotlib pattern to use:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["y"], label="observed", lw=1.2)
ax.plot(df.index, df["y_hat"], label="model", ls="--")
ax.set(xlabel="date", ylabel="value", title="Fit vs actual")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("out.png", dpi=150)
plt.close(fig)                    # important — keeps memory under control
```

**Why**: `plt.plot(...)` uses hidden global state, breaks in Jupyter, breaks
under parallel rendering. `fig, ax = plt.subplots()` is explicit and works
everywhere.

pandas has a `.plot()` shortcut that wraps matplotlib for quick EDA:

```python
df["y"].plot(kind="line", ax=ax)
df.plot.scatter(x="x", y="y", c="region", cmap="tab10")
df.hist(bins=30, figsize=(12, 8))
```

Use `.plot()` for quick looks; use `fig, ax = plt.subplots()` for anything you
save or share.

---

## §11 — Top traps

See the card's traps section for the full 15-item list. The most-common ones:

1. **Chained assignment** silently fails in pd 3.x — always `df.loc[...]`.
2. **`fillna(method="ffill")`** deprecated — use `df.ffill()`.
3. **`"M"` freq** deprecated — use `"ME"` (month-end) or `"MS"` (month-start).
4. **`"H"` hour** deprecated — lowercase `"h"`.
5. **Rolling leakage** — `df["y"].rolling(7).mean()` uses current row. Features need `.shift(1).rolling(7)`.
6. **Groupby on category without `observed=True`** — memory explosion.
7. **`inplace=True`** — inconsistent, no perf win, disables chaining.
8. **`df.values`** — legacy. Use `df.to_numpy()`.

---

## §12 — Cross-references

- Card (dense reference): `pandas_card.pdf` in this folder.
- Drills (hands-on practice): `../data_analysis/pandas_drill.py` and `pandas_project.py`.
- Companion: `toolkit/data_analysis/numpy_cheatsheet.md` + `numpy_card.pdf` — numeric core.
- Downstream: every ML subject card (`ml/01_regression/regression_card.pdf` etc.) uses pandas.
