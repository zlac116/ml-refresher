"""
TOOLKIT — pandas task-indexed drill
====================================

OBJECTIVE
    Practise the 10 canonical pandas idioms from the cheatsheet:
    CSV/parquet I/O, row/column selection, sorting, missing data, GroupBy,
    rolling/expanding, resample, and long↔wide reshape.

ESTIMATED TIME
    60–90 min

TOPICS
    pd.read_csv / to_csv with parse_dates round-trip
    .loc time-slice, boolean masks, .query
    .sort_values multi-key
    .fillna / .dropna / .groupby(...).ffill
    .groupby + .agg / .transform
    .rolling, .expanding
    .resample with .mean / .last
    .pivot, .melt, .stack, .unstack

EXPECTED OUTPUT
    df shape:            (291, 3)
    csv round-trip rows: 5
    time-slice rows:     9
    boolean mask rows:   149
    top close by sym:    [101.8523, 102.7202, 103.8527]
    dropna / fillna:     289 / 291
    group means:         A=99.8926  B=100.0363  C=100.0325
    transform mean ≈ 0:  0.0
    rolling 5 last:      99.9773
    daily resample rows: 15
    wide shape:          (97, 3)
    melt rows:           291

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd
from io import StringIO

# ── synthetic 3-symbol hourly frame ────────────────────────────────────────
np.random.seed(42)
_dates = pd.date_range("2024-01-01", "2024-01-05", freq="1h")
_syms = ["A", "B", "C"]
_records = []
for s in _syms:
    for d in _dates:
        _records.append({"ts": d, "symbol": s, "close": 100 + np.random.normal(0, 1)})
df = pd.DataFrame(_records)


# ── TASK 1 — CSV round-trip ───────────────────────────────────────────────
def csv_roundtrip(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Write df.head(n) to an in-memory CSV (StringIO), read it back with
    parse_dates=['ts']. Return the read-back DataFrame.
    """
    # TODO: implement (hint: use StringIO + to_csv(buf, index=False) + read_csv)
    raise NotImplementedError


# ── TASK 2 — Selection (time-slice + boolean mask) ───────────────────────
def select_time_slice_and_mask(df: pd.DataFrame) -> tuple[int, int]:
    """Set the index to 'ts', sort it, then return:
        n_in_window : number of rows where the index is in
                      ['2024-01-01 00:00', '2024-01-01 02:00'] inclusive
        n_above_100 : number of rows where close > 100
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 — Sort multi-key ───────────────────────────────────────────────
def top_close_per_symbol(df: pd.DataFrame) -> list[float]:
    """Sort by [symbol asc, close desc]. Return the top close per symbol as a
    list (3 values, one per symbol in alphabetical order).
    """
    # TODO: implement (hint: sort_values then groupby('symbol').head(1))
    raise NotImplementedError


# ── TASK 4 — Missing data ──────────────────────────────────────────────────
def missing_data_summary(df: pd.DataFrame) -> dict[str, int]:
    """Inject two NaNs at rows 5 and 10 of df['close']. Return dict:
        n_after_dropna : len after .dropna()
        n_after_fillna : len after .fillna(0)
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 — GroupBy: aggregate ───────────────────────────────────────────
def group_close_mean(df: pd.DataFrame) -> pd.Series:
    """Return df.groupby('symbol')['close'].mean()."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 6 — GroupBy: transform (per-group demean) ───────────────────────
def per_symbol_centered(df: pd.DataFrame) -> pd.Series:
    """Subtract the per-symbol mean from each row's close. Returns a Series
    aligned with df's index.

    Use df.groupby('symbol')['close'].transform('mean') — NOT .apply, NOT merge.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 7 — Rolling window ──────────────────────────────────────────────
def rolling_mean_per_symbol(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Per-symbol 5-row rolling mean of close (sorted by ts).

    Returns a Series with a MultiIndex (symbol, ts).
    Use df.set_index('ts').groupby('symbol')['close'].rolling(window).mean().
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 8 — Resample to daily ────────────────────────────────────────────
def daily_mean_close(df: pd.DataFrame) -> pd.Series:
    """Resample to daily mean close per symbol. Returns a Series with MultiIndex
    (symbol, ts).

    Use .set_index('ts').groupby('symbol')['close'].resample('1D').mean().
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 9 — Reshape: long → wide ─────────────────────────────────────────
def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to (index=ts, columns=symbol, values=close)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 10 — Reshape: wide → long via melt ──────────────────────────────
def melt_back_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Reverse the pivot: melt back to long with columns ['ts', 'symbol', 'close'].
    Use wide.reset_index().melt(id_vars='ts', value_name='close').
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assert df.shape == (291, 3)

    rt = csv_roundtrip(df)
    assert len(rt) == 5

    n_win, n_above = select_time_slice_and_mask(df)
    assert n_win == 9, n_win
    assert n_above == 149, n_above

    top = top_close_per_symbol(df)
    assert len(top) == 3
    assert abs(top[0] - 101.8523) < 1e-3
    assert abs(top[1] - 102.7202) < 1e-3
    assert abs(top[2] - 103.8527) < 1e-3

    miss = missing_data_summary(df)
    assert miss["n_after_dropna"] == 289
    assert miss["n_after_fillna"] == 291

    g = group_close_mean(df)
    assert g.index.tolist() == ["A", "B", "C"]
    assert abs(g["A"] -  99.8926) < 1e-3
    assert abs(g["B"] - 100.0363) < 1e-3
    assert abs(g["C"] - 100.0325) < 1e-3

    centered = per_symbol_centered(df)
    assert len(centered) == len(df)
    assert abs(centered.mean()) < 1e-8     # within-group demean → grand mean ≈ 0

    rolled = rolling_mean_per_symbol(df)
    assert abs(rolled.iloc[-1] - 99.9773) < 1e-3

    daily = daily_mean_close(df)
    assert len(daily) == 15  # 5 days × 3 symbols

    wide = pivot_to_wide(df)
    assert wide.shape == (97, 3)

    long_back = melt_back_to_long(wide)
    assert len(long_back) == 291

    print(f"df shape:            {df.shape}")
    print(f"csv round-trip rows: {len(rt)}")
    print(f"time-slice rows:     {n_win}")
    print(f"boolean mask rows:   {n_above}")
    print(f"top close by sym:    {[round(t, 4) for t in top]}")
    print(f"dropna / fillna:     {miss['n_after_dropna']} / {miss['n_after_fillna']}")
    print(f"group means:         A={g['A']:.4f}  B={g['B']:.4f}  C={g['C']:.4f}")
    print(f"transform mean ≈ 0:  {abs(centered.mean()):.1e}")
    print(f"rolling 5 last:      {rolled.iloc[-1]:.4f}")
    print(f"daily resample rows: {len(daily)}")
    print(f"wide shape:          {wide.shape}")
    print(f"melt rows:           {len(long_back)}")
    print("\n✓ All checks passed.")
