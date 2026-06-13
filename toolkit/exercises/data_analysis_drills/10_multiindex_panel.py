"""
DRILL 10 — MultiIndex Panel Operations
======================================

OBJECTIVE
    On a (date, ticker) MultiIndex frame: select one date, xs by ticker,
    unstack to wide form, stack back, and use IndexSlice to select a
    rectangular sub-panel.

ESTIMATED TIME
    20 min

TOPICS
    pd.MultiIndex.from_product
    .xs(key, level=...) — drop a level by selecting a single label
    .unstack(level) / .stack() — pivot a level between rows/columns
    pd.IndexSlice + .loc[idx[…], :] for cross-cuts

EXPECTED OUTPUT
    panel shape:          (1250, 1)
    levels:               2
    one-date rows:        5
    AAA xs rows:          250
    wide shape:           (250, 5)
    wide AAA last:        0.006575
    restacked length:     1250
    slab rows:            12   (6 dates x 2 tickers)

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
_tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
_n = 250
_dates = pd.bdate_range("2024-01-01", periods=_n)
_sectors = {"AAA": "Tech", "BBB": "Tech", "CCC": "Fin", "DDD": "Fin", "EEE": "Energy"}

_records = []
for t in _tickers:
    rr = np.random.normal(0.0005, 0.015, _n)
    for d, r in zip(_dates, rr):
        _records.append({"date": d, "ticker": t, "ret": r})
flat = pd.DataFrame(_records)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def to_panel(flat: pd.DataFrame) -> pd.DataFrame:
    """Return flat reindexed by MultiIndex (date, ticker), kept sorted.

    Columns: ['ret']. The MultiIndex must have names ('date', 'ticker').
    """
    # TODO: implement (hint: .set_index([...]).sort_index())
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def one_date(panel: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    """Return all rows for `when`. Use .loc[(when, slice(None))]."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Unstack the ticker level so columns are tickers, rows are dates.

    Returns a DataFrame with shape (n_dates, n_tickers).
    """
    # TODO: implement (hint: panel['ret'].unstack('ticker'))
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def rectangular_slab(panel: pd.DataFrame, start_date: pd.Timestamp,
                     end_date: pd.Timestamp, tickers: list[str]) -> pd.DataFrame:
    """Use pd.IndexSlice to select rows where date in [start, end] AND
    ticker in `tickers`.
    """
    # TODO: implement (hint: idx=pd.IndexSlice; panel.loc[idx[a:b, tickers], :])
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    panel = to_panel(flat)
    assert panel.shape == (1250, 1), panel.shape
    assert panel.index.nlevels == 2
    assert list(panel.index.names) == ["date", "ticker"]

    od = one_date(panel, _dates[10])
    assert len(od) == 5

    aaa = panel.xs("AAA", level="ticker")
    assert len(aaa) == 250

    wide = to_wide(panel)
    assert wide.shape == (250, 5)
    assert abs(wide["AAA"].iloc[-1] - 0.006575) < 1e-4
    # Round-trip: stack back to long
    restacked = wide.stack()
    assert len(restacked) == 1250

    slab = rectangular_slab(panel, _dates[5], _dates[10], ["AAA", "BBB"])
    assert len(slab) == 12, len(slab)

    print(f"panel shape:          {panel.shape}")
    print(f"levels:               {panel.index.nlevels}")
    print(f"one-date rows:        {len(od)}")
    print(f"AAA xs rows:          {len(aaa)}")
    print(f"wide shape:           {wide.shape}")
    print(f"wide AAA last:        {wide['AAA'].iloc[-1]:.6f}")
    print(f"restacked length:     {len(restacked)}")
    print(f"slab rows:            {len(slab)}")
    print("\n✓ All checks passed.")
