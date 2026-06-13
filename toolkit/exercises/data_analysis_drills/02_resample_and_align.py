"""
DRILL 2 — Resample and Align
============================

OBJECTIVE
    Take a 5-minute intraday price series, build a daily OHLC frame,
    then reindex the daily close onto a strict business-day calendar
    with forward-fill for gaps.

ESTIMATED TIME
    15 min

TOPICS
    pandas.Series.resample (.first/.last/.max/.min)
    pandas.bdate_range, .reindex, .ffill

EXPECTED OUTPUT
    intraday rows:       1807
    daily close len:     7
    day 1 OHLC:          100.05 / 100.45 / 98.65 / 99.10
    bday-reindexed len:  5
    last bday close:     108.29

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
idx = pd.date_range("2024-01-02 09:30", "2024-01-08 16:00", freq="5min")
intraday = pd.Series(np.cumsum(np.random.normal(0, 0.1, len(idx))) + 100.0, index=idx, name="px")


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def to_daily_ohlc(s: pd.Series) -> pd.DataFrame:
    """Resample to 1-day OHLC. Drop calendar days with no observations.

    Returns a DataFrame with columns ['open','high','low','close'] indexed
    by date.
    """
    # TODO: implement (hint: resample('1D') + .agg or call .first/.max/.min/.last)
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def to_business_days(close: pd.Series) -> pd.Series:
    """Reindex `close` onto a strict business-day calendar from min to max date,
    forward-filling any missing days.
    """
    # TODO: implement (hint: pd.bdate_range + .reindex + .ffill)
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ohlc = to_daily_ohlc(intraday)
    assert list(ohlc.columns) == ["open", "high", "low", "close"], ohlc.columns.tolist()
    assert len(ohlc) == 7
    d1 = ohlc.iloc[0]
    assert abs(d1["open"]  - 100.0497) < 1e-3
    assert abs(d1["high"]  - 100.4481) < 1e-3
    assert abs(d1["low"]   -  98.6473) < 1e-3
    assert abs(d1["close"] -  99.0989) < 1e-3
    # High >= max(open, close) and low <= min(open, close)
    assert (ohlc["high"] >= ohlc[["open", "close"]].max(axis=1)).all()
    assert (ohlc["low"]  <= ohlc[["open", "close"]].min(axis=1)).all()

    bday = to_business_days(ohlc["close"])
    assert len(bday) == 5, f"expected 5 business days, got {len(bday)}"
    assert abs(bday.iloc[-1] - 108.2919) < 1e-3
    assert not bday.isna().any()

    print(f"intraday rows:       {len(intraday)}")
    print(f"daily close len:     {len(ohlc)}")
    print(f"day 1 OHLC:          {d1['open']:.2f} / {d1['high']:.2f} / "
          f"{d1['low']:.2f} / {d1['close']:.2f}")
    print(f"bday-reindexed len:  {len(bday)}")
    print(f"last bday close:     {bday.iloc[-1]:.2f}")
    print("\n✓ All checks passed.")
