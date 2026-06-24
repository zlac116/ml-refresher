"""
PROJECT — pandas: Cross-Sectional Crypto Feature Factory
=========================================================

OBJECTIVE
    On the BTC/ETH/SOL/BNB hourly dataset, build the panel-style features that
    a crypto cross-sectional strategy needs:

      1. Continuity validation — count gaps per symbol.
      2. Per-symbol features — 24h trailing return + 24h forward return.
      3. Cross-sectional ranks — at every timestamp, rank symbols 1..4 by 24h ret.
      4. Multi-horizon resample — hourly → daily closes per symbol.
      5. Regime tagging + conditional analysis on BTC.

ESTIMATED TIME
    30 min

TOPICS
    groupby + transform (avoids merge gymnastics)
    pct_change with shift for forward-looking targets
    pivot vs unstack for wide ↔ long
    resample + last for daily reduction
    Conditional means via boolean masks

EXPECTED OUTPUT
    rows / symbols:        70080 / 4
    gaps per symbol (all): 0
    fwd_24h notna:         69984      (last 24h per symbol have no target)
    xs_rank max:           4
    daily rows:            2924
    BTC vol median:        0.004172
    hi-vol forward mean:   0.1802 %
    lo-vol forward mean:  -0.0815 %

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

DATA = "/home/zlac116/Code/learning/ml-revision/data/crypto_hourly.parquet"
df = pd.read_parquet(DATA).sort_values(["symbol", "ts"]).reset_index(drop=True)


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def count_gaps_per_symbol(df: pd.DataFrame) -> dict[str, int]:
    """For each symbol, count how many consecutive timestamps are NOT exactly 1h apart.

    Returns a dict {symbol: gap_count}. Should be 0 for all symbols in clean data.
    Hint: groupby('symbol')['ts'].diff() then count != Timedelta('1h').
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add two columns:
        ret_24h_trailing  = close / close.shift(24) - 1     (per symbol, backward looking)
        fwd_24h           = close.shift(-24) / close - 1    (per symbol, forward looking)

    Use groupby('symbol')['close'].transform(...).
    Returns the SAME df with two new columns.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def cross_sectional_rank(df: pd.DataFrame) -> pd.Series:
    """At every timestamp, rank the 4 symbols by their ret_24h_trailing.

    Returns a Series aligned with df's index. Values are 1.0 (worst) ... 4.0 (best).
    Hint: groupby('ts')['ret_24h_trailing'].rank().
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def daily_close(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to daily last-close per symbol.

    Returns a long DataFrame with columns ['ts', 'symbol', 'close'].
    Use .set_index('ts').groupby('symbol')['close'].resample('1D').last().
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def btc_regime_returns(df: pd.DataFrame) -> tuple[float, float]:
    """Tag BTC hours into hi-vol / lo-vol by trailing 24h std of pct_change.
    Threshold = median trailing vol over the whole sample.

    Returns (mean_fwd_24h_hi, mean_fwd_24h_lo) — both as decimals.
    Forward return = close.pct_change(24).shift(-24).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assert df.shape == (70080, 7)

    gaps = count_gaps_per_symbol(df)
    assert set(gaps.keys()) == {"BNB", "BTC", "ETH", "SOL"}
    assert all(g == 0 for g in gaps.values()), gaps

    df = add_returns(df)
    assert {"ret_24h_trailing", "fwd_24h"}.issubset(df.columns)
    assert df["fwd_24h"].notna().sum() == 69984

    rk = cross_sectional_rank(df)
    assert rk.max() == 4.0 and rk.min() == 1.0
    assert len(rk) == len(df)

    daily = daily_close(df)
    assert len(daily) == 2924

    hi, lo = btc_regime_returns(df)
    assert abs(hi - 0.001802) < 1e-4
    assert abs(lo - -0.000815) < 1e-4
    assert hi > lo  # high-vol regime tends to mean-revert UP in this sample

    print(f"rows / symbols:        {len(df)} / {df['symbol'].nunique()}")
    print(f"gaps per symbol (all): {sum(gaps.values())}")
    print(f"fwd_24h notna:         {df['fwd_24h'].notna().sum()}      (last 24h per symbol have no target)")
    print(f"xs_rank max:           {int(rk.max())}")
    print(f"daily rows:            {len(daily)}")
    print(f"hi-vol forward mean:   {hi * 100:.4f} %")
    print(f"lo-vol forward mean:  {lo * 100:.4f} %")
    print("\n✓ All checks passed.")
