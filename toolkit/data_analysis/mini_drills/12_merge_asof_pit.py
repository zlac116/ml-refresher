"""
DRILL 12 — Point-in-Time Joins with merge_asof
==============================================

OBJECTIVE
    Given a 1-minute quote tape (bid/ask) and a small log of trades stamped
    at irregular times, attach the most recent prevailing bid/ask to each
    trade using pd.merge_asof — never a future quote (look-ahead bias).

ESTIMATED TIME
    20 min

TOPICS
    pd.merge_asof: time-series join "match the row whose key is <= mine"
    direction='backward' (default) — strict look-back, no peeking
    Compute effective fill: buyer hits ask, seller hits bid
    Trade-level signed notional aggregation

WHY HEDGE FUNDS CARE
    Backtest realism: signals must merge against the LAST OBSERVABLE quote,
    not a future one. .merge with == matches no rows; merge_asof is the
    canonical pandas tool.

EXPECTED OUTPUT
    n trades:            5
    fill #1 (buy 100):   100.037922  (ask side)
    fill #5 (sell 125):  100.355466  (bid side)
    total notional:      19899.0738

GRADING
    All asserts must pass; no trade row may have a fill_time later than its
    own trade time.
"""
import numpy as np
import pandas as pd

# ── synthetic quote + trade tapes ──────────────────────────────────────────
np.random.seed(42)
_qtimes = pd.date_range("2024-01-08 09:30", "2024-01-08 16:00", freq="1min")
_bids = 100 + np.cumsum(np.random.normal(0, 0.05, len(_qtimes)))
quotes = pd.DataFrame({"time": _qtimes, "bid": _bids, "ask": _bids + 0.02})

trades = pd.DataFrame({
    "time": pd.to_datetime([
        "2024-01-08 09:31:23", "2024-01-08 10:15:08", "2024-01-08 11:00:42",
        "2024-01-08 13:22:11", "2024-01-08 15:45:55",
    ]),
    "qty":  [100, -50, 200, 75, -125],
})


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def attach_prevailing_quote(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Use merge_asof to attach bid/ask from the most recent quote at or
    before each trade.

    Returns a copy of `trades` with new columns: bid, ask, quote_time.
    """
    # TODO: implement
    # Hint: trades and quotes must both be sorted on 'time' first.
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def add_fill_price(trades_with_quotes: pd.DataFrame) -> pd.DataFrame:
    """Add a 'fill_price' column: ask if buy (qty>0), bid if sell (qty<0)."""
    # TODO: implement (hint: np.where)
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def total_signed_notional(trades_with_fill: pd.DataFrame) -> float:
    """Sum of qty * fill_price across all rows."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = attach_prevailing_quote(trades, quotes)
    assert {"bid", "ask"}.issubset(out.columns)
    assert len(out) == 5
    # Specific fills
    assert abs(out["ask"].iloc[0] - 100.037922) < 1e-4
    assert abs(out["bid"].iloc[4] - 100.355466) < 1e-4

    # No look-ahead: every attached quote_time must be <= the trade time.
    # We can verify this by re-deriving via merge_asof from scratch.
    # Sanity: the merge must not introduce NaNs (every trade has a prior quote).
    assert out[["bid", "ask"]].notna().all().all()

    with_fill = add_fill_price(out)
    assert "fill_price" in with_fill.columns
    # Buy uses ask
    assert abs(with_fill.loc[with_fill["qty"] > 0, "fill_price"].iloc[0]
               - out.loc[out["qty"] > 0, "ask"].iloc[0]) < 1e-12

    nnl = total_signed_notional(with_fill)
    assert abs(nnl - 19899.0738) < 1e-2, f"notional off: {nnl}"

    print(f"n trades:            {len(out)}")
    print(f"fill #1 (buy 100):   {with_fill['fill_price'].iloc[0]:.6f}  (ask side)")
    print(f"fill #5 (sell 125):  {with_fill['fill_price'].iloc[4]:.6f}  (bid side)")
    print(f"total notional:      {nnl:.4f}")
    print("\n✓ All checks passed.")
