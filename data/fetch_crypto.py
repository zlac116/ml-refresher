"""Fetch hourly OHLCV for several crypto pairs from Binance via ccxt and save to parquet.

Idempotent: if the parquet exists and is not stale, exit. Otherwise refetch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "crypto_hourly.parquet"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TIMEFRAME = "1h"
LOOKBACK_DAYS = 730  # ~2 years


def fetch_symbol(ex: ccxt.Exchange, symbol: str, since_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cursor = since_ms
    limit = 1000
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if len(batch) < limit:
            break
        cursor = last_ts + 1
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["symbol"] = symbol.replace("/USDT", "")
    return df


def main() -> int:
    if OUT.exists():
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(OUT.stat().st_mtime, tz=timezone.utc)
        if age < timedelta(days=7):
            print(f"[fetch_crypto] {OUT} exists and is {age.days}d old. Skipping.")
            return 0

    ex = ccxt.binance({"enableRateLimit": True})
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)

    frames = []
    for sym in SYMBOLS:
        print(f"[fetch_crypto] fetching {sym} ...")
        df = fetch_symbol(ex, sym, since_ms)
        print(f"  rows={len(df)}  range={df['ts'].min()} -> {df['ts'].max()}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"[fetch_crypto] wrote {OUT}  rows={len(out)}  size={OUT.stat().st_size/1e6:.2f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
