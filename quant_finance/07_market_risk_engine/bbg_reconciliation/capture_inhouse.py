"""Capture step — run the in-house engine ONCE and freeze its output.

    python capture_inhouse.py

Drives recon/engine_adapter.collect_all() (which mirrors your base + product
class hierarchy) and writes data/inhouse/inhouse_canonical.csv — the frozen
snapshot the pytest reconciliation reads. Keeping capture SEPARATE from pytest is
deliberate: the messy engine loads all inputs on import; the test session should
never pay that cost or depend on its side effects.

Until the two seams (build_engine / apply_stress) are wired, this reports which
products are connected and leaves any existing snapshot untouched.
"""

from pathlib import Path

from recon import engine_adapter, schema

OUT = Path(__file__).resolve().parent / "data" / "inhouse" / "inhouse_canonical.csv"


def main() -> int:
    try:
        df = engine_adapter.collect_all()
    except NotImplementedError as e:
        print(f"[capture] engine not wired yet: {e}")
        print("[capture] fill build_engine/apply_stress in recon/engine_adapter.py.")
        print(f"[capture] existing snapshot left as-is: {OUT}")
        return 1

    if df.empty:
        print("[capture] no rows produced — add trades to config/trades.yaml.")
        return 1

    schema.validate_canonical(df, source="capture")
    df.to_csv(OUT, index=False)
    n_trades = df[schema.TRADE_ID].nunique()
    n_types = df[schema.TRADE_TYPE].nunique()
    print(f"[capture] wrote {len(df)} rows ({n_trades} trades, {n_types} types) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
