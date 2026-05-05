"""Generate the synthetic portfolio + 5 days of market data + B&R marks.

Run from the project root:
    python scripts/generate_data.py

Produces:
    data/trades.parquet
    data/market_data/curves_D{1..5}.parquet
    data/market_data/swaption_cube_D{1..5}.parquet
    data/market_data/cap_vols_D{1..5}.parquet
    data/market_data/credit_spreads_D{1..5}.parquet
    data/br_marks/br_D{1..5}.parquet

Day moves engineered:
- D1: normal
- D2: bull-steepener (front -10bp, long -2bp)
- D3: vol blowout (+3 vol points across the cube)
- D4: parallel +25bp crisis
- D5: partial reversion to D1

All seeded for reproducibility (np.random.default_rng(42)).
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trades import (
    IRS,
    EuropeanSwaption,
    BermudanSwaption,
    VanillaBond,
    CallableBond,
    TARN,
    RangeAccrual,
)
from src.curves import Curve
from src.pricers.irs import price_irs
from src.pricers.bond import price_vanilla_bond
from src.pricers.black_76 import price_european_swaption


# ------------------------------------------------------------------
# Trade book generator
# ------------------------------------------------------------------
def generate_trades(seed: int = 42) -> list:
    """Generate ~200 trades with realistic concentration mix."""
    rng = np.random.default_rng(seed)
    trades = []

    # 70 vanilla IRS (USD/EUR mix, 2y-30y maturities)
    for i in range(70):
        ccy = rng.choice(["USD", "EUR"], p=[0.65, 0.35])
        T_end = float(rng.choice([2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
                                 p=[0.10, 0.10, 0.20, 0.10, 0.25, 0.10, 0.10, 0.05]))
        notional = float(rng.choice([5e6, 25e6, 50e6, 100e6, 250e6, 500e6],
                                    p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05]))
        pay_receive = rng.choice(["payer", "receiver"])
        # Strike near par with a small offset
        K_par_proxy = 0.040 - 0.005 * (T_end > 10)
        strike = K_par_proxy + rng.uniform(-0.005, 0.005)
        fixed_dcc = "act/360" if ccy == "USD" else "30/360"
        trades.append(IRS(
            trade_id=f"IRS_{i+1:03d}",
            notional=notional,
            currency=ccy,
            pay_receive=pay_receive,
            fixed_K=round(strike, 6),
            T_start=0.0,
            T_end=T_end,
            pay_freq=1.0,
            fixed_dcc=fixed_dcc,
            float_dcc="act/360",
        ))

    # 50 European swaptions (1y×5y, 1y×10y, 5y×10y, etc.)
    for i in range(50):
        ccy = "USD"
        T_e = float(rng.choice([1.0, 2.0, 5.0], p=[0.5, 0.3, 0.2]))
        tail = float(rng.choice([5.0, 10.0, 20.0], p=[0.5, 0.4, 0.1]))
        swap_T_end = T_e + tail
        notional = float(rng.choice([10e6, 25e6, 50e6, 100e6], p=[0.30, 0.30, 0.30, 0.10]))
        pay_receive = rng.choice(["payer", "receiver"])
        K_par_proxy = 0.040
        # ATM ± 50bp
        strike = K_par_proxy + rng.choice([-0.005, 0.0, 0.005])
        trades.append(EuropeanSwaption(
            trade_id=f"SWPT_{i+1:03d}",
            notional=notional,
            currency=ccy,
            pay_receive=pay_receive,
            strike_K=round(strike, 6),
            T_e=T_e,
            swap_T_end=swap_T_end,
            pay_freq=1.0,
        ))

    # 20 Bermudan swaptions
    for i in range(20):
        ccy = "USD"
        # Underlying: 5y or 10y swap, exercise dates quarterly or annual
        first_ex = float(rng.choice([1.0, 2.0]))
        tail = float(rng.choice([5.0, 10.0]))
        swap_T_end = first_ex + tail
        # Quarterly or annual exercise dates from first_ex up to (swap_T_end - 1y)
        last_ex = swap_T_end - 1.0
        if rng.random() < 0.5:
            # Annual
            exercises = tuple(np.arange(first_ex, last_ex + 0.001, 1.0).round(2))
        else:
            # Quarterly
            exercises = tuple(np.arange(first_ex, last_ex + 0.001, 0.25).round(2))
        notional = float(rng.choice([25e6, 50e6, 100e6], p=[0.30, 0.40, 0.30]))
        pay_receive = rng.choice(["payer", "receiver"])
        strike = 0.040 + rng.choice([-0.005, 0.0, 0.005])
        trades.append(BermudanSwaption(
            trade_id=f"BERM_{i+1:03d}",
            notional=notional,
            currency=ccy,
            pay_receive=pay_receive,
            strike_K=round(strike, 6),
            exercise_dates=exercises,
            swap_T_end=swap_T_end,
            pay_freq=1.0,
        ))

    # 25 callable bonds (corp)
    for i in range(25):
        ccy = "USD"
        T_end = float(rng.choice([5.0, 7.0, 10.0]))
        coupon = round(0.045 + rng.uniform(-0.005, 0.020), 4)  # 4-6.5%
        notional = float(rng.choice([5e6, 25e6, 50e6], p=[0.40, 0.40, 0.20]))
        spread_bps = float(rng.choice([50.0, 100.0, 200.0, 350.0], p=[0.30, 0.40, 0.20, 0.10]))
        # Call schedule: annual after 2y
        call_dates = tuple(np.arange(2.0, T_end - 0.5, 1.0))
        # Call prices: 102 declining to 100 over the schedule
        call_prices = tuple(np.linspace(1.02, 1.00, len(call_dates)).round(4))
        trades.append(CallableBond(
            trade_id=f"CBND_{i+1:03d}",
            notional=notional,
            currency=ccy,
            coupon=coupon,
            pay_freq=0.5,
            T_start=0.0,
            T_end=T_end,
            call_dates=call_dates,
            call_prices=call_prices,
            issuer="corp",
            credit_spread_bps=spread_bps,
            dcc="30/360",
        ))

    # 25 vanilla bonds (govt + corp)
    for i in range(25):
        ccy = "USD"
        is_govt = rng.random() < 0.6
        T_end = float(rng.choice([2.0, 5.0, 10.0, 30.0], p=[0.20, 0.30, 0.30, 0.20]))
        coupon = round(0.040 + rng.uniform(-0.010, 0.020), 4)
        notional = float(rng.choice([1e6, 5e6, 25e6, 50e6], p=[0.30, 0.30, 0.30, 0.10]))
        spread_bps = 0.0 if is_govt else float(rng.choice([50.0, 100.0, 200.0]))
        trades.append(VanillaBond(
            trade_id=f"BND_{i+1:03d}",
            notional=notional,
            currency=ccy,
            coupon=coupon,
            pay_freq=0.5,
            T_start=0.0,
            T_end=T_end,
            issuer="govt" if is_govt else "corp",
            credit_spread_bps=spread_bps,
            dcc="act/act" if is_govt else "30/360",
        ))

    # 10 misc (short-end IRS)
    for i in range(10):
        trades.append(IRS(
            trade_id=f"MISC_{i+1:03d}",
            notional=float(rng.choice([5e6, 25e6, 50e6])),
            currency="USD",
            pay_receive=rng.choice(["payer", "receiver"]),
            fixed_K=round(0.040 + rng.uniform(-0.003, 0.003), 6),
            T_start=0.0,
            T_end=float(rng.choice([1.0, 1.5, 2.0])),
            pay_freq=1.0,
        ))

    # 2 exotics (TARN + range accrual)
    trades.append(TARN(
        trade_id="EXOT_001", notional=10e6, currency="USD",
        cumulative_target=0.20, coupon_floor=0.005, coupon_cap=0.060,
        T_end=5.0, pay_freq=0.5,
    ))
    trades.append(RangeAccrual(
        trade_id="EXOT_002", notional=10e6, currency="USD",
        range_low=0.025, range_high=0.060,
        coupon=0.050, T_end=3.0, pay_freq=0.25,
    ))

    return trades


def trades_to_dataframe(trades: list) -> pd.DataFrame:
    """Flatten heterogeneous trade list into a tidy DataFrame for parquet."""
    rows = []
    for t in trades:
        d = asdict(t)
        # Convert tuples to comma-strings for parquet compatibility
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = ",".join(map(str, v))
        rows.append(d)
    df = pd.DataFrame(rows)
    return df


# ------------------------------------------------------------------
# Curve generator (per-currency, 5 days of moves)
# ------------------------------------------------------------------
BASE_TENORS = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

# USD SOFR-OIS curve (~4% range)
BASE_ZERO_RATES_USD = np.array([0.0428, 0.0420, 0.0410, 0.0395, 0.0380, 0.0370,
                                 0.0365, 0.0370, 0.0380, 0.0390, 0.0400])

# EUR €STR-OIS curve — generally lower than USD post-2024 ECB cuts, slight curve inversion at front
BASE_ZERO_RATES_EUR = np.array([0.0335, 0.0325, 0.0310, 0.0290, 0.0270, 0.0260,
                                 0.0255, 0.0265, 0.0275, 0.0285, 0.0295])


def _apply_day_shock(base_rates: np.ndarray, day: int) -> np.ndarray:
    """Apply the per-day shock pattern to a base curve.

    D1: base
    D2: bull steepener (front -10bp, long -2bp)
    D3: same as D1 (vol moves on D3, not curve)
    D4: parallel +25bp crisis
    D5: partial reversion (D1 + 5bp parallel)
    """
    rates = base_rates.copy()
    if day == 2:
        shock = np.interp(BASE_TENORS, [0, 30], [-0.0010, -0.0002])
        rates = rates + shock
    elif day == 4:
        rates = rates + 0.0025
    elif day == 5:
        rates = rates + 0.0005
    return rates


def curve_for_day(day: int, currency: str = "USD") -> tuple[np.ndarray, np.ndarray]:
    """Return (tenors_y, zero_rates) for trading day `day` ∈ {1..5} and currency.

    Same daily shock pattern is applied to both USD and EUR base curves — banks see
    correlated cross-currency moves on most days, with idiosyncratic moves layered on
    top in real life. The capstone keeps it simple: one shock, applied to both.
    """
    if currency == "USD":
        return BASE_TENORS.copy(), _apply_day_shock(BASE_ZERO_RATES_USD, day)
    if currency == "EUR":
        return BASE_TENORS.copy(), _apply_day_shock(BASE_ZERO_RATES_EUR, day)
    raise ValueError(f"unsupported currency: {currency}")


# ------------------------------------------------------------------
# Swaption cube generator
# ------------------------------------------------------------------
SWAPTION_EXPIRIES = [1.0, 2.0, 5.0]
SWAPTION_TAILS = [5.0, 10.0, 20.0]
SWAPTION_STRIKES_OFFSET_BP = [-100, -50, 0, 50, 100]   # offset from ATM


def swaption_cube_for_day(day: int, atm_proxy: float = 0.040) -> pd.DataFrame:
    """Synthetic swaption vol cube — ATM vol per (expiry × tail) + smile.

    Skew: longer tail → flatter, deeper expiry → richer wings.
    Day 3 vol blowout: +3 vol points across the surface.
    """
    rng = np.random.default_rng(100 + day)
    rows = []
    base_atm = {(T_e, tail): 0.30 + 0.02 * (T_e == 1.0) - 0.02 * (tail == 20.0)
                 for T_e in SWAPTION_EXPIRIES for tail in SWAPTION_TAILS}
    for T_e in SWAPTION_EXPIRIES:
        for tail in SWAPTION_TAILS:
            atm_vol = base_atm[(T_e, tail)]
            if day == 3:
                atm_vol += 0.03  # vol blowout
            for offset_bp in SWAPTION_STRIKES_OFFSET_BP:
                K = atm_proxy + offset_bp / 1e4
                # Smile parabola: + 4bp per 100bp² off-ATM
                smile = 0.0004 * (offset_bp / 100) ** 2
                # Skew: -1bp per 100bp move (equity-style downward)
                skew = -0.0001 * (offset_bp / 100)
                vol = atm_vol + smile + skew + rng.normal(0, 0.002)
                rows.append({
                    "expiry_y": T_e,
                    "tail_y": tail,
                    "strike_offset_bp": offset_bp,
                    "strike": round(K, 6),
                    "vol": round(vol, 6),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Cap vol surface generator
# ------------------------------------------------------------------
def cap_vols_for_day(day: int) -> pd.DataFrame:
    """Synthetic cap-vol curve (humped peaking at 2-3y).

    Day 3 vol blowout: +3 vol points across the curve.
    """
    cap_maturities = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    base_vols = np.array([0.22, 0.24, 0.235, 0.225, 0.21, 0.20])
    vols = base_vols.copy()
    if day == 3:
        vols = vols + 0.03
    return pd.DataFrame({"maturity_y": cap_maturities, "cap_vol": vols.round(6)})


# ------------------------------------------------------------------
# Credit spread generator
# ------------------------------------------------------------------
def credit_spreads_for_day(day: int) -> pd.DataFrame:
    """Synthetic credit spreads per issuer rating bucket (bps).

    Day 4 crisis: spreads widen.
    """
    base = {"AAA": 20.0, "AA": 50.0, "A": 100.0, "BBB": 200.0, "BB": 400.0, "B": 700.0}
    if day == 4:
        for k in base:
            base[k] = base[k] * 1.5  # 50% widening
    elif day == 5:
        for k in base:
            base[k] = base[k] * 1.2  # partial reversion
    return pd.DataFrame({"rating": list(base.keys()), "spread_bps": list(base.values())})


# ------------------------------------------------------------------
# B&R marks generator (with deliberate breaks)
# ------------------------------------------------------------------
def br_marks_for_day(day: int, trades: list, true_pvs: dict, seed: int = 200) -> pd.DataFrame:
    """Synthetic B&R per-trade mark.

    Most marks = true PV + small noise (under tolerance).
    8-12 deliberate breaks per day with realistic causes.
    """
    rng = np.random.default_rng(seed + day)
    rows = []
    n_breaks_target = int(rng.integers(8, 13))
    break_indices = set(rng.choice(len(trades), size=n_breaks_target, replace=False))

    for idx, t in enumerate(trades):
        true_pv = true_pvs.get(t.trade_id, 0.0)
        # Default tolerance per trade type (per million notional)
        tol_per_mm = {
            "irs": 100, "european_swaption": 500, "bermudan_swaption": 2000,
            "vanilla_bond": 50, "callable_bond": 2000, "tarn": 5000, "range_accrual": 5000,
        }
        tol = tol_per_mm.get(t.type, 500) * (t.notional / 1e6)
        # Within-tolerance noise
        noise = rng.normal(0, tol * 0.3)
        if idx in break_indices:
            # Deliberate break: 3-10x tolerance
            magnitude = rng.uniform(3, 10) * tol * rng.choice([-1, 1])
            cause = rng.choice([
                "stale_mark", "off_grid_smile", "day_count_mismatch",
                "bermudan_mc_noise", "model_choice",
            ])
            br_mark = true_pv + magnitude
            rows.append({
                "trade_id": t.trade_id, "type": t.type, "notional": t.notional,
                "br_mark": round(br_mark, 4), "true_pv_internal": round(true_pv, 4),
                "tolerance": round(tol, 2), "cause_hypothesis": str(cause),
                "is_deliberate_break": True,
            })
        else:
            br_mark = true_pv + noise
            rows.append({
                "trade_id": t.trade_id, "type": t.type, "notional": t.notional,
                "br_mark": round(br_mark, 4), "true_pv_internal": round(true_pv, 4),
                "tolerance": round(tol, 2), "cause_hypothesis": "",
                "is_deliberate_break": False,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    md_dir = data_dir / "market_data"
    br_dir = data_dir / "br_marks"
    data_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    br_dir.mkdir(parents=True, exist_ok=True)

    print("Generating trade book ...")
    trades = generate_trades(seed=42)
    print(f"  generated {len(trades)} trades")
    df_trades = trades_to_dataframe(trades)
    df_trades.to_parquet(data_dir / "trades.parquet", index=False)
    print(f"  saved -> data/trades.parquet  ({len(df_trades)} rows)")

    # For B&R marks generation we need per-trade TRUE PVs (using the user's pricers).
    # We compute these per currency using the right curve.
    print("\nGenerating market data + B&R marks per day ...")

    # Clean up legacy single-curve files if present
    for old_path in md_dir.glob("curves_D*.parquet"):
        if "_USD_" not in old_path.name and "_EUR_" not in old_path.name:
            old_path.unlink()

    for day in range(1, 6):
        # Per-currency curves
        for ccy in ("USD", "EUR"):
            tenors, rates = curve_for_day(day, currency=ccy)
            df_curve = pd.DataFrame({"tenor_y": tenors, "zero_rate": rates})
            df_curve.to_parquet(md_dir / f"curves_{ccy}_D{day}.parquet", index=False)

        # Swaption cube
        df_cube = swaption_cube_for_day(day)
        df_cube.to_parquet(md_dir / f"swaption_cube_D{day}.parquet", index=False)

        # Cap vols
        df_caps = cap_vols_for_day(day)
        df_caps.to_parquet(md_dir / f"cap_vols_D{day}.parquet", index=False)

        # Credit spreads
        df_credit = credit_spreads_for_day(day)
        df_credit.to_parquet(md_dir / f"credit_spreads_D{day}.parquet", index=False)

        print(f"  day {day}: curves(USD,EUR) + cube + caps + credit -> data/market_data/*_D{day}.parquet")

        # Build per-currency curves for this day to compute approximate true PVs for B&R
        curves_by_ccy = {
            ccy: Curve(*curve_for_day(day, currency=ccy), name=f"OIS_{ccy}")
            for ccy in ("USD", "EUR")
        }

        # Compute PVs for vanilla trades using the right curve per currency.
        # Bermudans/exotics get a placeholder approximation (B&R generator self-contained).
        true_pvs = {}
        for t in trades:
            ccy = getattr(t, "currency", "USD")
            cv = curves_by_ccy.get(ccy, curves_by_ccy["USD"])
            try:
                if t.type == "irs":
                    true_pvs[t.trade_id] = price_irs(t, cv)
                elif t.type == "vanilla_bond":
                    true_pvs[t.trade_id] = price_vanilla_bond(t, cv)
                elif t.type == "european_swaption":
                    # Use a flat 30% vol proxy (user re-prices with calibrated SABR)
                    true_pvs[t.trade_id] = price_european_swaption(t, cv, sigma=0.30)
                elif t.type == "callable_bond":
                    # Lower-bound proxy (user uses LMM stack)
                    underlying = VanillaBond(
                        trade_id=t.trade_id, notional=t.notional, currency=t.currency,
                        coupon=t.coupon, pay_freq=t.pay_freq, T_start=t.T_start,
                        T_end=t.T_end, issuer=t.issuer, credit_spread_bps=t.credit_spread_bps,
                        dcc=t.dcc,
                    )
                    true_pvs[t.trade_id] = price_vanilla_bond(underlying, cv) * 0.97
                else:
                    # Bermudan / exotic — placeholder (notional × small factor)
                    true_pvs[t.trade_id] = t.notional * 0.005
            except Exception:
                true_pvs[t.trade_id] = 0.0

        # B&R marks
        df_br = br_marks_for_day(day, trades, true_pvs)
        df_br.to_parquet(br_dir / f"br_D{day}.parquet", index=False)
        n_breaks = int(df_br["is_deliberate_break"].sum())
        print(f"             br_marks/br_D{day}.parquet  ({n_breaks} deliberate breaks)")

    print("\nDONE")
    print(f"  trades            : {len(trades)} rows")
    print(f"  market data files : {len(list(md_dir.glob('*.parquet')))}")
    print(f"  br_marks files    : {len(list(br_dir.glob('*.parquet')))}")


if __name__ == "__main__":
    main()
