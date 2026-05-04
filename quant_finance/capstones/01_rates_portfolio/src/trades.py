"""Trade dataclasses for the rates portfolio capstone.

Trade types modelled:
- IRS                 : plain-vanilla interest rate swap
- EuropeanSwaption    : single-exercise option on a swap
- BermudanSwaption    : multi-callable option on a swap
- VanillaBond         : fixed-coupon bond, govvie or corp
- CallableBond        : bond with American call schedule
- TARN / RangeAccrual : exotics (placeholder representation)

Day-count conventions are encoded per trade. The dispatcher in the user's
notebook reads `trade.type` and routes to the correct pricer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


Currency = Literal["USD", "EUR"]
PayerReceiver = Literal["payer", "receiver"]
DayCount = Literal["act/360", "30/360", "act/act", "act/365"]


@dataclass
class IRS:
    """Plain-vanilla interest rate swap.

    Single-currency, fixed-vs-floating, paying `fixed_K` against the index.
    """
    trade_id: str
    notional: float
    currency: Currency
    pay_receive: PayerReceiver        # "payer" pays fixed
    fixed_K: float                    # annualised fixed rate
    T_start: float                    # year-fraction from today to first fixing
    T_end: float                      # year-fraction from today to maturity
    pay_freq: float = 1.0             # pay every `pay_freq` years; e.g. 1.0 = annual
    fixed_dcc: DayCount = "act/360"   # USD swap default; EUR uses 30/360 (set explicitly)
    float_dcc: DayCount = "act/360"
    type: Literal["irs"] = "irs"


@dataclass
class EuropeanSwaption:
    """European swaption: option to enter a swap at T_e."""
    trade_id: str
    notional: float
    currency: Currency
    pay_receive: PayerReceiver        # underlying swap direction
    strike_K: float                   # fixed rate of underlying swap
    T_e: float                        # exercise date (year-fraction from today)
    swap_T_end: float                 # underlying swap maturity (from today)
    pay_freq: float = 1.0
    type: Literal["european_swaption"] = "european_swaption"


@dataclass
class BermudanSwaption:
    """Multi-callable Bermudan swaption."""
    trade_id: str
    notional: float
    currency: Currency
    pay_receive: PayerReceiver
    strike_K: float
    exercise_dates: tuple[float, ...]   # year-fractions from today
    swap_T_end: float                   # underlying swap maturity (from today)
    pay_freq: float = 1.0
    type: Literal["bermudan_swaption"] = "bermudan_swaption"


@dataclass
class VanillaBond:
    """Fixed-coupon bond, govvie or corp."""
    trade_id: str
    notional: float
    currency: Currency
    coupon: float                       # annual coupon rate (e.g. 0.045 for 4.5%)
    pay_freq: float = 0.5               # semi-annual = 0.5
    T_start: float = 0.0                # year-fraction to first coupon
    T_end: float = 5.0                  # year-fraction to maturity
    issuer: str = "govt"                # "govt" / "corp"
    credit_spread_bps: float = 0.0      # add to discount curve for corps
    dcc: DayCount = "act/act"
    type: Literal["vanilla_bond"] = "vanilla_bond"


@dataclass
class CallableBond:
    """Vanilla bond + American call schedule.

    Decompose as: PV(callable) = PV(vanilla bond) - PV(American call on bond).
    The American call is priced via the LMM stack (LSMC) since the underlying
    bond price is path-dependent in rates.
    """
    trade_id: str
    notional: float
    currency: Currency
    coupon: float
    pay_freq: float = 0.5
    T_start: float = 0.0
    T_end: float = 10.0
    call_dates: tuple[float, ...] = field(default_factory=tuple)   # American call schedule
    call_prices: tuple[float, ...] = field(default_factory=tuple)  # par + premium per call date
    issuer: str = "corp"
    credit_spread_bps: float = 50.0
    dcc: DayCount = "30/360"
    type: Literal["callable_bond"] = "callable_bond"


@dataclass
class TARN:
    """Target Accrual Redemption Note (placeholder, for exotic-pricer dispatch).

    Pays floating coupons subject to a cumulative target. The user's notebook
    routes these to a Monte Carlo pricer; we just need a representation.
    """
    trade_id: str
    notional: float
    currency: Currency
    cumulative_target: float
    coupon_floor: float
    coupon_cap: float
    T_end: float
    pay_freq: float = 0.5
    type: Literal["tarn"] = "tarn"


@dataclass
class RangeAccrual:
    """Range-accrual note (placeholder)."""
    trade_id: str
    notional: float
    currency: Currency
    range_low: float                  # accrue floor
    range_high: float                 # accrue cap
    coupon: float
    T_end: float
    pay_freq: float = 0.25
    type: Literal["range_accrual"] = "range_accrual"


# Type alias for the dispatcher
Trade = IRS | EuropeanSwaption | BermudanSwaption | VanillaBond | CallableBond | TARN | RangeAccrual


# ------------------------------------------------------------------
# Convenience: payment-date schedule for a swap leg
# ------------------------------------------------------------------
def pay_dates(T_start: float, T_end: float, freq: float) -> np.ndarray:
    """Generate payment dates as year-fractions from today.

    Includes T_end. Spacing = freq. Excludes T_start itself (first payment is at T_start + freq).
    """
    if T_end <= T_start:
        return np.array([])
    n = int(np.round((T_end - T_start) / freq))
    return T_start + freq * np.arange(1, n + 1)


def load_trades(parquet_path: str | "Path") -> list[Trade]:
    """Load trades from `data/trades.parquet` and reconstruct dataclass instances.

    The data generator stores tuple fields (`exercise_dates`, `call_dates`, `call_prices`)
    as comma-strings for parquet compatibility. This helper parses them back.
    """
    import pandas as pd
    from pathlib import Path

    df = pd.read_parquet(Path(parquet_path))
    trades: list[Trade] = []
    for _, row in df.iterrows():
        d = row.to_dict()
        ttype = d["type"]

        # Parse comma-string tuple fields
        def _parse_tuple(s: str, cast=float) -> tuple:
            if not s or (isinstance(s, float) and np.isnan(s)):
                return ()
            return tuple(cast(x) for x in str(s).split(","))

        if ttype == "irs":
            trades.append(IRS(
                trade_id=d["trade_id"], notional=float(d["notional"]),
                currency=d["currency"], pay_receive=d["pay_receive"],
                fixed_K=float(d["fixed_K"]), T_start=float(d["T_start"]),
                T_end=float(d["T_end"]), pay_freq=float(d["pay_freq"]),
                fixed_dcc=d["fixed_dcc"], float_dcc=d["float_dcc"],
            ))
        elif ttype == "european_swaption":
            trades.append(EuropeanSwaption(
                trade_id=d["trade_id"], notional=float(d["notional"]),
                currency=d["currency"], pay_receive=d["pay_receive"],
                strike_K=float(d["strike_K"]), T_e=float(d["T_e"]),
                swap_T_end=float(d["swap_T_end"]), pay_freq=float(d["pay_freq"]),
            ))
        elif ttype == "bermudan_swaption":
            trades.append(BermudanSwaption(
                trade_id=d["trade_id"], notional=float(d["notional"]),
                currency=d["currency"], pay_receive=d["pay_receive"],
                strike_K=float(d["strike_K"]),
                exercise_dates=_parse_tuple(d["exercise_dates"]),
                swap_T_end=float(d["swap_T_end"]), pay_freq=float(d["pay_freq"]),
            ))
        elif ttype == "vanilla_bond":
            trades.append(VanillaBond(
                trade_id=d["trade_id"], notional=float(d["notional"]),
                currency=d["currency"], coupon=float(d["coupon"]),
                pay_freq=float(d["pay_freq"]), T_start=float(d["T_start"]),
                T_end=float(d["T_end"]), issuer=d["issuer"],
                credit_spread_bps=float(d["credit_spread_bps"]), dcc=d["dcc"],
            ))
        elif ttype == "callable_bond":
            trades.append(CallableBond(
                trade_id=d["trade_id"], notional=float(d["notional"]),
                currency=d["currency"], coupon=float(d["coupon"]),
                pay_freq=float(d["pay_freq"]), T_start=float(d["T_start"]),
                T_end=float(d["T_end"]),
                call_dates=_parse_tuple(d["call_dates"]),
                call_prices=_parse_tuple(d["call_prices"]),
                issuer=d["issuer"], credit_spread_bps=float(d["credit_spread_bps"]),
                dcc=d["dcc"],
            ))
        elif ttype == "tarn":
            trades.append(TARN(
                trade_id=d["trade_id"], notional=float(d["notional"]),
                currency=d["currency"], cumulative_target=float(d["cumulative_target"]),
                coupon_floor=float(d["coupon_floor"]), coupon_cap=float(d["coupon_cap"]),
                T_end=float(d["T_end"]), pay_freq=float(d["pay_freq"]),
            ))
        elif ttype == "range_accrual":
            trades.append(RangeAccrual(
                trade_id=d["trade_id"], notional=float(d["notional"]),
                currency=d["currency"], range_low=float(d["range_low"]),
                range_high=float(d["range_high"]), coupon=float(d["coupon"]),
                T_end=float(d["T_end"]), pay_freq=float(d["pay_freq"]),
            ))
        else:
            raise ValueError(f"unknown trade type: {ttype}")
    return trades


def year_fraction(d1: float, d2: float, dcc: DayCount) -> float:
    """Day-count year fraction.

    For the synthetic data we model time as a continuous year-fraction from today,
    so the dcc adjustment is small (act/360 vs act/365 vs 30/360 differ by < 1.5%).
    Real systems compute from calendar dates; here we apply a constant scaling.
    """
    raw = d2 - d1
    if dcc == "act/360":
        return raw * 365.0 / 360.0
    if dcc == "act/365":
        return raw
    if dcc == "30/360":
        return raw * 360.0 / 365.0
    if dcc == "act/act":
        return raw
    raise ValueError(f"unknown dcc {dcc}")
