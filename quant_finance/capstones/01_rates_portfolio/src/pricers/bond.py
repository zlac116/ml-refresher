"""Vanilla and callable bond pricers.

Vanilla: PV = sum of discounted cash flows on the (OIS + credit-spread) curve.

Callable: PV(callable) = PV(vanilla) − PV(American call on the bond). The American
call is priced via the LMM stack (LSMC backward induction) in `src/pricers/lmm.py`.
For a quick lower bound (used as a sanity check), `price_callable_bond_lower_bound`
returns the smaller of the vanilla PV and the held-to-first-call PV.
"""

from __future__ import annotations

import numpy as np

from ..curves import Curve
from ..trades import VanillaBond, CallableBond, pay_dates


def price_vanilla_bond(trade: VanillaBond, disc: Curve) -> float:
    """PV of a fixed-coupon bond on (OIS curve + flat credit spread)."""
    spread = trade.credit_spread_bps / 1e4
    cfs = pay_dates(trade.T_start, trade.T_end, trade.pay_freq)
    coupon_per_pmt = trade.coupon * trade.pay_freq * trade.notional
    pv = 0.0
    for t in cfs:
        discount = float(disc.DF(t)) * np.exp(-spread * t)
        pv += coupon_per_pmt * discount
    pv += trade.notional * float(disc.DF(trade.T_end)) * np.exp(-spread * trade.T_end)
    return float(pv)


def price_callable_bond_lower_bound(trade: CallableBond, disc: Curve) -> float:
    """Lower bound: min(vanilla PV, held-to-first-call PV).

    Proper callable-bond pricing routes through the LMM Bermudan code; this is a
    sanity reference only.
    """
    underlying_vanilla = VanillaBond(
        trade_id=trade.trade_id,
        notional=trade.notional,
        currency=trade.currency,
        coupon=trade.coupon,
        pay_freq=trade.pay_freq,
        T_start=trade.T_start,
        T_end=trade.T_end,
        issuer=trade.issuer,
        credit_spread_bps=trade.credit_spread_bps,
        dcc=trade.dcc,
    )
    vanilla_pv = price_vanilla_bond(underlying_vanilla, disc)
    if not trade.call_dates:
        return vanilla_pv
    call_t = trade.call_dates[0]
    call_p = trade.call_prices[0] * trade.notional
    spread = trade.credit_spread_bps / 1e4
    coupon_per_pmt = trade.coupon * trade.pay_freq * trade.notional
    held_to_call = call_p * float(disc.DF(call_t))
    for t in pay_dates(trade.T_start, call_t, trade.pay_freq):
        held_to_call += coupon_per_pmt * float(disc.DF(t)) * np.exp(-spread * t)
    return min(vanilla_pv, held_to_call)
