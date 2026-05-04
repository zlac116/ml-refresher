"""Interest rate swap pricer.

Uses the compact PV form derived in `quant_finance/03_fixed_income/04_swaps_swaptions.ipynb`:

    PV_payer = -A · (K - K_par)   = A · (K_par - K)

where K_par is today's par swap rate and A is the annuity. Identity from
substituting the par-rate definition K_par = (1 - D(T_n)) / A into the cash-flow
form Σ_i (L_i - K) δ_i D(0, T_i).
"""

from __future__ import annotations

import numpy as np

from ..curves import Curve, par_swap_rate
from ..trades import IRS


def price_irs(trade: IRS, disc: Curve, proj: Curve | None = None) -> float:
    """Mark-to-market of an IRS.

    Returns PV in trade currency (notional-scaled). Sign convention:
    - payer swap > 0 when K_par > K (paying below market)
    - receiver swap > 0 when K_par < K (receiving above market)
    """
    if proj is None:
        proj = disc
    K_par, annuity = par_swap_rate(disc, proj, trade.T_start, trade.T_end, trade.pay_freq)
    sign = +1.0 if trade.pay_receive == "payer" else -1.0
    pv = sign * annuity * (K_par - trade.fixed_K) * trade.notional
    return float(pv)


def dv01_irs(trade: IRS, disc: Curve, proj: Curve | None = None, bp: float = 1.0) -> float:
    """Central-difference DV01 in PV currency per 1bp parallel shift in the disc curve.

    **Sign convention (note for desk users)**: this returns `(pv_dn - pv_up) / 2`,
    so DV01 is **positive when PV rises as rates fall**. Under this convention:

    - **Long receiver swap** → DV01 > 0 (long duration, gains when rates fall)
    - **Long payer swap**    → DV01 < 0 (short duration, loses when rates fall)

    Worked example: a USD 5y at-par payer swap on a 4% flat curve, $100mm notional,
    has DV01 ≈ -$45,000/bp (negative because payers are short duration).

    Bloomberg / MX.3 / OpenGamma sometimes quote `dPV/dy` (positive = PV rises with
    +1bp), which has the opposite sign. Convert by negating if interfacing with
    those systems.
    """
    bump = bp / 1e4
    pv_up = price_irs(trade, disc.shift(+bump), proj.shift(+bump) if proj is not None else None)
    pv_dn = price_irs(trade, disc.shift(-bump), proj.shift(-bump) if proj is not None else None)
    return (pv_dn - pv_up) / 2.0
