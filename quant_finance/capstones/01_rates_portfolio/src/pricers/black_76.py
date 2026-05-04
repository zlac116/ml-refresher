"""Black-76 swaption pricer.

Standard formula: $C = D(0,T) [F N(d_1) - K N(d_2)]$ on the forward swap rate.
For a payer swaption, $C$ above is the premium; for a receiver, swap signs.

Annuity-discounted form (used for swaptions where the natural numeraire is the annuity):
    Premium_payer = A · [F N(d1) − K N(d2)]
    Premium_receiver = A · [K N(−d2) − F N(−d1)]

where A is the annuity (sum of accrual-weighted DFs), F is the forward swap rate,
and the d1/d2 are the standard Black-76 expressions on (F, K, σ, T).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..curves import Curve, par_swap_rate
from ..trades import EuropeanSwaption


def black_76_annuity_premium(
    F: float, K: float, T: float, sigma: float, annuity: float, payer: bool = True
) -> float:
    """Premium of a swaption under Black-76, expressed against the annuity numeraire.

    F, K: forward swap rate and strike (annualised, decimal)
    T: time to exercise (years)
    sigma: lognormal vol on F
    annuity: sum of accrual-weighted discount factors over the underlying swap
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if payer else max(K - F, 0.0)
        return annuity * intrinsic
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if payer:
        return annuity * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return annuity * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def price_european_swaption(
    trade: EuropeanSwaption, disc: Curve, proj: Curve | None = None, sigma: float = 0.30
) -> float:
    """Price a European swaption under Black-76.

    sigma is the lognormal vol for the (T_e × tail) cell; user supplies it from
    the calibrated SABR cube in the notebook.
    """
    if proj is None:
        proj = disc
    F, annuity = par_swap_rate(disc, proj, trade.T_e, trade.swap_T_end, trade.pay_freq)
    payer = trade.pay_receive == "payer"
    return float(
        trade.notional
        * black_76_annuity_premium(F, trade.strike_K, trade.T_e, sigma, annuity, payer=payer)
    )
