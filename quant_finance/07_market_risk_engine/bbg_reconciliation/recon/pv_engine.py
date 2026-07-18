"""Reference discounted-cashflow engine — the BASE class analogue.

Mirrors the in-house mechanic exactly:

    PV = Σ  cashflow_amount_i · DF_ccy(t_i)      then × fx0 → reporting ccy

* Cashflows are a FLAT table of every deal's flows:
      deal_no, instrument, ccy, cashflow_date, amount   (+ any extra cols)
  A product engine is just that table FILTERED to one deal_no (+ instrument).
* Discount factors come from a shared input file, one curve per ccy:
      ccy, date, df
* A STRESS shifts a DF curve (parallel or key-rate, in bp) and/or scales FX,
  then re-PVs — i.e. revalue → stress market data → revalue.

Reference implementation so the pack runs end-to-end and the engine_adapter
seams have a concrete shape to copy. Swap for your real product class; keep the
contract (`revalue()` + a stress method). numpy + pandas only, every step a small
auditable closed form (same spirit as pra_liquidity_stress/engine.py).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# Input loaders — real-shaped files (swap paths for your production inputs).
# ---------------------------------------------------------------------------
def load_cashflows(path) -> pd.DataFrame:
    """Flat all-deals cashflow table. Required cols: deal_no, instrument, ccy,
    cashflow_date, amount. Extra cols are carried through untouched."""
    df = pd.read_csv(path, parse_dates=["cashflow_date"])
    need = {"deal_no", "instrument", "ccy", "cashflow_date", "amount"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"cashflow file missing columns: {sorted(missing)}")
    return df


def load_discount_curves(path, as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """DF pillars per ccy. Cols: ccy, date, df. Returns ccy -> DataFrame with a
    tenor_years column measured from `as_of` (for interpolation)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["tenor_years"] = (df["date"] - as_of).dt.days / DAYS_PER_YEAR
    out = {}
    for ccy, g in df.groupby("ccy"):
        out[ccy] = g.sort_values("tenor_years")[["tenor_years", "df"]].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Market data — the thing that gets stressed.
# ---------------------------------------------------------------------------
@dataclass
class MarketData:
    curves: dict[str, pd.DataFrame]   # ccy -> (tenor_years, df)
    fx: dict[str, float]              # ccy -> reporting-ccy per 1 unit ccy (fx0)
    reporting_ccy: str = "GBP"

    def copy(self) -> "MarketData":
        return MarketData({k: v.copy() for k, v in self.curves.items()},
                          dict(self.fx), self.reporting_ccy)

    def _zero(self, ccy: str) -> tuple:
        c = self.curves[ccy]
        t = c["tenor_years"].to_numpy(dtype=float)
        d = c["df"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(t > 0, -np.log(d) / t, 0.0)
        return t, z

    def df_at(self, ccy: str, tenors) -> np.ndarray:
        """DF at arbitrary tenors — interpolate the zero rate linearly in t and
        rebuild the DF (keeps DFs in (0,1] and monotone)."""
        if ccy not in self.curves:
            raise KeyError(f"no discount curve for ccy {ccy!r}")
        t, z = self._zero(ccy)
        tenors = np.asarray(tenors, dtype=float)
        zi = np.interp(tenors, t, z)
        return np.exp(-zi * tenors)

    def shift_bp(self, bp: float, ccy: str | None = None) -> None:
        """Parallel additive bp bump on the zero rate, one ccy curve or all
        (None). The discount-curve stress."""
        for cc in ([ccy] if ccy else list(self.curves)):
            t, z = self._zero(cc)
            z = z + bp * 1e-4
            self.curves[cc] = pd.DataFrame({"tenor_years": t, "df": np.exp(-z * t)})

    def scale_fx(self, mult: float, ccy: str | None = None) -> None:
        """Multiplicative FX shock (None ccy = all non-reporting ccys)."""
        for k in list(self.fx):
            if k == self.reporting_ccy:
                continue
            if ccy is None or k == ccy:
                self.fx[k] *= mult


# ---------------------------------------------------------------------------
# The engine — BASE class analogue.
# ---------------------------------------------------------------------------
class DiscountedCashflowEngine:
    """revalue() PVs the current market; stress() shifts it in place. Construct
    with the ONE deal's cashflows (product class = filter of the flat table)."""

    def __init__(self, cashflows: pd.DataFrame, market: MarketData, as_of: pd.Timestamp):
        self.cashflows = cashflows.copy()
        self.market = market.copy()   # own copy so stress never mutates the base
        self.as_of = as_of

    def revalue(self) -> float:
        cf = self.cashflows
        pv = 0.0
        for ccy, idx in cf.groupby("ccy").groups.items():
            rows = cf.loc[idx]
            t = (rows["cashflow_date"] - self.as_of).dt.days.to_numpy() / DAYS_PER_YEAR
            dfv = self.market.df_at(ccy, t)
            fx = self.market.fx.get(ccy)
            if fx is None:
                raise KeyError(f"no fx0 for ccy {ccy!r}")
            pv += float(np.sum(rows["amount"].to_numpy(dtype=float) * dfv) * fx)
        return pv

    def stress(self, dfs_shift_bp: float = 0.0, ccy: str | None = None,
               fx_mult: float | None = None, fx_ccy: str | None = None) -> None:
        if dfs_shift_bp:
            self.market.shift_bp(dfs_shift_bp, ccy)
        if fx_mult is not None:
            self.market.scale_fx(fx_mult, fx_ccy)
