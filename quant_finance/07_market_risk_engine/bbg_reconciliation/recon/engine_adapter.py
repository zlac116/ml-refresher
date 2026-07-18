"""Capture layer — mirror the in-house engine's base + product class shape.

MINIMAL scope: FX forwards & swaps only.

Your framework: a BASE class that PVs cashflows off discount factors, stresses
the market data, and re-PVs; PRODUCT classes that filter the flat cashflow table
to their deal and PV/stress it. This module mirrors that so wiring is a call into
your real classes, not a re-implementation:

    ProductAdapter.collect()   generic base -> stress -> revalue loop (the base
                               class behaviour): base PV, stressed PV, impact.
    build_engine(trade)        SEAM 1: pick this deal's cashflows, build engine.
    apply_stress(engine, ...)  SEAM 2: shock the market data for one scenario.

The reference engine (recon/pv_engine.py) already satisfies the contract, so this
runs end-to-end on the demo inputs. Swap build_engine to point at your real
product class when ready.
"""

from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config, pv_engine, schema

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INHOUSE_DIR = DATA_DIR / "inhouse"


# ---------------------------------------------------------------------------
# Trade spec + base market (from config/market.yaml + input files).
# ---------------------------------------------------------------------------
@dataclass
class TradeSpec:
    trade_id: str            # == deal_no in the cashflow table
    trade_type: str
    ccy: str                 # reporting/label ccy for the deal
    notional: float
    params: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _market():
    """Load base market ONCE: flat cashflow table, per-ccy DF curves, fx0."""
    m = config.load_yaml("market.yaml")
    as_of = pd.Timestamp(m["as_of"])
    cashflows = pv_engine.load_cashflows(INHOUSE_DIR / m["cashflows_file"])
    curves = pv_engine.load_discount_curves(INHOUSE_DIR / m["discount_curves_file"], as_of)
    market = pv_engine.MarketData(curves=curves, fx=dict(m["fx"]),
                                  reporting_ccy=m.get("reporting_ccy", "GBP"))
    return cashflows, market, as_of


# ---------------------------------------------------------------------------
# Base capture adapter — the BASE class loop. Two seams; loop inherited.
# ---------------------------------------------------------------------------
class ProductAdapter(ABC):
    trade_type: str = ""

    # ---- SEAM 1: this deal's engine ----------------------------------------
    def build_engine(self, trade: TradeSpec):
        """Filter the flat cashflow table to this deal (product class analogue)
        and build the engine on base market data.

        TODO(you): replace with your real product class, e.g.
            return FxForwardPricer(deal_no=trade.trade_id, market=..., ...)
        """
        cashflows, market, as_of = _market()
        deal = cashflows[cashflows["deal_no"] == trade.trade_id]
        if deal.empty:
            raise KeyError(f"no cashflows for deal_no {trade.trade_id!r}")
        instr = trade.params.get("instrument")
        if instr:
            deal = deal[deal["instrument"] == instr]
        return pv_engine.DiscountedCashflowEngine(deal, market, as_of)

    # ---- SEAM 2: apply one scenario's shock --------------------------------
    def apply_stress(self, engine, scenario_id: str, cfg: dict) -> None:
        """Translate the scenarios.yaml shock params into the base class stress
        call. For FX fwd/swap: FX spot and/or DF-curve bp shift."""
        engine.stress(
            dfs_shift_bp=cfg.get("dfs_shift_bp", 0.0),
            ccy=cfg.get("shift_ccy"),
            fx_mult=cfg.get("fx_mult"),
            fx_ccy=cfg.get("fx_ccy"),
        )

    # ---- generic loop (do not override) ------------------------------------
    def applicable_scenarios(self, scenarios_cfg: dict) -> list[str]:
        return [
            sid for sid, s in scenarios_cfg.items()
            if s.get("applies_to") == "all" or self.trade_type in s.get("applies_to", [])
        ]

    def collect(self, trades: list[TradeSpec], scenarios_cfg: dict) -> list[dict]:
        rows: list[dict] = []
        for tr in trades:
            base_pv = float(self.build_engine(tr).revalue())
            rows += [
                _row(tr, "BASE", schema.METRIC_BASE_PV, base_pv),
                _row(tr, "BASE", schema.METRIC_STRESSED_PV, base_pv),
                _row(tr, "BASE", schema.METRIC_IMPACT, 0.0),
            ]
            for sid in self.applicable_scenarios(scenarios_cfg):
                if sid == "BASE":
                    continue
                engine = self.build_engine(tr)      # fresh base market each scenario
                self.apply_stress(engine, sid, scenarios_cfg[sid])
                stressed = float(engine.revalue())
                rows += [
                    _row(tr, sid, schema.METRIC_STRESSED_PV, stressed),
                    _row(tr, sid, schema.METRIC_IMPACT, stressed - base_pv),
                ]
        return rows


class FxForwardAdapter(ProductAdapter):
    trade_type = "fx_forward"


class FxSwapAdapter(ProductAdapter):
    trade_type = "fx_swap"


PRODUCT_ADAPTERS: dict[str, ProductAdapter] = {
    a.trade_type: a for a in (FxForwardAdapter(), FxSwapAdapter())
}


def _row(tr: TradeSpec, scenario_id: str, metric: str, value: float) -> dict:
    return {
        schema.TRADE_ID: tr.trade_id, schema.TRADE_TYPE: tr.trade_type,
        schema.CCY: tr.ccy, schema.NOTIONAL: tr.notional,
        schema.SCENARIO_ID: scenario_id, schema.METRIC: metric, schema.VALUE: value,
    }


# ---------------------------------------------------------------------------
def load_trade_specs() -> dict[str, list[TradeSpec]]:
    raw = config.load_yaml("trades.yaml")
    out: dict[str, list[TradeSpec]] = {}
    for trade_type, trades in (raw or {}).items():
        out[trade_type] = [
            TradeSpec(trade_id=t["trade_id"], trade_type=trade_type, ccy=t["ccy"],
                      notional=float(t["notional"]), params=t.get("params", {}))
            for t in (trades or [])
        ]
    return out


def collect_all() -> pd.DataFrame:
    """Run every product adapter over its trades and scenarios -> canonical frame."""
    scenarios_cfg = config.load_scenarios()
    rows: list[dict] = []
    for trade_type, trades in load_trade_specs().items():
        adapter = PRODUCT_ADAPTERS.get(trade_type)
        if adapter and trades:
            rows += adapter.collect(trades, scenarios_cfg)
    df = pd.DataFrame(rows, columns=schema.CANONICAL_COLS)
    return schema.validate_canonical(df, source="inhouse-engine")
