"""FastMCP server — trade-desk co-pilot.

Spec: ./README.md (~2.5h, 2 tools, HITL on the write tool).

Run (stdio transport, default):
    python mcp_server.py

Test interactively:
    npx @modelcontextprotocol/inspector python mcp_server.py

Tools exposed:
  - get_portfolio_pnl(date)        # read
  - submit_trade_proposal(spec)    # WRITE — caller MUST use interrupt_before=['tools']
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

# --- TODO: imports ---
# from fastmcp import FastMCP
# from pydantic import BaseModel, Field
# import pandas as pd

HERE = Path(__file__).parent
PROPOSALS_DIR = HERE / "proposals"
PROPOSALS_DIR.mkdir(exist_ok=True)

# Reuse the existing rates-portfolio data — don't regenerate
RATES_DATA = HERE.parent.parent.parent / "quant_finance" / "capstones" / "01_rates_portfolio" / "data"
TRADES_PARQUET = RATES_DATA / "trades.parquet"
MARKET_DATA_DIR = RATES_DATA / "market_data"
BR_MARKS_DIR = RATES_DATA / "br_marks"

# mcp = FastMCP("trade-desk-co-pilot")


# ============================================================
# Tool 1 — get_portfolio_pnl  (read)
# ============================================================
# Hints:
#   - Inspect TRADES_PARQUET columns first: pd.read_parquet(TRADES_PARQUET).columns
#   - The br_marks/ folder has per-day mark-to-market parquets (D1-D5);
#     PnL = sum(MTM_today - MTM_prev_day) by trade_type or currency.
#   - For v1, just aggregate by trade_type for one day. Don't over-engineer.
#   - Raise ToolException for unknown date / missing file.
#
# @mcp.tool
# def get_portfolio_pnl(date: str) -> dict:
#     """Aggregate portfolio PnL by trade type for the given date.
#
#     Args:
#         date: YYYY-MM-DD. Must correspond to one of the available BR-marks days.
#
#     Returns:
#         {"date": str, "pnl_by_trade_type": {<type>: float}, "total": float, "ccy": str}
#     """
#     # TODO: load br_marks for `date` and `prev(date)`, diff, group by trade_type, sum
#     ...


# ============================================================
# Tool 2 — submit_trade_proposal  (WRITE — HITL)
# ============================================================
# Hints:
#   - Define TradeSpec with the minimum fields a desk needs: instrument, side,
#     notional, tenor, currency, rationale.
#   - The function writes JSON to PROPOSALS_DIR; the AGENT side gates this with
#     interrupt_before=['tools']. The server itself does NOT enforce HITL — that's
#     the agent's responsibility.
#   - Return a proposal_id (e.g., timestamp + 4 random chars) so the agent can refer to it.
#
# class TradeSpec(BaseModel):
#     instrument: str = Field(description="e.g. '10y EUR receiver swap'")
#     side: Literal["buy", "sell"]
#     notional: float = Field(gt=0, description="Notional in millions of `currency`.")
#     tenor: str = Field(description="e.g. '10Y', '2Y', '5Y'")
#     currency: Literal["USD", "EUR", "GBP", "JPY"]
#     rationale: str = Field(description="One-line analyst rationale.")
#
# @mcp.tool
# def submit_trade_proposal(spec: TradeSpec) -> str:
#     """Persist a trade proposal as JSON. Returns the proposal_id.
#
#     IMPORTANT: This is a WRITE tool. The agent client MUST gate calls with
#     interrupt_before=['tools'] so a human approves before this fires.
#     """
#     # TODO: build proposal_id, write spec.model_dump() to PROPOSALS_DIR / f"{proposal_id}.json"
#     ...


if __name__ == "__main__":
    pass
    # mcp.run()  # stdio by default
