"""LangChain @tool wrappers around the LMM surrogate API.

Each tool is a Python function decorated with @tool. The LLM reads:
  1. The function name → decides "this looks relevant"
  2. The argument names + type hints → knows what to pass
  3. The DOCSTRING → understands what the tool does and when to use it

So the docstring is part of your prompt to the LLM. Write it for the AI,
not the human. Include WHEN to call this tool and WHAT it returns.

Design choices:
  - Tools are SYNC (LangChain supports async too, but sync is simpler and
    httpx supports both). For a learning capstone, sync first.
  - Tools take an httpx.Client via a module-level singleton (see _client()).
    This is testable: tests can swap the client via respx or a fixture.
  - Errors bubble up as Python exceptions. LangGraph will include them
    in the message history so the LLM can react ("retry with different
    inputs" or "give up and report").

NB on tool naming: keep verb-first, snake_case, descriptive. The LLM uses
the name as its primary handle. `calibrate_surrogate` is better than
`do_calibration` or `call_calibrate_endpoint`.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Annotated

import httpx
from langchain.messages import ToolMessage
from langchain.tools import tool, InjectedToolCallId
from langgraph.types import Command

from app.config import get_settings


# ============================================================================
# HTTP client (module-level singleton, swappable in tests)
# ============================================================================
@lru_cache
def _client() -> httpx.Client:
    """Cached httpx client pointed at the surrogate API.

    `lru_cache` makes this a process-wide singleton. Tests override by
    calling `_client.cache_clear()` after monkeypatching the env.
    """
    s = get_settings()
    return httpx.Client(base_url=s.surrogate_api_url, timeout=s.api_timeout_sec)


# ============================================================================
# Tool 1 — fetch market quotes (local fixture, no HTTP)
# ============================================================================
# TODO T1 — implement fetch_market_quotes.
# WHY: in real life this would hit a broker feed (e.g., ICE, Bloomberg).
# For this capstone, read a stub JSON file from examples/sample_market.json.
# This isolates the agent's WORKFLOW logic from data-source plumbing.
#
# PATTERN:
#     @tool
#     def fetch_market_quotes(num_quotes: int = 4) -> list[dict]:
#         """Fetch today's swaption market quotes.
#
#         Use this FIRST in any calibration workflow — calibrate_surrogate
#         needs market quotes as input. Returns up to `num_quotes` quotes,
#         each shaped {"T": float, "K": float, "F": float, "iv": float}.
#         """
#         path = Path(__file__).resolve().parent.parent / "examples" / "sample_market.json"
#         quotes = json.loads(path.read_text())
#         return quotes[:num_quotes]
# ----------------------------------------------------------------------------
@tool
def fetch_market_quotes(num_quotes: int=4, tool_call_id: Annotated[str, InjectedToolCallId] = "") -> Command:
    """Fetch today's swaption market quotes.
    
    Use this FIRST in any calibration workflow — calibrate_surrogate
    needs market quotes as input. Returns up to `num_quotes` quotes,
    each shaped {"T": float, "K": float, "F": float, "iv": float}.
    """
    path = Path(__file__).resolve().parent.parent / "examples" / "sample_market.json"
    quotes = json.loads(path.read_text())
    return Command(
        update={
            "prices": quotes[:num_quotes],
            "messages": [ToolMessage(content=json.dumps(quotes[:num_quotes]), tool_call_id=tool_call_id,)]
        })


# ============================================================================
# Tool 2 — calibrate surrogate (POST /calibrate)
# ============================================================================
# TODO T2 — implement calibrate_surrogate.
# WHY: this is the headline tool — wraps POST /calibrate. The LLM should
# call this AFTER fetch_market_quotes returns a list of quotes.
#
# Input shape (the surrogate API's CalibrateRequest):
#     {"instruments": [{"T":..., "K":..., "F":...}, ...],
#      "market_ivs":  [float, ...]}
#
# Output shape (CalibrateResponse):
#     {"theta_star": {"sig_a":..., "sig_c":..., "sabr_alpha":..., "rho_inf":...},
#      "cost": float, "success": bool, "message": str,
#      "model_version": int,
#      "verify": {"rmse_calib_bp": ..., "rmse_surrogate_bp": ..., "rows": [...]}}
#
# PATTERN:
#     @tool
#     def calibrate_surrogate(quotes: list[dict]) -> dict:
#         """Calibrate the LMM surrogate to a set of market quotes.
#
#         Call this AFTER fetch_market_quotes. Each quote must have T, K, F, iv.
#         Returns calibrated parameters (theta_star) + a verify report with
#         rmse_calib_bp (lower is better; > 50 bp suggests calibration failure).
#         """
#         instruments = [{"T": q["T"], "K": q["K"], "F": q["F"]} for q in quotes]
#         market_ivs  = [q["iv"] for q in quotes]
#         r = _client().post("/calibrate",
#                            json={"instruments": instruments, "market_ivs": market_ivs})
#         r.raise_for_status()
#         return r.json()
# ----------------------------------------------------------------------------
@tool
def calibrate_surrogate(quotes: list[dict], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Calibrate the LMM surrogate to a set of market quotes (TODO T2)."""
    instruments = [{"T": q["T"], "K": q["K"], "F": q["F"]} for q in quotes]
    market_ivs  = [q["iv"] for q in quotes]
    r = _client().post("/calibrate", json={"instruments": instruments, "market_ivs": market_ivs})
    r.raise_for_status()
    return Command(
        update={
            "prices": r.json(),
            "messages": [ToolMessage(content=r.json(), tool_call_id=tool_call_id,)]
        })


# ============================================================================
# Tool 3 — price a swaption (POST /price)
# ============================================================================
# TODO T3 — implement price_swaption.
# WHY: takes calibrated params + new instruments and returns IVs. Call this
# AFTER calibrate_surrogate gives you a theta_star.
#
# Input shape:
#     params: {"sig_a":..., "sig_c":..., "sabr_alpha":..., "rho_inf":...}
#     instruments: [{"T":..., "K":..., "F":...}, ...]
#
# Output shape: {"ivs": [float, ...], "model_version": int}
#
# PATTERN:
#     @tool
#     def price_swaption(params: dict, instruments: list[dict]) -> dict:
#         """Predict implied vols for new swaption instruments using calibrated params.
#
#         Use AFTER calibrate_surrogate. Pass theta_star as `params`.
#         Returns one IV per instrument, in input order.
#         """
#         r = _client().post("/price", json={"params": params, "instruments": instruments})
#         r.raise_for_status()
#         return r.json()
# ----------------------------------------------------------------------------
@tool
def price_swaption(params: dict, instruments: list[dict], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Predict implied vols for new swaption instruments using calibrated params (TODO T3)."""
    r = _client().post("/price", json={"params": params, "instruments": instruments})
    r.raise_for_status()
    return Command(
        update={
            "prices": r.json(),
            "messages": [ToolMessage(content=r.json(), tool_call_id=tool_call_id,)]
        })
