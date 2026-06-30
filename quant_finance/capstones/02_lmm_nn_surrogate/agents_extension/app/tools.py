"""@tool wrappers around the LMM surrogate API.

Each tool is a Python function decorated with `@tool`. The LLM reads:
  1. The function name → decides "this looks relevant"
  2. The argument names + type hints → knows what to pass
  3. The docstring → understands WHEN to use it + WHAT it returns

So docstrings are part of the prompt to the LLM. Write them for the model.

Tools return `Command(update={...})` (canonical LangChain 1.x pattern) so the
result both surfaces as a `ToolMessage` for the LLM loop AND populates the
named state field (`market_quotes`, `calibration`, `prices`) that downstream
agents read. `runtime: ToolRuntime` is injected automatically by LangGraph
and gives access to `runtime.tool_call_id` for the matching ToolMessage.

Refs:
  - Tools (Command, ToolRuntime): https://docs.langchain.com/oss/python/langchain/tools
"""
import json
from functools import lru_cache
from pathlib import Path

import httpx
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from app.config import get_settings


@lru_cache
def _client() -> httpx.Client:
    """Cached httpx client pointed at the surrogate API.

    `lru_cache` makes this a process-wide singleton. Tests override by
    calling `_client.cache_clear()` after monkeypatching the env.
    """
    s = get_settings()
    return httpx.Client(base_url=s.surrogate_api_url, timeout=s.api_timeout_sec)


def _last_ai_message(state) -> AIMessage:
    """Pluck the worker's most recent AIMessage — the one whose tool_calls
    is asking for THIS tool call. Required in Command.update when crossing
    to parent via graph=Command.PARENT so the tool_use/tool_result pairing
    is preserved (otherwise the parent gets an orphaned ToolMessage and
    Anthropic/OpenAI reject the next LLM call).
    """
    return next(
        msg for msg in reversed(state["messages"])
        if isinstance(msg, AIMessage)
    )


@tool
def fetch_market_quotes(num_quotes: int = 4, *, runtime: ToolRuntime) -> Command:
    """Fetch today's swaption market quotes.

    Use this FIRST in any calibration workflow — `calibrate_surrogate` needs
    these quotes as input. Returns up to `num_quotes` quotes, each shaped
    `{"T": float, "K": float, "F": float, "iv": float}`. Stored to
    `state["market_quotes"]`.
    """
    path = Path(__file__).resolve().parent.parent / "examples" / "sample_market.json"
    quotes = json.loads(path.read_text())[:num_quotes]
    return Command(
        update={
            "market_quotes": quotes,
            "messages": [
                _last_ai_message(runtime.state),
                ToolMessage(
                    content=json.dumps(quotes),
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def calibrate_surrogate(quotes: list[dict], x0: list[float] | None = None, *, runtime: ToolRuntime) -> Command:
    """Calibrate the LMM surrogate to a list of market quotes.

    Call AFTER `fetch_market_quotes`. Each quote must have T, K, F, iv keys.
    Returns calibrated `theta_star` + a `verify` report with `rmse_calib_bp`
    (lower is better; > 50 bp suggests poor calibration). Stored to
    `state["calibration"]`.
    """
    effective_quotes = runtime.state.get("market_quotes") or quotes
    effective_x0 = runtime.state.get("retry_x0") or x0

    instruments = [{"T": q["T"], "K": q["K"], "F": q["F"]} for q in effective_quotes]
    market_ivs  = [q["iv"] for q in effective_quotes]

    if effective_x0:
        r = _client().post("/calibrate", json={"instruments": instruments, "market_ivs": market_ivs, "x0": effective_x0})
    else:
        r = _client().post("/calibrate", json={"instruments": instruments, "market_ivs": market_ivs})
    r.raise_for_status()
    result = r.json()

    return Command(
        update={
            "calibration": result,
            "messages": [
                _last_ai_message(runtime.state),
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def price_swaption(params: dict, instruments: list[dict], *, runtime: ToolRuntime) -> Command:
    """Predict implied vols for new swaption instruments using calibrated params.

    Use AFTER `calibrate_surrogate`. Pass `theta_star` as `params`. Returns
    one IV per instrument, in input order. Stored to `state["prices"]`.
    """
    r = _client().post("/price", json={"params": params, "instruments": instruments})
    r.raise_for_status()
    result = r.json()
    return Command(
        update={
            "prices": result,
            "messages": [
                _last_ai_message(runtime.state),
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )
