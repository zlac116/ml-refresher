"""Worker agents — each is a small ReAct agent bound to ONE tool.

WHY ONE TOOL PER WORKER:
  - Forces specialisation; the supervisor's routing decision becomes obvious
  - Cleaner messages — each worker either calls its tool or hands back
  - Easier to test in isolation (mock one tool, assert one worker's output)

`create_react_agent` is LangGraph's prebuilt for "LLM + tools" loops. It
handles the tool-calling protocol (LLM emits tool_calls → graph runs tool
→ feeds result back → LLM continues until no more tool calls).

The agent's NAME (the `name=` kwarg) is how the supervisor refers to it.
Keep names stable + descriptive — they appear in the supervisor's tools.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.prompts import (
    CALIBRATION_PROMPT,
    MARKET_DATA_PROMPT,
    PRICING_PROMPT,
    REPORT_PROMPT,
)
from app.tools import calibrate_surrogate, fetch_market_quotes, price_swaption


def _llm() -> ChatAnthropic:
    """Construct the agent LLM. Centralised so all workers share config."""
    s = get_settings()
    return ChatAnthropic(
        model=s.agent_model,
        api_key=s.anthropic_api_key,
        temperature=0,           # deterministic for testing + reproducibility
    )


# ============================================================================
# TODO A1 — market_data_agent.
# PATTERN:
#     market_data_agent = create_react_agent(
#         model=_llm(),
#         tools=[fetch_market_quotes],
#         name="market_data_agent",
#         prompt=MARKET_DATA_PROMPT,
#     )
# ----------------------------------------------------------------------------
# market_data_agent = ...   # TODO A1


# ============================================================================
# TODO A2 — calibration_agent.
# PATTERN: same shape as A1 with calibrate_surrogate tool + CALIBRATION_PROMPT.
# ----------------------------------------------------------------------------
# calibration_agent = ...   # TODO A2


# ============================================================================
# TODO A3 — pricing_agent.
# PATTERN: same shape with price_swaption tool + PRICING_PROMPT.
# ----------------------------------------------------------------------------
# pricing_agent = ...       # TODO A3


# ============================================================================
# TODO A4 — report_agent.
# WHY no tools: the report agent only reads state and writes a summary.
# It's the only worker without tools.
#
# PATTERN:
#     report_agent = create_react_agent(
#         model=_llm(),
#         tools=[],            # no tools — pure summarisation
#         name="report_agent",
#         prompt=REPORT_PROMPT,
#     )
# ----------------------------------------------------------------------------
# report_agent = ...        # TODO A4


# ============================================================================
# Convenience map — used by graph.py to wire workers into the StateGraph.
# Uncomment + populate AFTER A1-A4 are defined.
# ============================================================================
# WORKERS = {
#     "market_data_agent": market_data_agent,
#     "calibration_agent": calibration_agent,
#     "pricing_agent":     pricing_agent,
#     "report_agent":      report_agent,
# }
