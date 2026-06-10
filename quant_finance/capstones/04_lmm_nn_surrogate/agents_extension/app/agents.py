"""Worker agents — each is a small ReAct-style agent bound to ONE tool.

WHY ONE TOOL PER WORKER:
  - Forces specialisation; the supervisor's routing decision becomes obvious
  - Cleaner messages — each worker either calls its tool or hands back
  - Easier to test in isolation (mock one tool, assert one worker's output)

NB on the LangChain 1.x API:
  - Use `langchain.agents.create_agent` (NOT the deprecated
    `langgraph.prebuilt.create_react_agent`). The new function lives in
    LangChain 1.x and adds the optional middleware system. Same return
    type (CompiledStateGraph) — drops directly into a StateGraph node.
  - Kwarg rename: `prompt=` is now `system_prompt=`.
  - The agent's NAME (the `name=` kwarg) is how the supervisor refers to
    it — keep names stable + descriptive; they appear in the supervisor's
    handoff tools (see supervisor.py).

`create_agent` handles the tool-calling loop (LLM emits tool_calls →
graph runs tool → feeds result back → LLM continues until no more tool
calls).
"""
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from app.config import get_settings
from app.prompts import (
    CALIBRATION_PROMPT,
    MARKET_DATA_PROMPT,
    PRICING_PROMPT,
    REPORT_PROMPT,
)
from app.tools import calibrate_surrogate, fetch_market_quotes, price_swaption


def _llm() -> ChatAnthropic:
    """Construct the agent LLM. Centralised so all workers share config.

    `create_agent` also accepts a provider-prefixed string like
    "anthropic:claude-haiku-4-5-20251001" — but passing a ChatAnthropic
    instance lets you set temperature, max_tokens, etc.
    """
    s = get_settings()
    return ChatAnthropic(
        model=s.agent_model,
        api_key=s.anthropic_api_key,
        temperature=0,           # deterministic for testing + reproducibility
    )


# ============================================================================
# TODO A1 — market_data_agent.
# PATTERN:
#     market_data_agent = create_agent(
#         model=_llm(),
#         tools=[fetch_market_quotes],
#         name="market_data_agent",
#         system_prompt=MARKET_DATA_PROMPT,
#     )
# ----------------------------------------------------------------------------
# market_data_agent = ...   # TODO A1
market_data_agent = create_agent(
    model=_llm(),
    tools=[fetch_market_quotes],
    name="market_data_agent",
    system_prompt=MARKET_DATA_PROMPT,
)

# ============================================================================
# TODO A2 — calibration_agent.
# PATTERN: same shape as A1 with calibrate_surrogate tool + CALIBRATION_PROMPT.
# ----------------------------------------------------------------------------
# calibration_agent = ...   # TODO A2
calibration_agent = create_agent(
    model=_llm(),
    tools=[calibrate_surrogate],
    name="calibration_agent",
    system_prompt=CALIBRATION_PROMPT,
)

# ============================================================================
# TODO A3 — pricing_agent.
# PATTERN: same shape with price_swaption tool + PRICING_PROMPT.
# ----------------------------------------------------------------------------
# pricing_agent = ...       # TODO A3
pricing_agent = create_agent(
    model=_llm(),
    tools=[price_swaption],
    name="pricing_agent",
    system_prompt=PRICING_PROMPT,
)

# ============================================================================
# TODO A4 — report_agent.
# WHY no tools: the report agent only reads state and writes a summary.
# It's the only worker without tools. `tools=None` is the canonical idiom
# for a pure-LLM agent in create_agent.
#
# PATTERN:
#     report_agent = create_agent(
#         model=_llm(),
#         tools=None,          # pure summarisation, no tool loop
#         name="report_agent",
#         system_prompt=REPORT_PROMPT,
#     )
# ----------------------------------------------------------------------------
# report_agent = ...        # TODO A4
report_agent = create_agent(
    model=_llm(),
    tools=None,
    name="report_agent",
    system_prompt=REPORT_PROMPT,
)

# ============================================================================
# Convenience map — used by graph.py to wire workers into the StateGraph.
# Uncomment + populate AFTER A1-A4 are defined.
# ============================================================================
WORKERS = {
    "market_data_agent": market_data_agent,
    "calibration_agent": calibration_agent,
    "pricing_agent":     pricing_agent,
    "report_agent":      report_agent,
}
