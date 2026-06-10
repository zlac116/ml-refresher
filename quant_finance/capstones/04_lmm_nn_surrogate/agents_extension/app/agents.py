"""Worker agents — one create_agent per specialist, each bound to ONE tool.

Each worker is a `langchain.agents.create_agent` returning a CompiledStateGraph
that drops directly into the parent StateGraph as a node. Workers DON'T need
handoff tools — their parent edge (`worker → supervisor` in graph.py) returns
control to the supervisor, which then decides the next route via its own
handoff tools.

The report_agent has `tools=None` — it's a pure-LLM summariser; its sole job
is to write the final report based on prior tool outputs already in messages.

Refs:
  - create_agent: https://docs.langchain.com/oss/python/langchain/agents
"""
from langchain.agents import create_agent

from app.config import get_llm
from app.prompts import (
    CALIBRATION_PROMPT,
    MARKET_DATA_PROMPT,
    PRICING_PROMPT,
    REPORT_PROMPT,
)
from app.tools import calibrate_surrogate, fetch_market_quotes, price_swaption


market_data_agent = create_agent(
    model=get_llm(),
    tools=[fetch_market_quotes],
    name="market_data_agent",
    system_prompt=MARKET_DATA_PROMPT,
)

calibration_agent = create_agent(
    model=get_llm(),
    tools=[calibrate_surrogate],
    name="calibration_agent",
    system_prompt=CALIBRATION_PROMPT,
)

pricing_agent = create_agent(
    model=get_llm(),
    tools=[price_swaption],
    name="pricing_agent",
    system_prompt=PRICING_PROMPT,
)

report_agent = create_agent(
    model=get_llm(),
    tools=None,                # pure summarisation; no tool loop
    name="report_agent",
    system_prompt=REPORT_PROMPT,
)


# Map: worker_name → compiled agent. Used by graph.py to register nodes.
WORKERS = {
    "market_data_agent": market_data_agent,
    "calibration_agent": calibration_agent,
    "pricing_agent":     pricing_agent,
    "report_agent":      report_agent,
}
