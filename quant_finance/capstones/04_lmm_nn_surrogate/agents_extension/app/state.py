"""LangGraph state schema.

Canonical multi-agent supervisor pattern: routing happens via handoff tools
that return `Command(goto=..., graph=Command.PARENT)` from inside the
supervisor agent. The graph does NOT need a `next` field in state — the
Command's `goto` drives routing directly.

State fields:
  - messages       — full conversation history (workers + supervisor +
                     tool results). Reduced with `add_messages` so each
                     node's return APPENDS rather than REPLACES, and
                     duplicate-id messages are dedupd automatically.
  - market_quotes  — set by fetch_market_quotes tool via Command(update=...)
  - calibration    — set by calibrate_surrogate tool
  - prices         — set by price_swaption tool
  - final_report   — set by report_agent's final AIMessage (read at end)

Refs:
  - State schema:    https://docs.langchain.com/oss/python/langgraph/use-graph-api
  - add_messages:    https://docs.langchain.com/oss/python/langgraph/graph-api#messagesstate
  - Handoffs:        https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
"""
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict, total=False):
    messages:      Annotated[list[AnyMessage], add_messages]
    market_quotes: list[dict] | None
    calibration:   dict | None
    prices:        list[dict] | None
    final_report:  str | None
