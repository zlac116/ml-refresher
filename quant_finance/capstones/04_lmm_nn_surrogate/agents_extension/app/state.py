"""LangGraph state schema.

THE CENTRAL OBJECT in any LangGraph workflow. Every node reads from and
writes to this state. Nodes return a partial dict; LangGraph merges that
into the previous state using the field's reducer (default = replace).

WHY TypedDict (not Pydantic):
  - LangGraph's canonical pattern in the docs
  - No serialization overhead — the graph passes dicts internally
  - `Annotated[list, add_messages]` is the canonical way to ACCUMULATE
    messages across nodes; trivially expressed in TypedDict
  - You still get IDE autocomplete from the type hints

WHY this specific shape:
  - `messages` — the conversation history; ALL agents read this
  - `market_quotes` — set by MarketDataAgent, read by CalibrationAgent
  - `calibration` — set by CalibrationAgent (theta_star + verify report),
                    read by PricingAgent and ReportAgent
  - `prices` — set by PricingAgent, read by ReportAgent
  - `final_report` — set by ReportAgent at the very end
  - `next` — the supervisor sets this to pick the next worker
  - `step_count` — guard against runaway loops (max_supervisor_steps)
"""
from typing import Annotated, Any, Literal, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

# Names workers can be routed to. Keep in sync with agents.py.
WorkerName = Literal[
    "market_data_agent",
    "calibration_agent",
    "pricing_agent",
    "report_agent",
    "FINISH",
]


# ----------------------------------------------------------------------------
# TODO S1 — define the WorkflowState TypedDict.
# WHY: every node in graph.py reads/writes fields from here. Get this
# wrong and you'll spend an hour debugging "why is my data not flowing?".
#
# PATTERN:
#
#     class WorkflowState(TypedDict, total=False):
#         # Accumulating field — add_messages reducer APPENDS instead of REPLACES.
#         # Without `Annotated[..., add_messages]`, each node's return would
#         # overwrite the entire history.
#         messages: Annotated[list, add_messages]
#
#         # Working data — written by one agent, consumed by the next.
#         market_quotes: list[dict] | None     # MarketDataAgent → CalibrationAgent
#         calibration:   dict | None           # CalibrationAgent → Pricing/Report
#         prices:        list[dict] | None     # PricingAgent → ReportAgent
#         final_report:  str | None            # ReportAgent (terminal)
#
#         # Supervisor's routing decision (read by the conditional edge in graph.py).
#         next: WorkerName
#
#         # Safety: hard cap on supervisor iterations.
#         step_count: int
#
# NB on the shape of `calibration` / `prices`:
#   - calibration: {"theta_star": {...}, "cost": float, "success": bool,
#                   "model_version": int, "verify": {"rmse_calib_bp": ...}}
#   - prices: [{"instrument": {...}, "iv": float}, ...]
# Match this when you implement the tools in tools.py.
# ----------------------------------------------------------------------------
class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    market_quotes: list[dict] | None
    calibration: dict | None
    prices: list[dict] | None
    final_report: str | None
    next: WorkerName | None
    step_count: int
