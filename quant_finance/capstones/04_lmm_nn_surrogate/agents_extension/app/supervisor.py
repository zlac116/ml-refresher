"""The supervisor — routes the workflow by emitting a `next` worker name.

CURRENT 2026 RECOMMENDATION (per LangGraph team): build the supervisor
using tool-calling rather than the prebuilt `langgraph-supervisor` library.
The tool approach gives you more control over context engineering, prompt
shape, and routing logic. The prebuilt is a good convenience but harder
to customise.

Two design choices made here:

  1. **Handoff tools**: each worker has a corresponding handoff tool the
     supervisor can call (e.g., `transfer_to_calibration_agent`). When
     the supervisor's LLM emits a tool call, the graph routes to that
     worker. This is symmetric and easy to read.

  2. **Single-step supervisor**: the supervisor is invoked once per
     orchestration step; it picks ONE worker, the worker runs, then
     control returns to the supervisor. This is the canonical pattern —
     do NOT let the supervisor call multiple workers in parallel until
     you understand the message-ordering implications.

The graph wires it all up in graph.py.
"""
from langchain_anthropic import ChatAnthropic
from langchain.messages import ToolMessage, AIMessage, HumanMessage
from langchain.tools import tool
from langgraph.graph import END

from app.config import get_settings
from app.prompts import SUPERVISOR_PROMPT
from app.state import WorkerName, WorkflowState


# ============================================================================
# Handoff tools — the supervisor's "vocabulary".
# Each tool returns the name of the worker to route to. The graph reads
# the tool call and uses its name as the next node.
# ============================================================================
# TODO SUP1 — implement the four handoff tools + FINISH.
#
# WHY tools (not free text): the LLM's tool-calling API is more reliable
# than parsing free text like "next: calibration_agent". Tool calls are
# validated against your declared schema.
#
# PATTERN — one tool per worker, plus FINISH:
#
#     @tool
#     def transfer_to_market_data_agent() -> str:
#         """Hand off to the market data agent to fetch quotes."""
#         return "market_data_agent"
#
#     @tool
#     def transfer_to_calibration_agent() -> str:
#         """Hand off to the calibration agent to calibrate to market quotes."""
#         return "calibration_agent"
#
#     @tool
#     def transfer_to_pricing_agent() -> str:
#         """Hand off to the pricing agent to price new instruments."""
#         return "pricing_agent"
#
#     @tool
#     def transfer_to_report_agent() -> str:
#         """Hand off to the report agent to write the final summary."""
#         return "report_agent"
#
#     @tool
#     def finish() -> str:
#         """End the workflow. Call this when the report is complete."""
#         return "FINISH"
#
#     HANDOFF_TOOLS = [transfer_to_market_data_agent, transfer_to_calibration_agent,
#                      transfer_to_pricing_agent,     transfer_to_report_agent,
#                      finish]
# ----------------------------------------------------------------------------
@tool
def transfer_to_market_data_agent() -> str:
    """Hand off to the market data agent to fetch quotes."""
    return "market_data_agent"

@tool
def transfer_to_calibration_agent() -> str:
    """Hand off to the calibration agent to calibrate to market quotes."""
    return "calibration_agent"

@tool
def transfer_to_pricing_agent() -> str:
    """Hand off to the pricing agent to price new instruments."""
    return "pricing_agent"

@tool
def transfer_to_report_agent() -> str:
    """Hand off to the report agent to write the final summary."""
    return "report_agent"

@tool
def finish() -> str:
    """End the workflow. Call this when the report is complete."""
    return "FINISH"

HANDOFF_TOOLS = [transfer_to_market_data_agent, transfer_to_calibration_agent,
                 transfer_to_pricing_agent,     transfer_to_report_agent,
                 finish]


# ============================================================================
# The supervisor LLM — uses HANDOFF_TOOLS to pick the next worker.
# ============================================================================
# TODO SUP2 — build the supervisor LLM.
#
# PATTERN:
#     def make_supervisor():
#         s = get_settings()
#         llm = ChatAnthropic(
#             model=s.agent_model,
#             api_key=s.anthropic_api_key,
#             temperature=0,
#         )
#         # bind_tools forces the LLM to emit a tool call (or refuse).
#         return llm.bind_tools(HANDOFF_TOOLS, tool_choice="any")
# ----------------------------------------------------------------------------
def make_supervisor():
    s = get_settings()
    llm = ChatAnthropic(
        model=s.agent_model,
        api_key=s.anthropic_api_key,
        temperature=0,
    )
    return llm.bind_tools(HANDOFF_TOOLS, tool_choice="any")


# ============================================================================
# The supervisor NODE — reads state, calls the supervisor LLM, returns the
# chosen worker name (or END) so graph.py's conditional edge can route.
# ============================================================================
# TODO SUP3 — implement the supervisor node function.
#
# WHY a node (not just an edge): we want the supervisor's reasoning in
# the message history (so the LLM can see why it picked X). Adding a
# node lets us record the supervisor's tool call as a message.
#
# PATTERN:
#     def supervisor_node(state: WorkflowState) -> dict:
#         supervisor = make_supervisor()
#         messages   = [{"role": "system", "content": SUPERVISOR_PROMPT}, *state["messages"]]
#         response   = supervisor.invoke(messages)
#
#         # Extract the worker name from the tool call.
#         tool_call = response.tool_calls[0]
#         next_worker: WorkerName = tool_call["name"].replace("transfer_to_", "")
#         if tool_call["name"] == "finish":
#             next_worker = "FINISH"
#
#         step = state.get("step_count", 0) + 1
#         return {
#             "messages":   [response],   # the supervisor's tool call message
#             "next":       next_worker,
#             "step_count": step,
#         }
# ----------------------------------------------------------------------------
def supervisor_node(state: WorkflowState):
    """Route only. Extraction is handled by per-worker nodes in app/extractors.py."""
    supervisor = make_supervisor()
    if isinstance(state["messages"][-1], AIMessage):
        messages = [{"role": "system", "content": SUPERVISOR_PROMPT}, *state["messages"], HumanMessage(content="Select the next worker")]
    else:
        messages = [{"role": "system", "content": SUPERVISOR_PROMPT}, *state["messages"]]
    
    response = supervisor.invoke(messages)

    tool_call = response.tool_calls[0]
    if tool_call["name"] == "finish":
        next_worker: WorkerName = "FINISH"
    else:
        next_worker = tool_call["name"].replace("transfer_to_", "")

    # Anthropic protocol: every tool_use needs a tool_result in the next message.
    # Handoff "tools" don't do real work, so we synthesise the ack here.
    tool_ack = ToolMessage(
        content=f"Transferred to {next_worker}",
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )

    return {
        "messages":   [response, tool_ack],
        "next":       next_worker,
        "step_count": state.get("step_count", 0) + 1,
    }

# ============================================================================
# Routing helper — read state["next"] and return the next graph node.
# ============================================================================
# TODO SUP4 — the conditional edge function.
#
# WHY separate from supervisor_node: LangGraph wants edges to return a
# string naming the next node. The supervisor stores its decision in
# state["next"]; this function just plucks it out.
#
# Also enforces max_supervisor_steps (safety net).
#
# PATTERN:
#     def route_from_supervisor(state: WorkflowState) -> str:
#         s = get_settings()
#         if state.get("step_count", 0) >= s.max_supervisor_steps:
#             return END
#         choice = state.get("next", "FINISH")
#         return END if choice == "FINISH" else choice
# ----------------------------------------------------------------------------
def route_from_supervisor(state: WorkflowState) -> str:
    s = get_settings()
    if state.get("step_count", 0) >= s.max_supervisor_steps:
        return END
    choice = state.get("next", "FINISH")
    return END if choice == "FINISH" else choice
