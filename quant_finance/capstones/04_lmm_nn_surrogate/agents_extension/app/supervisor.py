"""The supervisor — a create_agent with handoff tools.

CANONICAL PATTERN (LangChain 1.x / LangGraph 1.x):

  - The supervisor IS an agent built via `create_agent(model, tools=HANDOFF_TOOLS, ...)`.
    Its LLM sees only the handoff tools (one per worker + a `finish` tool).
    When the LLM emits a tool call, the agent's execution loop runs the
    handoff tool, which returns a `Command(goto=..., graph=Command.PARENT, ...)`
    that routes the PARENT graph to the chosen worker (or to END for finish).

  - There is NO separate `supervisor_node` function, NO `route_from_supervisor`
    edge function, NO `state["next"]` field. Routing is driven entirely by
    the handoff tools' `Command(goto=...)` returns.

  - Each handoff tool's `update["messages"]` must include BOTH:
      1. The supervisor's last AIMessage (the one whose tool_calls contains
         this handoff). When Command(graph=Command.PARENT) escapes the
         supervisor agent's subgraph, the parent state would otherwise miss
         this message — which would break the Anthropic / OpenAI tool-call
         protocol when the worker invokes its LLM (a ToolMessage must have a
         preceding AIMessage with tool_calls).
      2. The synthesised ToolMessage acknowledging the handoff, paired by
         `tool_call_id`.

Refs:
  - Handoffs:    https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
  - create_agent: https://docs.langchain.com/oss/python/langchain/agents
  - Command:     https://docs.langchain.com/oss/python/langgraph/use-graph-api
"""
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage,AIMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.graph import END
from langgraph.types import Command

from app.config import get_llm
from app.prompts import SUPERVISOR_PROMPT


# ============================================================================
# Handoff tools — each emits a Command(goto=worker, graph=Command.PARENT).
# The supervisor's LLM selects one; the tool's execution routes the parent graph.
# ============================================================================
def _last_ai_message(state) -> AIMessage:
    """Pluck the most recent AIMessage from state — the one whose tool_calls
    is asking for THIS handoff. Required in Command.update so the parent
    state preserves the tool_call/tool_result pairing.
    """
    return next(
        msg for msg in reversed(state["messages"])
        if isinstance(msg, AIMessage)
    )


@tool
def transfer_to_market_data_agent(runtime: ToolRuntime) -> Command:
    """Hand off to the market data agent to fetch quotes."""
    return Command(
        goto="market_data_agent",
        update={
            "messages": [
                _last_ai_message(runtime.state),
                ToolMessage(
                    content="Transferred to market_data_agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_calibration_agent(runtime: ToolRuntime) -> Command:
    """Hand off to the calibration agent to calibrate to market quotes."""
    return Command(
        goto="calibration_agent",
        update={
            "messages": [
                _last_ai_message(runtime.state),
                ToolMessage(
                    content="Transferred to calibration_agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_pricing_agent(runtime: ToolRuntime) -> Command:
    """Hand off to the pricing agent to price new instruments."""
    return Command(
        goto="pricing_agent",
        update={
            "messages": [
                _last_ai_message(runtime.state),
                ToolMessage(
                    content="Transferred to pricing_agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_report_agent(runtime: ToolRuntime) -> Command:
    """Hand off to the report agent to write the final summary."""
    return Command(
        goto="report_agent",
        update={
            "messages": [
                _last_ai_message(runtime.state),
                ToolMessage(
                    content="Transferred to report_agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def finish(runtime: ToolRuntime) -> Command:
    """End the workflow. Call this when the report is complete."""
    return Command(
        goto=END,
        update={
            "messages": [
                _last_ai_message(runtime.state),
                ToolMessage(
                    content="Workflow complete",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


HANDOFF_TOOLS = [
    transfer_to_market_data_agent,
    transfer_to_calibration_agent,
    transfer_to_pricing_agent,
    transfer_to_report_agent,
    finish,
]

# ============================================================================
# Middleware. If the last message is an AI message, append a HumanMessage.
# ============================================================================
@wrap_model_call
def inject_human_after_ai(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject a HumanMessage when history ends with an AIMessage.

    Anthropic models without prefill support require the conversation to
    end with a user-role message. After a worker returns, state.messages
    ends with the worker's AIMessage summary — we add a sentinel HumanMessage
    only inside this LLM call, not in durable state.
    """
    if isinstance(request.messages[-1], AIMessage):
        new_messages = request.messages + [HumanMessage(content="Select the next worker.")]
        return handler(request.override(messages=new_messages))
    else:
        return handler(request)
            

# ============================================================================
# The supervisor agent. Drops directly into the parent StateGraph as a node.
# ============================================================================
supervisor = create_agent(
    model=get_llm(),
    tools=HANDOFF_TOOLS,
    system_prompt=SUPERVISOR_PROMPT,
    name="supervisor",
    middleware=[inject_human_after_ai],
)
