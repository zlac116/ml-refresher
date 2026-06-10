"""Assemble the StateGraph — nodes, edges, compile.

TOPOLOGY:

       START
         ↓
   ┌─────────────┐
   │ supervisor  │ ←──┐
   └─────────────┘    │ (workers return directly to supervisor)
         │            │
         │  conditional edge (route_from_supervisor)
         ↓            │
   ┌─────────────┐    │
   │  WORKER X   │ ───┘
   └─────────────┘
         ↓ (when supervisor returns FINISH or step cap hit)
        END

Inter-worker data flows entirely through state["messages"] (LLMs read the
tool outputs from prior ToolMessages). If you later want named state fields
(state["calibration"], state["prices"], etc.) populated, the canonical
LangGraph 1.x pattern is for tools to return Command(update={...}) — see
docs.langchain.com/oss/python/langchain/tools.
"""
from langgraph.graph import END, START, StateGraph

from app.agents import WORKERS
from app.state import WorkflowState
from app.supervisor import route_from_supervisor, supervisor_node


def build_graph():
    """Build and compile the supervisor + workers StateGraph."""
    g = StateGraph(WorkflowState)

    g.add_node("supervisor", supervisor_node)
    for worker_name, agent in WORKERS.items():
        g.add_node(worker_name, agent)

    g.add_edge(START, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {name: name for name in WORKERS} | {END: END},
    )

    for worker_name in WORKERS:
        g.add_edge(worker_name, "supervisor")

    return g.compile()
