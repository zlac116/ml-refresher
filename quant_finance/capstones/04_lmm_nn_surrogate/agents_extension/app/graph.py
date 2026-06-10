"""Assemble the StateGraph — nodes, edges, compile.

TOPOLOGY (canonical handoff pattern):

       START
         ↓
   ┌─────────────┐
   │ supervisor  │ ←──┐
   └─────────────┘    │  (each worker returns to the supervisor)
         │            │
         │ Command(goto="X", graph=Command.PARENT)
         ↓            │
   ┌─────────────┐    │
   │  WORKER X   │ ───┘
   └─────────────┘
         ↓ Command(goto=END, graph=Command.PARENT)  [from finish() tool]
        END

No conditional edges from supervisor — its handoff tools' `Command(goto=...)`
returns drive routing directly. Workers always edge back to supervisor; the
supervisor decides next via its handoff tools' goto.

Refs:
  - Handoffs:  https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
  - StateGraph: https://docs.langchain.com/oss/python/langgraph/use-graph-api
"""
from langgraph.graph import START, StateGraph

from app.agents import WORKERS
from app.state import WorkflowState
from app.supervisor import supervisor


def build_graph():
    """Build and compile the supervisor + workers StateGraph."""
    g = StateGraph(WorkflowState)

    g.add_node("supervisor", supervisor)
    for worker_name, agent in WORKERS.items():
        g.add_node(worker_name, agent)

    g.add_edge(START, "supervisor")
    for worker_name in WORKERS:
        g.add_edge(worker_name, "supervisor")

    return g.compile()
