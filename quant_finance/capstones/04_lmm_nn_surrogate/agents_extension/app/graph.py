"""Assemble the StateGraph — nodes, edges, compile.

This is the wiring file. By itself it does no work; it just connects
nodes (the agents + supervisor) with edges (the routing logic).

THE TOPOLOGY:

       START
         ↓
   ┌─────────────┐
   │ supervisor  │ ←──┐
   └─────────────┘    │ (after every worker, back to supervisor)
         │            │
         │  conditional edge (route_from_supervisor)
         ↓            │
   ┌─────────────┐    │
   │  WORKERS    │ ───┘
   │  market     │
   │  calibration│
   │  pricing    │
   │  report     │
   └─────────────┘
         │
         ↓ (when supervisor returns FINISH or step cap hit)
        END

This is the supervisor pattern: workers never call each other directly;
the supervisor always picks who runs next. All inter-worker data flows
through the shared WorkflowState.
"""
from langgraph.graph import END, START, StateGraph

# TODO G0 — uncomment these imports AFTER agents.py / supervisor.py are done.
# from app.agents import WORKERS
# from app.state import WorkflowState
# from app.supervisor import route_from_supervisor, supervisor_node


# ============================================================================
# TODO G1 — build and compile the graph.
#
# PATTERN:
#     def build_graph():
#         g = StateGraph(WorkflowState)
#
#         # Add the supervisor + every worker as a node.
#         g.add_node("supervisor", supervisor_node)
#         for name, agent in WORKERS.items():
#             g.add_node(name, agent)
#
#         # Entry: start at the supervisor.
#         g.add_edge(START, "supervisor")
#
#         # From supervisor, conditionally route to a worker or END.
#         g.add_conditional_edges(
#             "supervisor",
#             route_from_supervisor,
#             {name: name for name in WORKERS} | {END: END},
#         )
#
#         # Every worker returns to the supervisor for the next decision.
#         for name in WORKERS:
#             g.add_edge(name, "supervisor")
#
#         return g.compile()
# ----------------------------------------------------------------------------
def build_graph():
    """Build and compile the supervisor + workers StateGraph (TODO G1)."""
    raise NotImplementedError("TODO G1: build_graph")
