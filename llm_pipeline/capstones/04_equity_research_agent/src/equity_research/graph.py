"""Supervisor graph — entry point for LangGraph Studio.

YOU BUILD THIS (~50 min). This is the core skill of the capstone.

Required nodes:
    supervisor_node(state)   → decides next sub-agent OR "finalise"
    fundamentals_node(state) → invokes the fundamentals sub-agent
    news_node(state)         → invokes the news sub-agent
    filings_node(state)      → invokes the filings sub-agent
    finalise_node(state)     → writes draft, calls interrupt() for HITL,
                                then either returns the draft or revises
                                using human feedback

Required edges:
    START → supervisor
    supervisor → conditional → {fundamentals, news, filings, finalise}
    {fundamentals, news, filings} → supervisor
    finalise → END

Required pieces (skills being exercised):
    1. Supervisor LLM call with a routing prompt (see SUPERVISOR_PROMPT).
       It should look at which notes are already populated and decide next.
       Either parse a free-text response OR use structured output with a
       Pydantic schema (Literal["fundamentals", "news", "filings", "finalise"]).
    2. Conditional edge using `add_conditional_edges` keyed on state["next_step"].
    3. Each sub-agent node: invoke the agent and write its summary back to
       state under the matching key (fundamentals_notes / news_notes /
       filings_notes).
    4. HITL: inside finalise_node, after generating the draft, call
            from langgraph.types import interrupt
            decision = interrupt({"draft_report": draft, "instruction": ...})
       Then branch on decision["approved"] vs decision["feedback"].
    5. Memory: compile with MemorySaver() as checkpointer so the same
       thread_id keeps state across runs.

Export `graph` at module level — langgraph.json points Studio at it.

References:
    - StateGraph: https://langchain-ai.github.io/langgraph/how-tos/state-reducers/
    - Conditional edges: https://langchain-ai.github.io/langgraph/how-tos/branching/
    - interrupt (HITL): https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/
    - MemorySaver: https://langchain-ai.github.io/langgraph/how-tos/persistence/
"""

# from typing import Literal
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.types import interrupt
#
# from equity_research.state import ResearchState
# from equity_research.prompts import SUPERVISOR_PROMPT, FINALISE_PROMPT
# from equity_research.agents.fundamentals import fundamentals_agent
# from equity_research.agents.news import news_agent
# from equity_research.agents.filings import filings_agent


# TODO: build the graph and export `graph` at module level
graph = None
