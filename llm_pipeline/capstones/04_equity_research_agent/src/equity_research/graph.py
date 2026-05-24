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

from typing import Literal
from langchain.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel

from equity_research.state import ResearchState
from equity_research.prompts import SUPERVISOR_PROMPT, FINALISE_PROMPT
from equity_research.agents.fundamentals import fundamentals_agent
from equity_research.agents.news import news_agent
from equity_research.agents.filings import filings_agent
from equity_research.configuration import Configuration

# Output schema for supervisor node
class SupervisorOutput(BaseModel):
    next_step: Literal["fundamentals", "news", "filings", "finalise"]

MEMORY = InMemorySaver()

# Node functions
def supervisor_node(state: ResearchState) -> Command[Literal["fundamentals_node", "news_node", "filings_node", "finalise_node"]]:
    """Look at state and decide which node to go to next."""
    
    llm = init_chat_model(model=Configuration.supervisor_model).with_structured_output(SupervisorOutput)
    
    structured_response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT), 
        HumanMessage(content=f"""State:
                     ticker: {state.get('ticker')}
                     question: {state.get('question')}
                     filings_notes: {state.get('filings_notes')}
                     news_notes: {state.get('news_notes')}
                     fundamentals_notes: {state.get('fundamentals_notes')}
                     """)
        ])
    
    next_step = structured_response.next_step
    
    return Command(
        update={"next_step": next_step},
        goto=next_step + "_node"
    )

def fundamentals_node(state: ResearchState) -> ResearchState:
    """Invoke the fundamentals agent and write summary back to state."""
    response = fundamentals_agent.invoke({"messages": [HumanMessage(content=f"Summerise the fundamentals for {state.get('ticker')}")]})
    
    return {"fundamentals_notes": response["messages"][-1].content}

def news_node(state: ResearchState) -> ResearchState:
    """Invoke the news agent and write summary back to state."""
    
    response = news_agent.invoke({"messages": [HumanMessage(content=f"Summerise the news for {state.get('ticker')}")]})
    
    return {"news_notes": response["messages"][-1].content}

def filings_node(state: ResearchState) -> ResearchState:
    """Invoke the filings agent and write summary back to state."""
    
    response = filings_agent.invoke({"messages": [HumanMessage(content=f"Summerise the filings for {state.get('ticker')}")]})
    
    return {"filings_notes": response["messages"][-1].content}

def finalise_node(state: ResearchState) -> ResearchState:
    """Write draft, call interrupt() for HITL, then either return the draft or revise using human feedback."""
    
    prompt = f"""
    Draft a response to the following question for {state.get('ticker')}, using the filing, fundamentals
    and news information.
    
    question: {state.get('question')}
    
    filings: {state.get('filings_notes')}
    fundamentals: {state.get('fundamentals_notes')}
    news: {state.get('news_notes')}
    """
    
    llm = init_chat_model(model=Configuration.finaliser_model)
    
    draft_response = llm.invoke([SystemMessage(content=FINALISE_PROMPT), HumanMessage(content=prompt)])
    
    decision = interrupt({
        "draft_report": draft_response.content,
        "instruction": "Approve the draft or provide feedback for revision."
    })
    
    if decision.get("approved", False):
        return {"draft_report": draft_response.content, "approved": True}
    else:
        revised = llm.invoke([SystemMessage(content=FINALISE_PROMPT), HumanMessage(content=f"Original draft:\n{draft_response.content}\n\nRevise with this feedback: {decision.get('feedback', '')}")])
    
        return {
            "draft_report": revised.content,
            "approved": decision.get("approved", False),
        }


# Add Nodes
builder = StateGraph(ResearchState)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("fundamentals_node", fundamentals_node)
builder.add_node("news_node", news_node)
builder.add_node("filings_node", filings_node)
builder.add_node("finalise_node", finalise_node)

# Add Edges
builder.add_edge(START, "supervisor_node")
builder.add_edge("filings_node", "supervisor_node")
builder.add_edge("news_node", "supervisor_node")
builder.add_edge("fundamentals_node", "supervisor_node")
builder.add_edge("finalise_node", END)

# TODO: build the graph and export `graph` at module level
graph = builder.compile(checkpointer=MEMORY)

