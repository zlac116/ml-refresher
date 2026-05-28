"""Equity research graph — entry point for LangGraph Studio.

YOU BUILD THIS (~60 min). This is the core skill of the capstone.

Required nodes:
    extract_intent_node(state) → reads the latest HumanMessage from
                                  state["messages"] and populates
                                  state["ticker"] + state["question"] via
                                  structured output. Acts as the entry
                                  adapter — translates chat-UI input
                                  (messages only) into typed state fields
                                  so the rest of the graph has what it needs.
    supervisor_node(state)     → decides next sub-agent OR "finalise"
    fundamentals_node(state)   → invokes the fundamentals sub-agent
    news_node(state)           → invokes the news sub-agent
    filings_node(state)        → invokes the filings sub-agent
    finalise_node(state)       → writes draft, calls interrupt() for HITL,
                                  then either returns the draft or revises
                                  using human feedback (with the original
                                  draft passed in as context)

Required edges:
    START → extract_intent → supervisor
    supervisor → (via Command.goto) → {fundamentals, news, filings, finalise}
    {fundamentals, news, filings} → supervisor
    finalise → END

Required pieces (skills being exercised):
    1. Intent extraction with structured output:
       Pydantic schema with `ticker: str` and `question: str` fields.
       LLM call: init_chat_model(...).with_structured_output(IntentSchema)
       reading state["messages"][-1].content. Returns a dict update for
       state["ticker"] and state["question"]. Handle the "no ticker found"
       case explicitly (e.g. Optional[str] + a clarify branch, or default
       to an empty string + check in the supervisor).
    2. Supervisor LLM with structured output (Literal of the four next
       steps) — routes via Command(goto=next_step + "_node").
    3. Each sub-agent node: invoke the agent and write its summary back to
       state under the matching key (fundamentals_notes / news_notes /
       filings_notes).
    4. HITL: inside finalise_node, after generating the draft, call
            from langgraph.types import interrupt
            decision = interrupt({"draft_report": draft, "instruction": ...})
       Then branch on decision["approved"] vs decision["feedback"].
       Pass the original draft + feedback to the revision LLM call so the
       revision happens in context, not in isolation.
    5. Memory: compile with InMemorySaver() as checkpointer so the same
       thread_id keeps state across runs (production: SqliteSaver /
       PostgresSaver).

Export `graph` at module level — langgraph.json points Studio at it.

References:
    - StateGraph: https://langchain-ai.github.io/langgraph/how-tos/state-reducers/
    - Command-based routing: https://langchain-ai.github.io/langgraph/concepts/low_level/#command
    - interrupt (HITL): https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/
    - Checkpointer / persistence: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    - Structured output: https://python.langchain.com/docs/concepts/structured_outputs/
"""

from typing import Literal, Optional
from langchain.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field

from equity_research.state import ResearchState
from equity_research.prompts import SUPERVISOR_PROMPT, FINALISE_PROMPT, INTENT_PROMPT
from equity_research.agents.fundamentals import fundamentals_agent
from equity_research.agents.news import news_agent
from equity_research.agents.filings import filings_agent
from equity_research.configuration import Configuration

class IntentSchema(BaseModel):
    ticker: Optional[str] = Field(description="The US-equity ticker symbol (1-5 uppercase letters) mentioned in the user's message. None if no ticker can be confidently identified.")
    question: str = Field(description="The research question or analysis the user wants answered, rewritten as a clear directive.")

# Output schema for supervisor node
class SupervisorSchema(BaseModel):
    next_step: Literal["fundamentals", "news", "filings", "finalise"]

MEMORY = InMemorySaver()

# Node functions
def extract_intent_node(state: ResearchState) -> ResearchState:
    """Extract ticker and question from user query."""
    
    llm = init_chat_model(model=Configuration.intent_model).with_structured_output(IntentSchema)
    
    messages = state.get("messages") or []
    if not messages:
        return {"ticker": "", "question": ""}
    
    user_text = messages[-1].content
    
    prompt = f"""Extract the ticker and question from the following user query:
    
    user query: {user_text}
    """
    
    response = llm.invoke([SystemMessage(content=INTENT_PROMPT), HumanMessage(content=prompt)])
    
    return {
        "ticker": response.ticker,
        "question": response.question
    }

def supervisor_node(state: ResearchState) -> Command[Literal["fundamentals_node", "news_node", "filings_node", "finalise_node"]]:
    """Look at state and decide which node to go to next."""
    
    if not state.get("ticker"):
        return Command(
            update={"next_step": "finalise"},
            goto="finalise_node"
        )
    
    llm = init_chat_model(model=Configuration.supervisor_model).with_structured_output(SupervisorSchema)
    
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
builder.add_node("extract_intent_node", extract_intent_node)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("fundamentals_node", fundamentals_node)
builder.add_node("news_node", news_node)
builder.add_node("filings_node", filings_node)
builder.add_node("finalise_node", finalise_node)

# Add Edges
builder.add_edge(START, "extract_intent_node")
builder.add_edge("extract_intent_node", "supervisor_node")
builder.add_edge("filings_node", "supervisor_node")
builder.add_edge("news_node", "supervisor_node")
builder.add_edge("fundamentals_node", "supervisor_node")
builder.add_edge("finalise_node", END)

# TODO: build the graph and export `graph` at module level
graph = builder.compile(checkpointer=MEMORY)

if __name__ == "__main__":
    
    query = "Is $AAPL a good investment right now?"
    
    from IPython.display import Image, display
    
    print(graph.get_graph().draw_ascii())
    
    # messages = [HumanMessage(content=query)]
    # messages = graph.invoke({"messages": messages})
    # for m in messages["messages"]:
    #     m.pretty_print()