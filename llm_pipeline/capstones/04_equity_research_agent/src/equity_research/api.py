"""FastAPI wrapper around the LangGraph supervisor.

YOU BUILD THIS (~30 min). Skill exercised: LangGraph in a real service.

Required endpoints:
    POST /research
        Body: {ticker, question, thread_id?}
        Kicks off the graph. Will pause at the HITL interrupt — return the
        draft report and ask the caller to confirm via /research/approve.

    POST /research/approve
        Body: {thread_id, approved, feedback?}
        Resumes a paused run with Command(resume={...}).

    GET /health
        Trivial 200 OK.

Hints:
    - Build the initial state dict carefully — all keys in ResearchState must
      be present or LangGraph will complain. Empty strings / None for the rest.
    - When the graph hits interrupt(), the result dict will have an
      "__interrupt__" key containing the value you passed to interrupt().
      Check for that, return draft to the caller, persist thread_id so they
      can come back to /research/approve.
    - Use Command(resume={...}) from langgraph.types to resume.

Run:
    uvicorn src.api:app --reload --port 8000
    Open http://127.0.0.1:8000/docs to try it interactively.
"""

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from equity_research.graph import builder

# build the FastAPI app
app = FastAPI()

graph = builder.compile(checkpointer=InMemorySaver())

@app.post("/research")
def start_research(ticker: str, question: str, thread_id: str = None):
    prompt = f"Research request for {ticker}: {question}"
    config = {"configurable": {"thread_id": thread_id if thread_id else "1"}}
    response = graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    
    # Check if we hit the interrupt (HITL) and return appropriate response
    if "__interrupt__" in response:
        return {
            "status": "pending_approval",
            "draft_report": response["__interrupt__"][-1].value["draft_report"],
        }
    else:
        return {
            "status": "completed",
            "report": response["draft_report"],
        }

@app.post("/research/approve")
def approve_research(approved: bool, thread_id: str=None, feedback: str = None):
    config = {"configurable": {"thread_id": thread_id if thread_id else "1"}}
    
    state = graph.get_state(config=config)
    
    if not state.next:
        raise HTTPException(
            status_code=409,
            detail=f"No run awaiting approval for thread_id:{thread_id}"
        )
        
    response = graph.invoke(Command(resume={"approved": approved, "feedback": feedback}), config=config)
    return {
        "status": "approved" if approved else "rejected",
        "report": response["draft_report"],
    }

@app.get("/health")
def health():
    return {"status": 200, "message": "OK"}
