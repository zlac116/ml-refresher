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

# from fastapi import FastAPI
# from langchain_core.messages import HumanMessage
# from langgraph.types import Command
# from equity_research.graph import graph


# TODO: build the FastAPI app
app = None
