"""Filings sub-agent.

YOU BUILD THIS (~5 min). Skill exercised: RAG-in-an-agent.

What to build:
    A ReAct agent that uses retrieve_filings as its tool and is briefed by
    FILINGS_PROMPT.
"""
from langchain.agents import create_agent
from equity_research.tools.retriever import retrieve_filings
from equity_research.prompts import FILINGS_PROMPT
from equity_research.configuration import Configuration


filings_agent = create_agent(
    model=Configuration.subagent_model,
    tools=[retrieve_filings],
    system_prompt=FILINGS_PROMPT,
    name="filings_agent",
)
