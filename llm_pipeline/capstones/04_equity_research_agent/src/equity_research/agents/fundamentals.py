"""Fundamentals sub-agent.

YOU BUILD THIS (~5 min). Skill exercised: ReAct agent construction.

What to build:
    A ReAct agent that uses the market_data tools and is briefed by
    FUNDAMENTALS_PROMPT.

Hint: langgraph.prebuilt.create_react_agent(model, tools, prompt=...)
"""
from langchain.agents import create_agent
from equity_research.tools.market_data import get_price_summary, get_fundamentals
from equity_research.prompts import FUNDAMENTALS_PROMPT
from equity_research.configuration import Configuration


fundamentals_agent = create_agent(
    model=Configuration.subagent_model,
    tools=[get_price_summary, get_fundamentals],
    system_prompt=FUNDAMENTALS_PROMPT,
    name="fundamentals_agent",
)
