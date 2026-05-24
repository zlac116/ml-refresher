"""News sub-agent.

YOU BUILD THIS (~5 min). Skill exercised: web search tool in an agent.

What to build:
    A ReAct agent that uses Tavily web search (topic="news") and is briefed
    by NEWS_PROMPT.

Hints:
    - langchain_tavily.TavilySearch(max_results=5, topic="news")
    - Set TAVILY_API_KEY in .env
"""
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from equity_research.prompts import NEWS_PROMPT
from equity_research.configuration import Configuration


news_agent = create_agent(
    model=Configuration.subagent_model,
    tools=[
        TavilySearch(
            max_results=Configuration.tavily_max_results,
            topic="news"
        )
    ],
    system_prompt=NEWS_PROMPT,
    name="news_agent",
)
