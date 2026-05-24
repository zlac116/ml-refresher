from equity_research.agents.fundamentals import fundamentals_agent
from equity_research.agents.filings import filings_agent
from equity_research.agents.news import news_agent
from langchain_core.messages import HumanMessage

result = fundamentals_agent.invoke({"messages": [HumanMessage("Ticker: AAPL. Get fundamentals.")]})
print(result["messages"][-1].content)

# result = filings_agent.invoke({"messages": [HumanMessage(content="Ticker: AAPL. Supply chain risk")]})
# print(result["messages"][-1].content)

# result = news_agent.invoke({"messages": [HumanMessage(content="Ticker: AAPL. Recent news")]})
# print(result["messages"][-1].content)