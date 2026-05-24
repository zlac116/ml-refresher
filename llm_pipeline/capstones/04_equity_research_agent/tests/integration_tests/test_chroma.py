from src.equity_research.tools.market_data import get_price_summary, get_fundamentals
print(get_price_summary.invoke({"ticker": "AAPL"}))
# Should print a dict with last_price, change_1y_pct, etc.

print(get_fundamentals.invoke({"ticker": "AAPL"}))
# Should print a dict with market_cap, trailingPE, etc.

from src.equity_research.tools.retriever import retrieve_filings
print(retrieve_filings.invoke({"ticker": "AAPL", "query": "supply chain risk"}))
# Should print a list of chunk dicts.