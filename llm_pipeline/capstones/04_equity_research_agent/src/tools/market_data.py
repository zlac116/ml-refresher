"""yfinance-backed tools for the fundamentals agent.

YOU BUILD THIS (~15 min). Skill exercised: LangChain @tool wrappers.

Required tools:
    1. get_price_summary(ticker: str) -> dict
       Return: last_price, 1y % change, 52-week high, 52-week low.
       Use yf.Ticker(ticker).history(period="1y") + .info.

    2. get_fundamentals(ticker: str) -> dict
       Return: market_cap, trailingPE, forwardPE, profitMargins,
               operatingMargins, sector.
       Use yf.Ticker(ticker).info.

Hints:
    - Decorate with @tool from langchain_core.tools — the docstring becomes the
      tool description the LLM sees, so be precise.
    - Return dicts, not strings. The LLM handles structured output better.
    - Add basic error handling for empty/missing data.

References:
    - LangChain tools: https://python.langchain.com/docs/how_to/custom_tools/
    - yfinance: https://github.com/ranaroussi/yfinance
"""

from langchain_core.tools import tool
import yfinance as yf

@tool
def get_price_summary(ticker: str) -> dict:
    """
    Use this tool to retrieve last price, 1y % change, 52-week high and 52-week low
    for a given ticker from yahoo finance.
    Returns a dictionary
    """
    try:
        t = yf.Ticker(ticker)
        history = t.history(period="1y")
        info = t.info
    except Exception as e:
        return {'error': f"Error fetching data for {ticker}: {str(e)}"}
    
    def _f(x): return float(x) if x else None
    
    change_1y_pct = info.get('52WeekChange')
    change_1y_pct = change_1y_pct * 100 if change_1y_pct else None
    
    return {
        'last_price': _f(history['Close'].iloc[-1]),
        'change_1y_pct': _f(change_1y_pct),
        'fiftytwo_week_high': _f(info.get('fiftyTwoWeekHigh')),
        'fiftytwo_week_low': _f(info.get('fiftyTwoWeekLow')),
    }

@tool
def get_fundamentals(ticker: str) -> dict:
    """Use this tool to retrieve ticker fundamentals. Returns a dictionary."""
    try:
        res = yf.Ticker(ticker).info
        return {
            'market_cap': res.get('marketCap'),
            'trailingPE': res.get('trailingPE'),
            'forwardPE': res.get('forwardPE'),
            'profitMargins': res.get('profitMargins'),
            'operatingMargins': res.get('operatingMargins'),
            'sector': res.get('sector'),
        }
    except Exception as e:
        return {'error': f"Error fetching data for {ticker}: {str(e)}"}
