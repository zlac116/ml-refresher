"""Chroma-backed RAG retriever for 10-K excerpts.

YOU BUILD THIS (~15 min). Skill exercised: RAG over LangChain Chroma.

What to build:
    A @tool retrieve_filings(ticker, query, k=4) that returns the top-k chunks
    from the Chroma index, filtered by ticker.

Hints:
    - The index is built by scripts/ingest_filings.py — write that first.
    - Use Chroma(persist_directory=..., embedding_function=...).
    - Chroma similarity_search supports a `filter` kwarg — use {"ticker": ticker}.
    - Return a list of dicts: [{"chunk_id": i, "ticker": ..., "text": ...}, ...]
      so the agent's LLM can cite chunk IDs in its answer.

References:
    - langchain-chroma docs:
      https://python.langchain.com/docs/integrations/vectorstores/chroma/
    - Filtering by metadata:
      https://python.langchain.com/docs/how_to/retrievers_chroma/
"""

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"

@tool
def retrieve_filings(ticker: str, query: str, k: int = 4) -> list[dict]:
    """
    Use this tool to retrieve filing chunks relevant to the query and ticker.
    1. Load the Chroma index from CHROMA_DIR, using OpenAIEmbeddings.
    2. Perform a similarity search with the query, filtering by ticker.
    3. Return the top-k results as a list of dicts with keys: chunk_id, ticker, text.
    """
    # Instantiate the Chroma vector store
    vector_store = Chroma(
      collection_name='filings',
      embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
      persist_directory=str(CHROMA_DIR)
    )
    
    # Perform similarity search with filtering by ticker
    response = vector_store.similarity_search(
      query=query,
      k=k,
      filter={'ticker': ticker}
    )
    
    return [
      {'chunk_id': i, 'ticker': d.metadata['ticker'], 'text': d.page_content}
      for i, d in enumerate(response)
    ]