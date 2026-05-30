"""Build the Chroma index from data/filings/{TICKER}.txt files.

YOU BUILD THIS (~15 min — first thing, before everything else). Skill
exercised: RAG ingestion pipeline.

Steps:
    1. Glob data/filings/*.txt
    2. For each file, infer ticker = filename stem (uppercased)
    3. Split each file into chunks (RecursiveCharacterTextSplitter,
       chunk_size ~800, overlap ~100)
    4. Wrap chunks as Document with metadata={"ticker": ticker, "source": filename}
    5. Persist via Chroma.from_documents(...) at data/chroma/

Run once:
    uv run python scripts/ingest_filings.py

References:
    - Chroma + embeddings: https://python.langchain.com/docs/integrations/vectorstores/chroma/
    - Text splitters:      https://python.langchain.com/docs/how_to/recursive_text_splitter/
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.equity_research.configuration import Configuration

MAIN_DIR = Path(__file__).parent.parent
CHROMA_DIR = MAIN_DIR.joinpath("data", "chroma")
DATA_DIR = MAIN_DIR.joinpath("data", "filings")

def main() -> None:
    
    # Create Chroma vector store with OpenAI embeddings
    openai_embeddings = OpenAIEmbeddings(model=Configuration.openai_embedding_model)
    vector_store = Chroma(
        collection_name='filings',
        embedding_function=openai_embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    
    # Glob all .txt files in data/filings/
    for file in DATA_DIR.glob("*.txt"):
        ticker = Path(file).stem.upper()
        
        text = file.read_text() # Read the content of the file as text
        
        docs = [Document(page_content=text, metadata={"ticker": ticker, "source": str(file)})]
        chunks = text_splitter.split_documents(docs)
        
        print(f"Adding {len(chunks)} chunks for {ticker} to Chroma...")
        
        vector_store.add_documents(chunks)

if __name__ == "__main__":
    main()
