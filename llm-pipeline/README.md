# llm-pipeline — LangChain + LangGraph cheatsheet

Task-indexed reference covering the full LangChain / LangGraph / LangSmith Python documentation surface.

## Setup

These examples need additional dependencies not in the main `requirements.txt`:

```bash
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-anthropic langsmith langchain-mcp-adapters chromadb
```

Plus environment variables for whichever provider(s) you use:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=my-project
```

## Contents

`langchain_langgraph.ipynb` — 14 sections mirroring the LangChain Python docs structure:

1. **Quickstart** — minimal `create_react_agent` in 10 lines
2. **Models** — `init_chat_model`, structured output, provider switching
3. **Messages** — System/Human/AI/Tool message types
4. **Tools** — `@tool` decorator, structured args, `ToolException`
5. **Agents** — prebuilt `create_react_agent`, system prompts, interrupt-before
6. **LangGraph `StateGraph`** — TypedDict state, nodes, conditional edges
7. **Memory** — `MemorySaver` (short-term), `InMemoryStore` (long-term), persistent checkpointers
8. **RAG** — vector store, retriever-as-tool, document splitters
9. **Multi-agent** — supervisor pattern, shared state
10. **MCP** — `MultiServerMCPClient` for external tool servers
11. **Streaming** — `stream_mode='messages' / 'updates' / 'values'`
12. **Observability (LangSmith)** — env-var setup, tags, metadata
13. **Evaluation (LangSmith)** — datasets, evaluators, LLM-as-judge
14. **Deployment** — FastAPI integration, streaming endpoints

## Note on validation

Cells in this cheatsheet are **not** auto-validated against a live API — they're shown as reference patterns to copy into your own environment. The LangChain ecosystem moves fast; if any API has shifted, cross-check against the linked docs page on each section header.
