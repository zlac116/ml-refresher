# Capstone 4 — Equity Research Desk Assistant

## AIM

**Practice the LangChain / LangGraph skills from your two courses by building a working multi-agent system on real US-equity data in 4 hours.**

The system: an equity research desk assistant. The user gives it a ticker and a question. It coordinates three specialist agents (live fundamentals, news, 10-K filings), pauses for a human to approve the draft recommendation, and remembers state across follow-up questions. Deployed as both a LangGraph Studio app and a FastAPI service.

The aim is **building** it, not researching equities. The equity-research framing is the wrapper that makes the LangGraph skills land on a realistic problem.

---

## YOUR TASK

Implement 8 Python files in the order below. Each file is a stub with a docstring describing what to build. **Read each file's docstring before you write code.**

| # | File | What you implement | Skill from your courses |
|---|---|---|---|
| 1 | `scripts/ingest_filings.py` | Read `data/filings/*.txt`, chunk, embed, persist to Chroma | RAG ingestion |
| 2 | `src/tools/market_data.py` | Two `@tool` functions calling `yfinance` | LangChain `@tool` |
| 3 | `src/tools/retriever.py` | `@tool` that queries Chroma, filtered by ticker | RAG retrieval |
| 4 | `src/agents/fundamentals.py` | `create_react_agent` bound to market_data tools | ReAct agent |
| 5 | `src/agents/news.py` | `create_react_agent` bound to `TavilySearch` | Web search inside an agent |
| 6 | `src/agents/filings.py` | `create_react_agent` bound to `retrieve_filings` | RAG inside an agent |
| 7 | `src/graph.py` | Supervisor + sub-agent nodes + HITL `interrupt()` + `MemorySaver` | **Multi-agent + HITL + memory (the main piece)** |
| 8 | `src/api.py` | FastAPI with `POST /research` + `POST /research/approve` | LangGraph in a service |

**Files I've provided so you don't burn time on boilerplate** (don't modify):

- `pyproject.toml`, `.env.example`, `langgraph.json` — config
- `src/state.py` — the shared `ResearchState` TypedDict your nodes read/write
- `src/prompts.py` — starter prompts (refine if you like)
- `data/filings/{AAPL,MSFT,NVDA}.txt` — sample 10-K excerpts
- `tests/test_smoke.py` — run after building to verify

---

## WHEN YOU'RE DONE

You've succeeded if all six are true:

1. ✅ `uv run langgraph dev` opens Studio and shows the graph (supervisor + 3 sub-agents + finalise).
2. ✅ A query in Studio pauses at the HITL `interrupt()` before issuing the recommendation; approving with `{"approved": true}` resumes it.
3. ✅ Using the same `thread_id` for a follow-up question reuses prior state.
4. ✅ `uv run uvicorn src.api:app` exposes both endpoints; both work from http://127.0.0.1:8000/docs.
5. ✅ LangSmith traces exist for runs on AAPL, MSFT, NVDA.
6. ✅ `uv run pytest -v` is green.

---

## STEP-BY-STEP

### Step 0 — Environment (5 min)

```bash
cd /home/zlac116/Code/learning/ml-revision/llm_pipeline/capstones/04_equity_research_agent
cp .env.example .env
# Edit .env: OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true
uv sync
```

Verify: `uv run python -c "import langgraph; print('ok')"` prints `ok`.

### Step 1 — RAG ingestion (20 min) → builds file #1

Open `scripts/ingest_filings.py`. Read the docstring. Implement `main()`.

```bash
uv run python scripts/ingest_filings.py
```

**Done when:** `data/chroma/` exists with files inside it.

### Step 2 — Tools (15 min) → builds files #2 and #3

Open `src/tools/market_data.py` and `src/tools/retriever.py`. Read each docstring. Implement.

Quick test:

```python
from src.tools.market_data import get_price_summary
print(get_price_summary.invoke({"ticker": "AAPL"}))

from src.tools.retriever import retrieve_filings
print(retrieve_filings.invoke({"ticker": "AAPL", "query": "supply chain risk"}))
```

**Done when:** both calls return populated structured data.

### Step 3 — Three sub-agents (15 min) → builds files #4, #5, #6

Open each of the three agent files. Read each docstring. Implement using `create_react_agent` from `langgraph.prebuilt`.

Quick test:

```python
from src.agents.fundamentals import fundamentals_agent
from langchain_core.messages import HumanMessage
result = fundamentals_agent.invoke({"messages": [HumanMessage("Ticker: AAPL. Get fundamentals.")]})
print(result["messages"][-1].content)
```

**Done when:** each of the three agents runs end-to-end and returns a sensible answer.

### Step 4 — Supervisor graph (60 min) → builds file #7 (the main piece)

Open `src/graph.py`. **Read the docstring carefully** — it lists every node, every edge, and the four LangGraph patterns to implement:

1. Supervisor node that picks the next sub-agent (or `finalise`)
2. Conditional edges driven by `state["next_step"]`
3. Finalise node that calls `interrupt()` for human approval
4. `MemorySaver` checkpointer so the same `thread_id` keeps state

```bash
uv run langgraph dev
```

In Studio, run a query. It **must** pause at the HITL interrupt. Resume with `{"approved": true}` → final recommendation appears.

**Done when:** Studio shows the graph, a run pauses at HITL, resumes correctly, returns a final briefing.

### Step 5 — FastAPI service (30 min) → builds file #8

Open `src/api.py`. Read the docstring. Implement the two endpoints.

```bash
uv run uvicorn src.api:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs.

Test:

1. `POST /research` with `ticker=AAPL`, `question="Is current valuation reasonable?"`, `thread_id=test1`. → Returns paused response with `draft_report`.
2. `POST /research/approve` with `thread_id=test1`, `approved=true`. → Returns final report.

**Done when:** both endpoints work from the `/docs` UI.

### Step 6 — Three demo runs + smoke tests (15 min)

Three queries in Studio (each on its own `thread_id`):

1. `AAPL` — "Is current valuation reasonable given fundamentals and recent news?"
2. `MSFT` — "What does the latest 10-K say about cloud growth risks?"
3. `NVDA` — "Recommendation given news sentiment and supply-chain risk?"

Capture LangSmith trace URL for each. Then:

```bash
uv run pytest -v
```

Add a 5-line reflection at the bottom of this README: which step took longest, which design choice you'd change, one thing the prototype doesn't yet do.

---

## FILES

```
04_equity_research_agent/
├── README.md
├── pyproject.toml
├── .env.example
├── langgraph.json
├── src/
│   ├── state.py                ← provided
│   ├── prompts.py              ← provided
│   ├── graph.py                ← YOU BUILD  (step 4)
│   ├── api.py                  ← YOU BUILD  (step 5)
│   ├── agents/
│   │   ├── fundamentals.py     ← YOU BUILD  (step 3)
│   │   ├── news.py             ← YOU BUILD  (step 3)
│   │   └── filings.py          ← YOU BUILD  (step 3)
│   └── tools/
│       ├── market_data.py      ← YOU BUILD  (step 2)
│       └── retriever.py        ← YOU BUILD  (step 2)
├── scripts/
│   └── ingest_filings.py       ← YOU BUILD  (step 1)
├── data/
│   └── filings/                ← provided
└── tests/
    └── test_smoke.py           ← provided
```

---

## OUT OF SCOPE

Not part of the 4-hour build. Add if you spend a second day:

- Eval suite for briefing quality
- Persistent checkpointer (`SqliteSaver` / `PostgresSaver`)
- Cost controls / token budgets
- Prompt-injection hardening on news content
- Failover when OpenAI / Tavily is unavailable
- Multi-tenant isolation

---

## PRE-REQUISITES

- Python 3.12+, `uv` installed
- `.env` with `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`

Don't overshoot 4h — cut scope, not the project.
