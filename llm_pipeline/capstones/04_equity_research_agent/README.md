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
| 2 | `src/equity_research/tools/market_data.py` | Two `@tool` functions calling `yfinance` | LangChain `@tool` |
| 3 | `src/equity_research/tools/retriever.py` | `@tool` that queries Chroma, filtered by ticker | RAG retrieval |
| 4 | `src/equity_research/agents/fundamentals.py` | `create_agent` bound to market_data tools | Agent w/ tools |
| 5 | `src/equity_research/agents/news.py` | `create_agent` bound to `TavilySearch` | Web search inside an agent |
| 6 | `src/equity_research/agents/filings.py` | `create_agent` bound to `retrieve_filings` | RAG inside an agent |
| 7 | `src/equity_research/graph.py` | Intent extractor + supervisor + sub-agent nodes + HITL `interrupt()` + `MemorySaver` | **Multi-agent + HITL + memory + structured-output intent extraction (the main piece)** |
| 8 | `src/equity_research/api.py` | FastAPI with `POST /research` + `POST /research/approve` | LangGraph in a service |

**Files provided so you don't burn time on boilerplate** (don't modify):

- `pyproject.toml`, `.env.example`, `langgraph.json` — config
- `src/equity_research/__init__.py` — loads `.env` once on first import (centralised)
- `src/equity_research/configuration.py` — `Configuration` dataclass: model choices, tool budgets, flags
- `src/equity_research/state.py` — `ResearchState` TypedDict (per-run mutable state)
- `src/equity_research/prompts.py` — starter prompts for all four roles
- `data/filings/{AAPL,MSFT,NVDA}.txt` — sample 10-K excerpts
- `tests/integration_tests/test_smoke.py` — run after building to verify

---

## WHEN YOU'RE DONE

You've succeeded if all six are true:

1. ✅ `uv run langgraph dev` opens Studio and shows the graph (supervisor + 3 sub-agents + finalise).
2. ✅ A query in Studio pauses at the HITL `interrupt()` before issuing the recommendation; approving with `{"approved": true}` resumes it.
3. ✅ Using the same `thread_id` for a follow-up question reuses prior state.
4. ✅ `uv run uvicorn equity_research.api:app` exposes both endpoints; both work from http://127.0.0.1:8000/docs.
5. ✅ LangSmith traces exist for runs on AAPL, MSFT, NVDA (project `equity-research-agent`, **Traces** tab).
6. ✅ `uv run pytest -v` is green.

---

## STEP-BY-STEP

### Step 0 — Environment (5 min)

```bash
cd /home/zlac116/Code/learning/ml-revision/llm_pipeline/capstones/04_equity_research_agent
cp .env.example .env
# Edit .env: OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY
#            LANGSMITH_TRACING=true
#            LANGSMITH_PROJECT=equity-research-agent
uv sync
```

Verify: `uv run python -c "from equity_research.configuration import Configuration; print('ok')"` prints `ok`.

### Step 1 — RAG ingestion (20 min) → builds file #1

Open `scripts/ingest_filings.py`. Read the docstring. Implement `main()`.

```bash
uv run python scripts/ingest_filings.py
```

**Done when:** `data/chroma/` exists with files inside it.

### Step 2 — Tools (15 min) → builds files #2 and #3

Open `src/equity_research/tools/market_data.py` and `src/equity_research/tools/retriever.py`. Read each docstring. Implement.

Quick test:

```python
from equity_research.tools.market_data import get_price_summary
print(get_price_summary.invoke({"ticker": "AAPL"}))

from equity_research.tools.retriever import retrieve_filings
print(retrieve_filings.invoke({"ticker": "AAPL", "query": "supply chain risk"}))
```

**Done when:** both calls return populated structured data.

### Step 3 — Three sub-agents (15 min) → builds files #4, #5, #6

Open each of the three agent files. Read each docstring. Implement using `create_agent` from `langchain.agents` (the LangChain 1.x canonical agent factory).

Quick test:

```python
from equity_research.agents.fundamentals import fundamentals_agent
from langchain_core.messages import HumanMessage
result = fundamentals_agent.invoke({"messages": [HumanMessage("Ticker: AAPL. Get fundamentals.")]})
print(result["messages"][-1].content)
```

**Done when:** each of the three agents runs end-to-end and returns a sensible answer.

### Step 4 — Graph (60 min) → builds file #7 (the main piece)

Open `src/equity_research/graph.py`. **Read the docstring carefully** — it lists every node, every edge, and the five LangGraph patterns to implement:

1. **Intent extractor node** that reads the latest `HumanMessage` and populates `state["ticker"]` + `state["question"]` via structured output. This is the **entry adapter** — translates chat-UI input (messages only) into the typed state fields the rest of the graph relies on. Without it, Studio sends *"Analyse AAPL"* and the sub-agents search for a ticker called `"None"`.
2. **Supervisor node** that picks the next sub-agent (or `finalise`) — LLM with structured output for the routing decision.
3. **Sub-agent nodes** (fundamentals / news / filings) that invoke each pre-built agent and write notes back to state.
4. **Finalise node** that calls `interrupt()` for human approval; branches on approve vs feedback (revises with feedback + original draft in context).
5. **`MemorySaver` checkpointer** so the same `thread_id` keeps state.

```bash
uv run langgraph dev
```

In Studio, type a query like *"Analyse AAPL"*. The intent extractor turns it into `ticker="AAPL"`. The sub-agents loop, the supervisor routes, and the run **must** pause at the HITL interrupt. Resume with `{"approved": true}` → final recommendation appears. Try *"Send me feedback"*-style resume too to exercise the revision branch.

**Done when:** Studio shows the graph, the chat-UI input populates state correctly, a run pauses at HITL, both approve and feedback resume paths work, and a follow-up question on the same `thread_id` reuses prior state.

### Step 5 — FastAPI service (30 min) → builds file #8

Open `src/equity_research/api.py`. Read the docstring. Implement the two endpoints.

```bash
uv run uvicorn equity_research.api:app --reload --port 8000
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

Capture LangSmith trace URL for each (Traces tab, project `equity-research-agent`). Then:

```bash
uv run pytest -v
```

Add a 5-line reflection at the bottom of this README: which step took longest, which design choice you'd change, one thing the prototype doesn't yet do.

---

## STRUCTURE

Production-style layout matching the canonical LangGraph CLI template (`langgraph new`):

```
04_equity_research_agent/
├── README.md
├── pyproject.toml                              # name = "equity-research-agent"
├── langgraph.json                              # graph = ./src/equity_research/graph.py:graph
├── .env.example
├── src/
│   └── equity_research/                        # the importable package
│       ├── __init__.py                         # load_dotenv on import (centralised)
│       ├── configuration.py                    # Configuration dataclass (per-run config)
│       ├── state.py                            # ResearchState TypedDict (per-run mutable state)
│       ├── prompts.py                          # all system prompts
│       ├── graph.py                            # supervisor — YOU BUILD (step 4)
│       ├── api.py                              # FastAPI    — YOU BUILD (step 5)
│       ├── agents/
│       │   ├── fundamentals.py                 # YOU BUILD (step 3)
│       │   ├── news.py                         # YOU BUILD (step 3)
│       │   └── filings.py                      # YOU BUILD (step 3)
│       └── tools/
│           ├── market_data.py                  # YOU BUILD (step 2)
│           └── retriever.py                    # YOU BUILD (step 2)
├── scripts/
│   └── ingest_filings.py                       # YOU BUILD (step 1)
├── data/
│   ├── filings/                                # sample 10-Ks (provided)
│   └── chroma/                                 # built by step 1
└── tests/
    ├── unit_tests/                             # deterministic — no live LLM calls
    └── integration_tests/                      # live — gated by pytest marker
        └── test_smoke.py
```

**Why this layout** (production best practice, not opinion):

- `src/<package_name>/` (not bare `src/`) — matches the `langgraph new` template; lets the project install as a real package; eliminates the `sys.path` hacks bare `src/` requires.
- **State vs configuration separation** — `state.py` holds what changes during a run (messages, intermediate notes); `configuration.py` holds what's fixed at invocation (model name, tool budgets, feature flags). LangGraph 0.3+ treats them as different concepts via `Runtime[Context]`.
- **`load_dotenv()` in package `__init__.py`** — runs once when the package is first imported, before any LangChain module resolves API keys. No scattered `load_dotenv()` calls in every file.
- **Tests split into `unit_tests/` and `integration_tests/`** — unit tests must be deterministic (mock the LLM); integration tests can hit live LLMs but are gated by a marker.
- **Functions, not classes** — the LangGraph canonical pattern. Nodes are pure functions of state; OOP appears only in Pydantic/dataclass data containers (`Configuration`, `ResearchState`). No agent classes, service classes, or manager classes.

---

## CONFIGURATION USAGE

Read configuration inside a graph node:

```python
from langgraph.runtime import Runtime
from equity_research.configuration import Configuration

def supervisor_node(state, runtime: Runtime[Configuration]):
    model_name = runtime.context.supervisor_model
    ...
```

Override at invocation:

```python
graph.invoke(
    {"ticker": "AAPL", ...},
    config={"configurable": {"subagent_model": "openai:gpt-5-mini"}},
)
```

This is what separates an "experiment" from a "system" — config knobs are explicit, not hard-coded across modules.

---

## OUT OF SCOPE

Not part of the 4-hour build. Add if you spend a second day:

- Eval suite for briefing quality
- Persistent checkpointer (`SqliteSaver` / `PostgresSaver`) — currently in-memory
- Cost controls / token budgets
- Prompt-injection hardening on news content
- Failover when OpenAI / Tavily is unavailable
- Multi-tenant isolation

---

## PRE-REQUISITES

- Python 3.12+, `uv` installed
- `.env` with `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=equity-research-agent`

Don't overshoot 4h — cut scope, not the project.

## Demo runs (langsmith traces)
- AAPL: https://smith.langchain.com/public/4146d51c-df2f-49fb-9fb3-06796e4980fc/r
- MSFT: https://smith.langchain.com/public/a462fafc-2403-42ec-a536-d16ffba357eb/r
- NVDA: https://smith.langchain.com/public/15902668-a29e-4fa7-b736-f488ac576869/r

## Reflection
1. Longest step: extract_intent, news, fundamental, filings
2. Design choice I'd change: Unsure
3. Not done yet: Unsure