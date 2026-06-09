# Capstone 04 — agents_extension

A multi-agent **swaption desk assistant** built with LangGraph. A team of
Claude-powered agents takes a natural-language question, orchestrates the
**LMM surrogate API** (the sibling `api_extension/` capstone), and returns
a structured answer + report.

This extension demonstrates the **tool-based supervisor pattern** — the
current (2026) LangGraph-team recommendation for multi-agent systems. You
build the supervisor manually with handoff tools rather than using the
`langgraph-supervisor` prebuilt; the manual approach makes the routing
mechanics explicit and gives full control over context engineering.

---

## Why this exists

You've built the surrogate API (`api_extension/`). Now LET AGENTS DRIVE IT.
The same `/calibrate` + `/price` endpoints can be invoked by:
- A human via Swagger UI
- A Python script (`requests.post(...)`)
- **An LLM that decides for itself which endpoint to call** — that's this capstone

The educational payoff is internalising the multi-agent control flow:
- How a supervisor routes work via tool calls
- How state flows between specialised workers
- How tools wrap external APIs as LLM-callable functions
- How to bound a workflow (max steps, conditional exits)

---

## 1. The mental model — READ THIS BEFORE THE CODE

```
USER QUESTION (natural language)
         ↓
   ┌─────────────┐
   │ SUPERVISOR  │  ← LLM with handoff TOOLS only — no domain logic
   └─────────────┘
         ↓ routes to ONE worker per step
   ┌──────────────────┬──────────────────┬──────────────────┐
   │ MarketDataAgent  │ CalibrationAgent │ PricingAgent     │
   │ tool:            │ tool:            │ tool:            │
   │ fetch_quotes()   │ POST /calibrate  │ POST /price      │
   └──────────────────┴──────────────────┴──────────────────┘
                            ↓ supervisor decides "all data gathered"
                    ┌───────────────┐
                    │ ReportAgent   │  ← LLM with NO tools — pure summarisation
                    └───────────────┘
                            ↓
                    FINAL ANSWER (markdown report + structured state)
```

**Two key invariants**:
1. **The supervisor is the only router.** Workers never call each other —
   control always returns to the supervisor between worker calls.
2. **All inter-worker data flows through `WorkflowState`** (a TypedDict).
   Tools write to it; later workers read from it. No back-channels.

**The supervisor pattern vs the swarm pattern** (just so you know): in a
*swarm*, workers can hand off directly to each other. That's more flexible
but harder to debug + reason about. The supervisor pattern centralises
routing, which is easier to audit ("why did it call X?" → check the
supervisor's tool call in messages).

---

## 2. Setup (2 min)

```bash
# In this directory:
cd quant_finance/capstones/04_lmm_nn_surrogate/agents_extension

# Install deps with uv
uv sync

# Configure secrets
cp .env.example .env
# Then edit .env: set ANTHROPIC_API_KEY
```

In **another terminal**, start the LMM surrogate API:

```bash
cd ../api_extension
uv run uvicorn app.main:app --reload --port 8003
```

Verify it's up: `curl http://localhost:8003/` → should return JSON.

---

## 3. Ordered build steps (~2.5h CORE)

Build in this order — each step builds on the previous. Skip the STRETCH
section entirely on your first pass.

| # | File | TODO | Time | What you'll write |
|---|---|---|---|---|
| 1 | `app/state.py`            | S1   | 5 min  | `WorkflowState` TypedDict with `messages` (add_messages reducer), `market_quotes`, `calibration`, `prices`, `final_report`, `next`, `step_count` |
| 2 | `app/tools.py`            | T1   | 5 min  | `fetch_market_quotes` — reads `examples/sample_market.json` |
| 3 | `app/tools.py`            | T2   | 10 min | `calibrate_surrogate` — POSTs to `/calibrate`, returns JSON |
| 4 | `app/tools.py`            | T3   | 5 min  | `price_swaption` — POSTs to `/price`, returns JSON |
| 5 | `app/prompts.py`          | P1–P5 | 20 min | 5 system prompts (supervisor + 4 workers). The prompt IS your context engineering. |
| 6 | `app/agents.py`           | A1–A4 | 10 min | 4 `create_agent` calls (from `langchain.agents` — the LangChain 1.x replacement for the deprecated `langgraph.prebuilt.create_react_agent`), each bound to ONE tool (or none, for report) |
| 7 | `app/supervisor.py`       | SUP1 | 10 min | 4 handoff tools (`transfer_to_X`) + `finish` tool |
| 8 | `app/supervisor.py`       | SUP2–SUP4 | 15 min | Supervisor LLM + node + routing edge function |
| 9 | `app/graph.py`            | G1   | 10 min | `build_graph()` — `StateGraph` with supervisor + workers + conditional edges |
| 10 | `examples/run_workflow.py` | R1   | 10 min | CLI: invoke graph, print every message with rich Panels |
| 11 | `tests/test_tools.py`     | TT1–TT3 | 15 min | Unit tests with respx mocks |
| 12 | `tests/test_e2e.py`       | TE1  | 10 min | End-to-end against live API |

**Smoke test after step 10**:

```bash
uv run python -m examples.run_workflow \
  "Fetch 2 market quotes, calibrate, then price a 1y ATM swaption."
```

Expected: you see the supervisor's tool calls, each worker's response,
and a final markdown report.

---

## 4. Each file's job (one line each)

```
app/
  config.py        → Settings (env-backed via pydantic-settings); read-only after import
  state.py         → WorkflowState TypedDict — the SHARED MEMORY across all agents
  tools.py         → 3 @tool wrappers around the surrogate API; the LLM's "vocabulary"
  prompts.py       → System prompts for every agent (5 in CORE); the LLM's "instructions"
  agents.py        → create_react_agent × 4; each worker = LLM + prompt + ONE tool
  supervisor.py    → Handoff tools + supervisor LLM + routing decision logic
  graph.py         → StateGraph wiring; nodes = agents, edges = routing rules

examples/
  run_workflow.py  → CLI: invoke the graph and pretty-print the conversation
  sample_market.json → Stub market data (4 quotes); replace with real feed in prod

tests/
  conftest.py      → Fixtures (env setup, cache clearing, sample_quotes)
  test_tools.py    → Unit-test the 3 @tool wrappers with respx (mocked httpx)
  test_e2e.py      → End-to-end: real Claude + real API + real graph
```

---

## 5. IGNORE for CORE — STRETCH goals

Don't touch these until CORE is green. They each unlock a real LangGraph
concept but cost time you may not have:

| Stretch | What | Why nontrivial |
|---|---|---|
| ST1 | **Validator agent + retry loop** — reads `verify.rmse_calib_bp`, routes back to CalibrationAgent if > 50bp | Teaches conditional edges + state-driven loops |
| ST2 | **HITL approval** before `/promote` — pause for user input via `interrupt()` | Teaches LangGraph checkpointing |
| ST3 | **LangSmith tracing** — set `LANGSMITH_TRACING=true`, get a click-through trace | Production observability |
| ST4 | **FastAPI wrapper** exposing `POST /workflow` — same shape as api_extension | Productionising agent workflows |
| ST5 | **Streaming output** — use `graph.stream(...)` + Rich `Live`; user sees agents thinking | Better UX, deeper graph understanding |

---

## 6. Test it

```bash
# Unit tests (fast, no API needed)
uv run pytest tests/test_tools.py -v

# End-to-end (requires running API + Claude)
uv run pytest tests/test_e2e.py -v

# Everything
uv run pytest -v
```

Test debugging recipes are in [`../../../toolkit/pytest_debug_cheatsheet.md`](../../../../toolkit/pytest_debug_cheatsheet.md).

---

## 7. Running locally — the golden path

```bash
# Terminal 1: surrogate API
cd ../api_extension
uv run uvicorn app.main:app --reload --port 8003

# Terminal 2: agent workflow
cd ../agents_extension
uv run python -m examples.run_workflow \
  "Fetch 4 market quotes, calibrate the LMM, then price 1y/2y/5y/10y ATM swaptions. Report concisely."
```

You'll see:
1. Supervisor picks `market_data_agent` → quotes appear
2. Supervisor picks `calibration_agent` → theta_star + RMSE appear
3. Supervisor picks `pricing_agent` → IVs appear
4. Supervisor picks `report_agent` → markdown summary
5. Supervisor picks `FINISH` → workflow ends

If any step misroutes, **read the supervisor's tool call message** —
that's where its reasoning lives. Iterate on `SUPERVISOR_PROMPT`.

---

## 8. Where to go after CORE

- **Read `NAVIGATION.md`** — a 60-min reader's guide to the assembled code,
  showing the actual flow with breakpoint-friendly traces.
- **Add ValidatorAgent (ST1)** — first real conditional routing.
- **Add HITL (ST2)** — teaches LangGraph checkpointing, which is essential
  for any production agent system.
- **Switch to `langgraph-supervisor`** — replace your manual supervisor
  with the prebuilt; compare context size + control trade-offs.
- **Read the parent project's `LESSONS.md`** for connecting patterns
  (LangChain LCEL ↔ FastAPI dependency injection ↔ LangGraph state graph).
