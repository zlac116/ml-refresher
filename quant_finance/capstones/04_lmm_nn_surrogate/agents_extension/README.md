# Capstone 04 — agents_extension

A multi-agent **swaption desk assistant** built with LangChain 1.x / LangGraph 1.x.
A team of LLM-powered agents takes a natural-language question, orchestrates the
**LMM surrogate API** (the sibling `api_extension/` capstone), and returns a
structured answer + report.

The whole thing is wired against current `docs.langchain.com` patterns:

- `init_chat_model("openai:gpt-5-mini" | "anthropic:claude-...")` — provider-prefixed model string
- `create_agent(...)` for every agent (supervisor + 4 workers)
- Tool-based **handoff** routing via `Command(goto=..., graph=Command.PARENT)`
- Tools return `Command(update={field: value, "messages": [ToolMessage(...)]})` to populate state
- `state_schema=WorkflowState` on workers so tool updates propagate to parent state
- `@wrap_model_call` middleware to satisfy Anthropic's "conversation must end with user message" rule
- Streaming via `stream_mode=["updates", "values"], version="v2"`

There is NO manual supervisor node, NO `state["next"]` field, NO conditional edges from
supervisor, NO step counter — routing is driven entirely by handoff tools' `Command(goto)`.

---

## Why this exists

You've built the surrogate API (`api_extension/`). Now LET AGENTS DRIVE IT.
The same `/calibrate` + `/price` endpoints can be invoked by:

- A human via Swagger UI
- A Python script (`requests.post(...)`)
- **An LLM that decides for itself which endpoint to call** — that's this capstone

The educational payoff is internalising the canonical multi-agent control flow:

- How a supervisor routes work via handoff tools (`Command(goto, graph=Command.PARENT)`)
- How tools update both messages AND named state fields in one return
- How `state_schema` lets workers participate in the parent's typed state
- How middleware adapts the agent loop to model-specific quirks

---

## 1. The mental model — read this BEFORE the code

```
USER QUESTION (natural language)
         │
         ▼
   ┌─────────────┐
   │ SUPERVISOR  │  create_agent with HANDOFF_TOOLS only
   └─────┬───────┘  (transfer_to_X, finish)
         │ Command(goto="X", graph=Command.PARENT)
         ▼
   ┌──────────────────┬──────────────────┬──────────────────┐
   │ MarketDataAgent  │ CalibrationAgent │ PricingAgent     │
   │ fetch_market_…   │ calibrate_…      │ price_swaption   │
   └──────────────────┴──────────────────┴──────────────────┘
                 │ returns to supervisor via worker→supervisor edge
                 ▼
                supervisor decides "all data gathered"
                 │ Command(goto="report_agent", ...)
                 ▼
         ┌───────────────┐
         │ ReportAgent   │  no tools — pure summarisation
         └─────┬─────────┘
               │ supervisor sees report, calls finish()
               │ Command(goto=END, graph=Command.PARENT)
               ▼
              END
```

**Two key invariants:**

1. **The supervisor is the only router.** Workers never call each other. Each
   worker has an edge back to the supervisor (`graph.add_edge(worker, "supervisor")`).
2. **All inter-worker data flows through `WorkflowState`.** Tools write to it via
   `Command(update=...)`; later agents read it. No back-channels.

**Supervisor pattern vs swarm:** in a swarm, workers can hand off directly to
each other (each agent gets handoff tools too). That's more flexible but harder
to audit. The supervisor pattern centralises routing — every routing decision
is in one agent's tool call, easy to inspect in the message history.

---

## 2. Setup (2 min)

```bash
cd quant_finance/capstones/04_lmm_nn_surrogate/agents_extension

uv sync

cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and/or OPENAI_API_KEY
# Optionally override MODEL="anthropic:claude-haiku-4-5-20251001"
```

In another terminal, start the LMM surrogate API (sibling capstone):

```bash
cd ../api_extension
uv run uvicorn app.main:app --reload --port 8003
```

Verify: `curl http://localhost:8003/` should return the service JSON.

---

## 3. The file map

```
app/
  config.py        Settings (env-backed via pydantic-settings) + get_llm() factory
  state.py         WorkflowState (TypedDict with add_messages reducer)
  tools.py         3 @tool wrappers returning Command(update={...}) — populate state
  prompts.py       5 system prompts: supervisor + 4 workers
  agents.py        4 create_agent workers (state_schema=WorkflowState)
  supervisor.py    5 handoff tools (Command goto, graph=PARENT) + create_agent supervisor
                   + wrap_model_call middleware for Anthropic prefill
  graph.py         StateGraph wiring: START→supervisor; worker→supervisor (×4)

examples/
  run_workflow.py   CLI — streams the workflow with rich panels
  sample_market.json  Stub market data

tests/
  conftest.py       env-reset + sample_quotes fixtures
  test_tools.py     unit tests for tools (TODOs)
  test_e2e.py       full workflow against live API (TODOs)
```

---

## 4. Running locally — golden path

```bash
# Terminal 1: surrogate API
cd ../api_extension
uv run uvicorn app.main:app --reload --port 8003

# Terminal 2: agent workflow
cd ../agents_extension
uv run python -m examples.run_workflow \
  "Fetch 2 market quotes, calibrate the LMM, then price a 1y ATM swaption."
```

You'll see panels in this order:

1. `user` — the question
2. `supervisor (ai)` — empty content + handoff tool call
3. `supervisor (tool)` — "Transferred to market_data_agent"
4. `market_data_agent (ai/tool/ai)` — planning, tool result, summary
5. … repeats for calibration → pricing → report …
6. `supervisor (tool)` — "Workflow complete" (from `finish` tool)
7. `FINAL REPORT` — the report agent's markdown

If a step misroutes, **read the supervisor's tool call message** — that's where
its reasoning lives. Iterate on `SUPERVISOR_PROMPT` in `app/prompts.py`.

---

## 5. Canonical patterns this capstone demonstrates

| Pattern | Where | Reference |
|---|---|---|
| `init_chat_model` factory | `config.py:get_llm()` | https://docs.langchain.com/oss/python/langchain/models |
| `create_agent` + `state_schema` | `agents.py`, `supervisor.py` | https://docs.langchain.com/oss/python/langchain/agents |
| Handoff tools with `Command(goto=, graph=Command.PARENT)` | `supervisor.py` | https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs |
| Tools returning `Command(update={...})` with `ToolRuntime` | `tools.py` | https://docs.langchain.com/oss/python/langchain/tools |
| `add_messages` reducer on TypedDict | `state.py` | https://docs.langchain.com/oss/python/langgraph/use-graph-api |
| `@wrap_model_call` middleware | `supervisor.py:inject_human_after_ai` | https://docs.langchain.com/oss/python/langchain/middleware/custom |
| `stream_mode=["updates","values"], version="v2"` | `examples/run_workflow.py` | https://docs.langchain.com/oss/python/langgraph/streaming |

---

## 6. STRETCH

| # | What | Why nontrivial |
|---|---|---|
| ST1 | **Validator agent + retry loop** — reads `verify.rmse_calib_bp`, hands off back to calibration if > 50bp | Real conditional routing via Command |
| ST2 | **HITL approval** before a `promote` step — pause via `interrupt()` | Teaches LangGraph checkpointing |
| ST3 | **LangSmith tracing** — `LANGSMITH_TRACING=true` + key | Production observability |
| ST4 | **FastAPI wrapper** exposing `POST /workflow` | Productionising an agent system |
| ST5 | **Live UI** — Rich `Live` view of the streaming workflow | Better UX |

---

## 7. Tests

```bash
# Unit tests (fast — no API or LLM needed)
uv run pytest tests/test_tools.py -v

# End-to-end (slow + billable — needs live API + LLM key)
uv run pytest tests/test_e2e.py -v
```

Tests are currently TODO stubs; see file headers for the scaffolding.

Debugging recipes: [`../../../../toolkit/pytest_debug_cheatsheet.md`](../../../../toolkit/pytest_debug_cheatsheet.md).

---

## 8. Where to go after CORE

- **Read `NAVIGATION.md`** — guided tour through the assembled code.
- **Implement at least one STRETCH** — ST1 (validator loop) teaches conditional
  routing via Command; ST2 (HITL) teaches checkpointers.
- **Compare with `langgraph-supervisor` prebuilt** — replace the manual supervisor
  with the prebuilt and see how much code disappears (and what control you lose).
