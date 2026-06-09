# NAVIGATION.md — How to read this capstone in VSCode

A guided tour through `agents_extension`: how the multi-agent pieces fit
together, the order to read them, and how to trace one workflow end-to-end
with the LangGraph debugger. Designed so you can sit down with VSCode and
build a complete mental model in **~60 minutes**.

If you've already done the sibling `api_extension` walkthrough, much of
the file-organisation discipline will feel familiar — that's deliberate.

---

## 0. One-time VSCode setup (2 min)

```
Ctrl+Shift+P  →  Python: Select Interpreter
              →  pick .venv/bin/python in this capstone (agents_extension/)
```

Drop a `.vscode/launch.json` at the **workspace root** so debug runs are
one-click (`F5`):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run workflow (debug) — agents_extension",
      "type": "debugpy",
      "request": "launch",
      "module": "examples.run_workflow",
      "args": ["Fetch 2 quotes, calibrate, then price T=1 K=0.035 F=0.035."],
      "cwd": "${workspaceFolder}/quant_finance/capstones/04_lmm_nn_surrogate/agents_extension",
      "python": "${workspaceFolder}/quant_finance/capstones/04_lmm_nn_surrogate/agents_extension/.venv/bin/python",
      "justMyCode": false,
      "console": "integratedTerminal"
    }
  ]
}
```

`justMyCode: false` lets you step INTO LangGraph + LangChain + Claude SDK
code — invaluable for understanding the supervisor's message routing.

Make sure the surrogate API is running in a separate terminal before you
hit F5, or the tools will error out.

---

## 1. The mental model — read this BEFORE the code

### Three distinct workflows that share one codebase

```
┌──────────────────────────────────────────────────────────────────────┐
│ ONE-OFF (script)                                                     │
│   examples/run_workflow.py                                           │
│     ↓                                                                │
│     graph.invoke({...}) → final state                                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ UNIT TESTS (no LLM, no live API)                                     │
│   tests/test_tools.py uses respx to mock httpx                       │
│   Each @tool tested in isolation                                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ E2E TESTS (real Claude, real surrogate API)                          │
│   tests/test_e2e.py drives the full graph against a live system     │
│   Skipped automatically if API not reachable                         │
└──────────────────────────────────────────────────────────────────────┘
```

### The control flow (the headline pattern)

```
START
  ↓
SUPERVISOR (LLM with handoff tools)
  ↓ tool call: transfer_to_X
WORKER X (LLM with ONE domain tool)
  ↓ worker calls its tool, gets result, returns
SUPERVISOR (reads updated state, decides next worker or FINISH)
  ↓ ...loop until FINISH...
END
```

**The supervisor never does domain work itself.** Its only job is routing.
Workers never call each other. State is the only inter-worker channel.

### Layered architecture

```
┌────────────────────────────────────────────────────────────────┐
│  examples/run_workflow.py  ← OUTERMOST — CLI entrypoint        │
│  app/graph.py              ← StateGraph assembly                │
│  app/supervisor.py         ← routing brain                      │
│  app/agents.py             ← worker bodies (LLM + 1 tool each)  │
│  app/tools.py              ← @tool wrappers around HTTP API     │
│  app/prompts.py            ← system prompts (no logic)          │
│  app/state.py              ← shared TypedDict (the data plane)  │
│  app/config.py             ← env-driven Settings                │
└────────────────────────────────────────────────────────────────┘
```

Strict directionality:
- `graph.py` imports from `supervisor.py` + `agents.py`
- `agents.py` imports from `tools.py` + `prompts.py`
- `tools.py` imports from `config.py`
- `state.py` is pure data — imported by everyone, depends on nothing

---

## 2. The file map (3 minutes)

```
agents_extension/
├── README.md                ← spec + how to run; READ FIRST
├── NAVIGATION.md            ← this guide
├── pyproject.toml           ← uv deps (langgraph, langchain-anthropic, httpx)
│
├── app/
│   ├── config.py            ← Settings: ANTHROPIC_API_KEY, SURROGATE_API_URL
│   ├── state.py             ← WorkflowState TypedDict (the data contract)
│   ├── tools.py             ← @tool: fetch_quotes, calibrate, price
│   ├── prompts.py           ← 5 SYSTEM PROMPTS — the LLM's instructions
│   ├── agents.py            ← create_react_agent × 4 workers
│   ├── supervisor.py        ← Supervisor LLM + handoff tools + routing
│   └── graph.py             ← StateGraph wiring + compile
│
├── examples/
│   ├── run_workflow.py      ← CLI runner
│   └── sample_market.json   ← Stub quotes
│
└── tests/
    ├── conftest.py          ← Fixtures (env reset, cache clear, sample_quotes)
    ├── test_tools.py        ← Unit tests w/ respx
    └── test_e2e.py          ← End-to-end against live API
```

---

## 3. Phase 1 — Static reading (15 min)

### 3.1 Read `README.md` (5 min)
Should already be done. Confirm you understand:
- The supervisor pattern + why workers don't call each other
- The role of `WorkflowState` (TypedDict, not Pydantic)
- The role of `@tool` decorators (LLM reads docstring → decides when to call)

### 3.2 Read `app/state.py` (3 min)
```
Ctrl+P → state.py
```
The MOST IMPORTANT FILE for understanding the workflow. Every node reads
and writes here. Notice:
- `messages: Annotated[list, add_messages]` — appends, not replaces
- All other fields are simple types — they REPLACE on each write
- `WorkerName` Literal is the supervisor's vocabulary

### 3.3 Read `app/tools.py` (3 min)
Each `@tool` function's **docstring is part of the prompt**. The LLM
reads the docstring to decide which tool fits the current step. Notice:
- One tool per worker (single responsibility)
- Types in signatures → LLM knows the input/output shape
- `_client()` is module-level → swappable for tests

### 3.4 Read `app/prompts.py` (4 min)
Centralised system prompts. This is where you'd iterate the most in a
real project. Notice:
- The supervisor prompt enumerates workers + when to use each
- Worker prompts are short — they have exactly one job
- The report prompt has explicit MARKDOWN STRUCTURE — that's a contract

---

## 4. Phase 2 — Trace ONE workflow end-to-end (15 min)

Pick "Fetch 1 quote, calibrate, price 1 swaption". Follow the trace.

### 4.1 Start at `examples/run_workflow.py`

```python
graph = build_graph()
result = graph.invoke({"messages": [HumanMessage(question)], "step_count": 0})
```

Hit `F12` on `build_graph` → jumps to `app/graph.py`.

### 4.2 Follow into `build_graph`

Read top to bottom. Notice:
- `StateGraph(WorkflowState)` — the type-checked state container
- `add_node("supervisor", supervisor_node)` — supervisor is just another node
- `add_node(name, agent)` for each worker
- `add_conditional_edges(...)` — the routing rule
- `add_edge(name, "supervisor")` for each worker — workers ALWAYS return to supervisor

### 4.3 Follow into `supervisor_node`

`F12` on `supervisor_node` → `app/supervisor.py`.

The supervisor:
1. Reads `state["messages"]`
2. Prepends `SUPERVISOR_PROMPT` as the system message
3. Invokes the LLM with handoff tools bound
4. Reads the LLM's tool call → maps to a worker name
5. Returns `{"messages": [response], "next": worker_name, "step_count": +1}`

LangGraph then merges that into state and calls `route_from_supervisor`,
which reads `state["next"]` and returns the next node's name.

### 4.4 Follow into a worker (e.g. `calibration_agent`)

`F12` on the `WORKERS` map → `app/agents.py`.

A worker is a `create_react_agent(model=llm, tools=[ONE], prompt=PROMPT)`.
When invoked, it:
1. Reads `state["messages"]` (full history)
2. Calls the LLM with its prompt + tool bound
3. If LLM emits a tool call, runs the tool, feeds result back, loops
4. Returns the updated messages

The result of the tool (e.g., calibration JSON) flows back through
messages — the next worker can read it.

### 4.5 Back-trace your reading history

- **Linux/Windows**: `Alt+Left`
- **macOS**: `Ctrl+-`

Step back from `tools.py` → `agents.py` → `supervisor.py` → `graph.py`
→ `run_workflow.py`. You've now seen the entire chain.

---

## 5. Phase 3 — Debugger trace (the killer step, 20 min)

The static read tells you WHAT the code does. The debugger shows you
HOW the agents reason. This is the biggest jump in understanding.

### 5.1 Set breakpoints

- `app/supervisor.py` — first line of `supervisor_node` (right before LLM call)
- `app/supervisor.py` — after `response = supervisor.invoke(...)` (to inspect tool calls)
- `app/tools.py` — first line of `calibrate_surrogate`
- `app/graph.py` — first line of `build_graph`

### 5.2 Hit F5

You're paused in `build_graph`. F10 through it. Watch the graph assemble.

F5 to continue. Next pause: the supervisor's first invocation.

### 5.3 Inspect the supervisor's reasoning

Before the LLM call:
- `state["messages"]` should have one HumanMessage (your question)
- Hover `messages[0].content` to read it

After the LLM call:
- Inspect `response.tool_calls` — this is the supervisor's DECISION
- Note `tool_calls[0]["name"]` — should be `transfer_to_market_data_agent`

Continue (F5). Next pause: inside `calibrate_surrogate` (after market data
returns). Inspect `quotes` — see the actual market quotes flowing.

### 5.4 The "aha moment"

You'll see the supervisor make 4-5 LLM calls — one per routing decision.
Between them, workers run (each their own LLM call + tool call). The
**state grows** as each worker writes back.

You can step through the full conversation and see, message by message,
how a high-level question turns into a sequence of HTTP calls + an LLM
summary. **This is what production agent systems look like under the
covers.**

### 5.5 Look at the Call Stack

While paused inside a tool:
```
calibrate_surrogate                ← currently here
  → tool execution                 ← LangGraph's tool runner
    → calibration_agent            ← worker
      → graph.invoke / .stream     ← StateGraph runtime
        → main                     ← your CLI
```

Click frames to inspect that scope.

---

## 6. Phase 4 — Tests as ground-truth documentation (5 min)

```
Ctrl+P → tests/test_tools.py
Ctrl+P → tests/test_e2e.py
```

`test_tools.py` documents the @tool contracts (input shape → output shape).
`test_e2e.py` documents the workflow contract (question → final state).

Run them:
```bash
uv run pytest tests/ -v
```

A green run = the workflow does what the docs say.

---

## 7. Useful shortcuts (memorise 5)

| Shortcut | What | When |
|---|---|---|
| `Ctrl+P` | Quick file open | Jump anywhere |
| `Ctrl+T` | Workspace symbol search | "Where is `build_graph`?" |
| `F12` / `Ctrl+Click` | Go to definition | Follow imports |
| `Alt+Left` (Linux/Win) / `Ctrl+-` (mac) | Back in reading history | Bounce back after dive |
| `F5` / `F10` / `F11` | Continue / step-over / step-in | Active debugging |
| `Ctrl+Shift+O` | File outline | Skim symbols |

---

## 8. Self-test — answer these without re-reading

1. **What's the role of `state["next"]`?** (Supervisor writes it; `route_from_supervisor` reads it to pick the next node.)
2. **Why is `messages` Annotated with `add_messages`?** (To APPEND each node's response to the history instead of overwriting.)
3. **What stops the supervisor from looping forever?** (`max_supervisor_steps` in Settings + `route_from_supervisor` returns END if exceeded.)
4. **Why does each worker have exactly one tool?** (Forces specialisation; supervisor routing becomes trivial; testable in isolation.)
5. **How does the supervisor "decide" which worker to call?** (Its LLM is bound to handoff tools via `bind_tools(..., tool_choice="any")`. The tool the LLM picks IS the routing decision.)
6. **What happens if `calibrate_surrogate` returns `success: false`?** (CORE: nothing — the report agent surfaces it in prose. STRETCH ST1 adds a ValidatorAgent that loops back.)

If you can answer 5/6 without scrolling back, you've internalised the
project.

---

## 9. Where to go next

- **Implement STRETCH ST1** (validator agent + retry loop) — your first
  real conditional routing
- **Switch to `langgraph-supervisor`** (`pip install langgraph-supervisor`)
  — compare context size + ease of customisation
- **Add streaming** (`graph.stream(...)`) — watch agents think in real time
- **Wrap as FastAPI** — same shape as the sibling `api_extension/`

---

## TL;DR — 60-min reading sequence

| Time | Phase | What |
|---|---|---|
| 0-5 min | 1   | Read `README.md` |
| 5-10 min | 2  | Skim file tree |
| 10-15 min | 3.2-3.3 | `state.py` + `tools.py` |
| 15-20 min | 3.4 | `prompts.py` |
| 20-30 min | 4   | Trace one workflow with F12 / Alt+Left |
| 30-50 min | 5   | Debugger walkthrough |
| 50-55 min | 6   | Read tests |
| 55-60 min | 8   | Self-test |
