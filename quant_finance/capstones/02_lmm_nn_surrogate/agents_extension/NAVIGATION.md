# NAVIGATION.md — How to read this capstone in VSCode

A guided tour through `agents_extension`: how the multi-agent pieces fit
together, the order to read them, and how to trace one workflow end-to-end
with the LangGraph debugger. Designed for ~60 minutes with VSCode.

This capstone uses the canonical LangChain 1.x / LangGraph 1.x patterns —
see `README.md §5` for the source-doc URL for each one.

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

`justMyCode: false` lets you step INTO LangGraph + LangChain code — invaluable
for understanding the agent execution loop.

Make sure the surrogate API is running in a separate terminal before you hit F5.

---

## 1. The mental model — read this BEFORE the code

### The control flow

```
START
  │
  ▼
SUPERVISOR (create_agent with HANDOFF_TOOLS)
  │ LLM picks a handoff tool → tool returns Command(goto="X", graph=Command.PARENT)
  ▼
WORKER X (create_agent with one domain tool + state_schema=WorkflowState)
  │ tool returns Command(update={field: ..., "messages": [...]})
  │ writes BOTH state field AND ToolMessage
  ▼
SUPERVISOR (via worker→supervisor edge)
  │ LLM decides next: another worker OR finish()
  ▼
... loop ...
  │
  ▼ finish() returns Command(goto=END, graph=Command.PARENT)
END
```

**The supervisor never does domain work itself.** It only routes. Workers never
call each other — every worker has an edge back to supervisor.

### Layered architecture

```
┌────────────────────────────────────────────────────────────────┐
│  examples/run_workflow.py  CLI — graph.stream + rich panels    │
│  app/graph.py              StateGraph assembly + compile        │
│  app/supervisor.py         supervisor agent + handoff tools     │
│                            + wrap_model_call middleware         │
│  app/agents.py             4 worker agents (state_schema=…)     │
│  app/tools.py              @tool wrappers (Command(update=…))   │
│  app/prompts.py            5 system prompts (data only)         │
│  app/state.py              WorkflowState TypedDict              │
│  app/config.py             Settings + get_llm()                 │
└────────────────────────────────────────────────────────────────┘
```

Strict directionality (no circular imports):

- `graph.py` imports `supervisor` + `WORKERS`
- `supervisor.py` imports `get_llm`, `SUPERVISOR_PROMPT`
- `agents.py` imports `get_llm`, the 4 worker prompts, the 3 tools, `WorkflowState`
- `tools.py` imports `get_settings` (only for `_client()`)
- `state.py` is pure data — depends on nothing in this project

---

## 2. The file map

```
agents_extension/
├── README.md                 spec + how to run; READ FIRST
├── NAVIGATION.md             this guide
├── pyproject.toml            uv deps (langchain, langgraph, httpx, rich)
├── .env.example              ANTHROPIC_API_KEY / OPENAI_API_KEY / MODEL / SURROGATE_API_URL
│
├── app/
│   ├── config.py             Settings + get_settings() + get_llm()
│   ├── state.py              WorkflowState TypedDict (messages + 4 data fields)
│   ├── tools.py              3 @tool wrappers; each returns Command(update={...})
│   ├── prompts.py            5 system prompts (one constant per agent)
│   ├── agents.py             4 create_agent workers + WORKERS map
│   ├── supervisor.py         5 handoff tools + supervisor create_agent
│   │                         + inject_human_after_ai @wrap_model_call middleware
│   └── graph.py              build_graph(): START→supervisor; worker→supervisor (×4)
│
├── examples/
│   ├── run_workflow.py       CLI: graph.stream + rich Panels
│   └── sample_market.json    4 stub swaption quotes
│
└── tests/
    ├── conftest.py           env-reset, sample_quotes fixture
    ├── test_tools.py         unit tests (TODO)
    └── test_e2e.py           full workflow against live API (TODO)
```

---

## 3. Phase 1 — Static reading (15 min)

### 3.1 Read `README.md` (5 min)

Confirm you understand:

- The handoff pattern (Command goto, graph=Command.PARENT)
- WHY workers have `state_schema=WorkflowState` (so tool updates flow to parent state)
- WHY the supervisor has a `wrap_model_call` middleware (Anthropic prefill rule)

### 3.2 Read `app/state.py` (3 min)

The most important file. Every node reads and writes to this TypedDict.

```
messages       Annotated[list, add_messages] — APPENDS (and dedupes by id)
market_quotes  set by fetch_market_quotes (via Command(update=...))
calibration    set by calibrate_surrogate
prices         set by price_swaption
final_report   set by — currently NOT populated (see README §STRETCH for option 1)
```

`add_messages` is the canonical reducer — without it, every node return would
OVERWRITE the message list instead of appending.

### 3.3 Read `app/tools.py` (3 min)

Each tool returns `Command(update={...})` with BOTH:

- A named state field (e.g. `"market_quotes": quotes`)
- A `ToolMessage` paired by `runtime.tool_call_id` (so the LLM loop sees the result)

`runtime: ToolRuntime` is injected by LangGraph automatically — never passed
explicitly by your code or the LLM.

Note `*` before `runtime` — it's keyword-only, so callers can't accidentally
pass it positionally.

### 3.4 Read `app/prompts.py` (4 min)

Five constants. Notice the `CALIBRATION_PROMPT` is much more constrained than
the others ("DO NOT… write prose, markdown…") — that's deliberate. Without those
constraints the calibration agent hallucinates pricing analysis it doesn't have
a tool for.

---

## 4. Phase 2 — Trace ONE workflow end-to-end (15 min)

Question: "Fetch 2 quotes, calibrate, then price T=1, K=0.035, F=0.035."

### 4.1 Start at `examples/run_workflow.py`

```python
graph = build_graph()
for chunk in graph.stream(initial_state, stream_mode=["updates","values"], version="v2"):
    ...
```

`F12` on `build_graph` → `app/graph.py`.

### 4.2 Read `build_graph` (it's only ~10 lines)

```
g.add_node("supervisor", supervisor)
for worker_name, agent in WORKERS.items():
    g.add_node(worker_name, agent)

g.add_edge(START, "supervisor")
for worker_name in WORKERS:
    g.add_edge(worker_name, "supervisor")
```

There are NO `add_conditional_edges` from the supervisor. The supervisor's
handoff tools' `Command(goto=...)` returns drive routing directly.

### 4.3 Follow `supervisor` into `app/supervisor.py`

```python
supervisor = create_agent(
    model=get_llm(),
    tools=HANDOFF_TOOLS,
    system_prompt=SUPERVISOR_PROMPT,
    name="supervisor",
    middleware=[inject_human_after_ai],
)
```

The supervisor is just a `create_agent`. Its tools are the 5 handoff tools
(`transfer_to_X` × 4 + `finish`). Each handoff tool:

1. Reads the supervisor's most recent AIMessage (via `_last_ai_message(state)`).
2. Builds a synthetic ToolMessage acknowledging the handoff (`tool_call_id=runtime.tool_call_id`).
3. Returns `Command(goto="X", graph=Command.PARENT, update={"messages": [...]})`.

`graph=Command.PARENT` is what makes the goto route the PARENT graph (not the
supervisor's own internal agent loop).

The AIMessage must be included in the update because `Command(graph=PARENT)`
otherwise leaves it stranded in the supervisor's subgraph — and Anthropic /
OpenAI require every tool_result to be paired with a preceding tool_use AIMessage.

### 4.4 Follow into a worker (e.g. `calibration_agent`)

`F12` on `calibration_agent` → `app/agents.py`.

```python
calibration_agent = create_agent(
    model=get_llm(),
    tools=[calibrate_surrogate],
    name="calibration_agent",
    system_prompt=CALIBRATION_PROMPT,
    state_schema=WorkflowState,
)
```

`state_schema=WorkflowState` is the canonical way to make the worker share the
parent graph's state schema — so when its tool returns `Command(update={"calibration": ...})`,
that field is written to the parent state, not just to the worker's internal
`AgentState`.

### 4.5 Back-trace your reading history

- Linux/Windows: `Alt+Left`
- macOS: `Ctrl+-`

Step back from `agents.py` → `supervisor.py` → `graph.py` → `run_workflow.py`.

---

## 5. Phase 3 — Debugger trace (the killer step, 20 min)

The static read tells you WHAT the code does. The debugger shows you HOW the
agent loops actually move messages around. This is where the mental model lands.

### 5.1 Set breakpoints

- `app/graph.py` — first line of `build_graph` (watch the graph compile)
- `app/supervisor.py` — first line of `transfer_to_market_data_agent` (see the handoff tool fire)
- `app/supervisor.py` — first line of `inject_human_after_ai` (see the middleware in action)
- `app/tools.py` — first line of `calibrate_surrogate` (see the surrogate API call)

### 5.2 Hit F5

Paused in `build_graph`. F10 through it — watch the graph register nodes + edges.
F5 to continue.

Next pause: the first handoff tool invocation (when the supervisor's LLM has
already emitted a tool call). Inspect:

- `runtime.tool_call_id` — the ID that pairs the supervisor's AIMessage tool_call
  with the ToolMessage we're about to produce
- `runtime.state["messages"]` — the current message history; you should see the
  user's HumanMessage at index 0 and an AIMessage with tool_calls right before

### 5.3 Inspect the middleware

Continue until paused in `inject_human_after_ai`. Inspect:

- `request.messages[-1]` — was this an AIMessage? If so, the middleware appends
  a HumanMessage before calling `handler(request.override(messages=…))`
- This injection is scoped to THIS LLM call — `request.override` doesn't write
  back to durable state

### 5.4 Inspect a tool firing

Continue until paused in `calibrate_surrogate`. Inspect:

- `quotes` — the list of market quote dicts the LLM passed in
- `runtime.tool_call_id` — same role as in the handoff tools
- The `Command` you're about to return — `update["calibration"]` will land
  directly in `state["calibration"]` because the worker uses `state_schema=WorkflowState`

### 5.5 Look at the Call Stack

While paused inside a tool:

```
calibrate_surrogate                       ← currently here
  → tool execution (langgraph internals)
    → calibration_agent (compiled agent)
      → parent StateGraph
        → graph.stream / main
```

Click frames to inspect each scope.

---

## 6. Phase 4 — Self-test

After completing the walkthrough, answer these without re-reading:

1. **How does the supervisor route to a worker?** (Handoff tool returns
   `Command(goto="worker_name", graph=Command.PARENT)`. The PARENT specifies that
   `goto` routes the parent graph, not the supervisor's internal subgraph.)
2. **Why is the last AIMessage included in each handoff tool's `update["messages"]`?**
   (Without it, `Command(graph=Command.PARENT)` strips the supervisor's tool_call AIMessage
   from parent state; the worker's next LLM call then sees a ToolMessage with no
   matching tool_call, and the LLM API rejects.)
3. **Why does each worker have `state_schema=WorkflowState`?** (So tools' `Command(update=…)`
   writes to PARENT state's named fields, not just the worker's internal `AgentState`.)
4. **What does the `wrap_model_call` middleware do?** (Injects a HumanMessage when
   the supervisor's request messages list ends with an AIMessage — required by
   Anthropic models that don't support assistant prefill.)
5. **Why is there no conditional edge from the supervisor?** (The supervisor's
   handoff tools' `Command(goto=…)` provides routing directly. Conditional edges
   would only be needed if routing depended on state inspection, not on a tool call.)
6. **What stops the workflow looping forever?** (The `finish` tool returns
   `Command(goto=END, graph=Command.PARENT)` when the supervisor's LLM decides
   the report agent has produced the final report.)

If you can answer 5/6, you've internalised the canonical pattern.

---

## 7. Where to go next

- **STRETCH ST1**: Validator agent — your first real conditional flow (calibration_agent
  → validator → either back to calibration OR forward to pricing)
- **Switch to `langgraph-supervisor` prebuilt**: replace the manual supervisor
  with `langgraph_supervisor.create_supervisor(...)`; compare what disappears
  (your handoff tools, the wrap_model_call middleware) and what you lose in
  flexibility
- **Wrap as FastAPI**: same shape as the sibling `api_extension/` — drop
  `graph.invoke(...)` into a FastAPI route handler

---

## TL;DR — 60-min reading sequence

| Time | Phase | What |
|---|---|---|
| 0–5 min  | 1     | Read `README.md` |
| 5–10 min | 2     | Skim file tree |
| 10–15 min | 3.2 | `state.py` |
| 15–20 min | 3.3 | `tools.py` |
| 20–25 min | 3.4 | `prompts.py` |
| 25–40 min | 4   | Trace `supervisor.py` + `agents.py` + `graph.py` |
| 40–55 min | 5   | Debugger walkthrough |
| 55–60 min | 6   | Self-test |
