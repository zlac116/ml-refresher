# LLM Cheatsheet — LangChain + LangGraph (code-first, 2026)

Dense reference for LLM application patterns, beginner → multi-agent. Targets
**LangChain ≥ 0.3** and **LangGraph ≥ 0.2** — modern API only. Each section:
paste-able code → why → trap.

**Modern conventions worth knowing upfront** — verified against `docs.langchain.com` (the new unified docs hub, which replaced `python.langchain.com` and `langchain-ai.github.io/langgraph`):

- **`from langchain.chat_models import init_chat_model`** for model init.
- **`from langchain.messages import HumanMessage, AIMessage, SystemMessage`** — re-exported through the unified `langchain` package (was `langchain_core.messages`).
- **`from langchain.tools import tool`** — re-exported through `langchain` (was `langchain_core.tools`).
- **LCEL with `|`** for chain composition — everything is a `Runnable`.
- **`with_structured_output(Pydantic)`** for typed output (also accepts `TypedDict` / JSON Schema; can pass `include_raw=True` to also get the raw `AIMessage`).
- **`model.bind_tools([t1, t2])`** for tool calling, not the legacy `Tool(...)` class.
- **`create_react_agent`** from `langgraph.prebuilt`, not legacy `initialize_agent`.
- **`StateGraph` with TypedDict state** (preferred — dataclasses also fine; Pydantic is "less performant" per docs).
- **`Command`** for combined state-update + routing — `update`/`goto` are RETURN-only from nodes; `resume` is INPUT-only to `invoke/stream`. Don't conflate.
- **`add_edge(START, ...)` / `add_edge(..., END)`** over the older `set_entry_point()` / `set_finish_point()`.
- **`interrupt()` + checkpointer** for HITL.
- **`InMemorySaver`** (was `MemorySaver` — renamed) for dev checkpointer; persistent backends in prod.
- **`MessagesState`** built-in convenience for chat workflows.

> **Verification status (2026-06-08)**: directly fetched and verified above
> from `docs.langchain.com`. Some sub-pages (create_react_agent hooks,
> multi-agent supervisor specifics, caching imports) couldn't be reached
> by URL — patterns shown for those areas reflect the most recent
> known-stable form but **verify against current docs before shipping
> production code**. Last known-good API at:
> - <https://docs.langchain.com/oss/python/langchain/models>
> - <https://docs.langchain.com/oss/python/langgraph/graph-api>
> - <https://docs.langchain.com/oss/python/langgraph/human-in-the-loop>
> - <https://docs.langchain.com/oss/python/langgraph/streaming>

---

## 0. The general pattern — LangChain vs LangGraph

Internalise this mental model first. Everything else in this doc is a
variation on one of these two shapes.

### Two paradigms, one ecosystem

**LangChain (LCEL — pipelines)**: linear composition of `Runnables` with `|`.
Use when the work is a **fixed pipeline** — input → transform → model → parse → output.

```
input  →  [prompt]  →  [model]  →  [parser]  →  output
                                       (every box is a Runnable;
                                        `|` pipes one's output to the next)
```

**LangGraph (state machines — agents)**: nodes mutate a shared `state`; edges
route between them. Use when the work has **loops, branches, or
human-in-the-loop** — anything an agent does.

```
                ┌─────────┐
START  ──→  ──→ │ node A  │ ──→ updates state ──→ conditional edge
                └─────────┘                              ↓
                                              ┌──────┐  ┌──────┐
                                              │  B   │  │  C   │
                                              └──────┘  └──────┘
                                                ↓          ↓
                                                          END
```

### When to use which

| Need | Use |
|---|---|
| RAG (retrieve → augment → generate), no loop | **LCEL** |
| Few-shot classifier, summariser, structured extraction | **LCEL** |
| Tool-calling agent with reflection / retry / loop | **LangGraph** |
| Multi-agent with routing decisions | **LangGraph** |
| Workflow with HITL approval gates | **LangGraph** |
| Long-running conversation with branching memory | **LangGraph** |

**Rule**: linear = LCEL, conditional/iterative = LangGraph. Don't shoehorn a state machine into LCEL; don't wrap a 3-line RAG chain in a graph.

### Per-LCEL pattern (4 steps, always the same)

```python
prompt | model | parser                        # 1. COMPOSE: pipe Runnables
chain.invoke({"input": ...})                   # 2. INVOKE: sync, single shot
async for chunk in chain.astream({"input":...}): ...   # 3. STREAM: async tokens
chain.batch([...])                             # 4. BATCH: concurrent runs
```

Every Runnable supports `invoke / ainvoke / stream / astream / batch / abatch`.

### Per-LangGraph pattern (5 steps, always the same)

```python
# 1. DEFINE STATE — the shape that flows between nodes
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # built-in reducer
    next: str

# 2. BUILD NODES — pure functions: state -> state-update
def my_node(state: State) -> dict:
    return {"messages": [model.invoke(state["messages"])]}

# 3. WIRE THE GRAPH
graph = StateGraph(State)
graph.add_node("agent", my_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

# 4. COMPILE (optionally with checkpointer for HITL/memory)
app = graph.compile(checkpointer=InMemorySaver())

# 5. INVOKE / STREAM
result = app.invoke({"messages": [...]}, config={"configurable": {"thread_id": "1"}})
async for ev in app.astream_events({"messages": [...]}, config=...): ...
```

### The "everything is a Runnable" insight

In modern LangChain, **models, prompts, parsers, retrievers, tools, even
chains-of-chains** all implement the `Runnable` interface. That's why `|`
works between them — the interface guarantees the contract. Once you
internalise this, the API stops feeling magic and starts feeling regular.

LangGraph nodes are typically *functions* (not Runnables), but they can
*call* Runnables and even *compile* sub-Runnables.

### Variations covered below

- Few-shot, structured output, parallelism → §3-§5 (LCEL)
- RAG (basic, multi-query, parent-doc, reranking) → §7
- ReAct agent, multi-agent, HITL → §8-§10 (LangGraph)
- Streaming, tracing, persistence → §11-§13

---

## 1. Setup

```bash
uv add langchain langchain-core langchain-community
uv add langchain-openai langchain-anthropic langchain-google-genai   # provider integrations
uv add langgraph                                                       # state machines
uv add langchain-chroma langchain-text-splitters                       # RAG essentials
uv add langsmith                                                       # tracing
```

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=ls__...
LANGSMITH_TRACING=true          # auto-traces every Runnable invocation
LANGSMITH_PROJECT=my-project
```

**Why split packages**: `langchain-core` is the stable primitive layer.
Provider packages (`langchain-openai`, etc.) update independently of core,
so a bumped OpenAI SDK doesn't force a core upgrade.

---

## 2. Chat models — init, invoke, structured output, streaming

```python
from langchain.chat_models import init_chat_model

# Modern init — handles provider routing automatically
model = init_chat_model("openai:gpt-4o-mini",       temperature=0)
model = init_chat_model("anthropic:claude-haiku-4-5-20251001", temperature=0)
model = init_chat_model("google_genai:gemini-2.0-flash")
```

```python
# Single-shot
resp = model.invoke("Summarise quantum computing in 1 sentence.")
print(resp.content)                                  # AIMessage.content

# Multi-message
from langchain.messages import HumanMessage, SystemMessage
resp = model.invoke([
    SystemMessage("You are a terse quant analyst."),
    HumanMessage("Explain implied vol skew."),
])
```

```python
# Streaming (tokens as they arrive)
for chunk in model.stream("Write a haiku about gradient descent."):
    print(chunk.content, end="", flush=True)

# Async stream (preferred in service code)
async for chunk in model.astream("..."):
    print(chunk.content, end="", flush=True)
```

```python
# STRUCTURED OUTPUT — Pydantic in, validated Pydantic out
from pydantic import BaseModel, Field

class Sentiment(BaseModel):
    label:    str   = Field(description="positive | negative | neutral")
    score:    float = Field(ge=0, le=1)
    reason:   str

structured = model.with_structured_output(Sentiment)
out: Sentiment = structured.invoke("This product changed my life.")
print(out.label, out.score)
```

**Why `with_structured_output`**: uses the provider's **native tool-calling**
under the hood (OpenAI function-calling, Anthropic tools), returns a
validated Pydantic object. Replaces the old `PydanticOutputParser` text-parsing dance.
**Trap**: not all providers support every Pydantic feature (deeply nested unions, recursive types). Test early.

---

## 3. Prompts — templates + few-shot

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Be concise."),
    ("human",  "Question: {question}"),
])
chain = prompt | model
chain.invoke({"role": "quant analyst", "question": "What is delta?"})
```

```python
# FEW-SHOT (chat-style — preferred for modern chat models)
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "Price went up 5%",    "output": "POSITIVE"},
    {"input": "Earnings missed",     "output": "NEGATIVE"},
    {"input": "Trading sideways",    "output": "NEUTRAL"},
]
example_template = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}"),
])
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_template,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the sentiment as POSITIVE / NEGATIVE / NEUTRAL."),
    few_shot,
    ("human", "{input}"),
])
```

```python
# Example selectors — pick the K most-relevant examples at runtime
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

selector = SemanticSimilarityExampleSelector.from_examples(
    examples, OpenAIEmbeddings(), Chroma, k=2,
)
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_template,
    example_selector=selector,
    input_variables=["input"],
)
```

**Why chat-style few-shot**: modern chat models are trained on multi-turn
conversation, not raw text. Chat-style few-shots match the training distribution.
**Trap**: too many examples → token bloat and worse signal. 3-5 well-chosen examples usually beat 20 random ones — use a selector.

---

## 4. LCEL composition — Runnables + `|`

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

# Linear: prompt → model → parser
chain = prompt | model | StrOutputParser()

# Parallel branches (RunnableParallel)
combined = RunnableParallel(
    summary=summarise_prompt | model | StrOutputParser(),
    keywords=keyword_prompt  | model | StrOutputParser(),
)
result = combined.invoke({"text": "..."})
# → {"summary": "...", "keywords": "..."}

# Pass input through unchanged + add new fields
chain = (
    RunnablePassthrough.assign(
        retrieved=lambda x: retriever.invoke(x["question"]),
    )
    | prompt | model | StrOutputParser()
)

# Lambda — wrap any function as a Runnable
wrap = RunnableLambda(lambda x: x.upper())
chain = wrap | model
```

```python
# Runtime configurability
chain.with_config({"tags": ["prod"], "metadata": {"user": "alice"}})
chain.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
chain.with_fallbacks([cheap_model | parser])         # try main, fall back on error
```

**Why LCEL**: composable, async-by-default, streamable, traceable in LangSmith
automatically. Replaces the old `LLMChain` / `SequentialChain` classes (deprecated).
**Trap**: `lambda x: ...` inside a chain runs SYNC even in `ainvoke`. Wrap with `RunnableLambda` and use `async def` for true async.

---

## 5. Tool calling

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

# Decorator style — type hints + docstring become the tool schema
@tool
def get_stock_price(symbol: str) -> float:
    """Get the current stock price for a ticker symbol."""
    return prices_db[symbol]

# Pydantic args for richer validation
class SearchArgs(BaseModel):
    query: str           = Field(description="The search query")
    top_k: int           = Field(default=5, ge=1, le=20)
    filter_date: str | None = None

@tool("web_search", args_schema=SearchArgs)
def web_search(query: str, top_k: int = 5, filter_date: str | None = None) -> list[dict]:
    """Search the web. Returns a list of {title, url, snippet}."""
    ...

# Bind tools to model — model now knows it can call them
tools = [get_stock_price, web_search]
model_with_tools = model.bind_tools(tools)

# Invoke + handle tool calls
resp = model_with_tools.invoke("What's Apple's price?")
if resp.tool_calls:
    for tc in resp.tool_calls:
        print(tc["name"], tc["args"])                # {"symbol": "AAPL"}
```

**Why `@tool`**: the function's docstring becomes the LLM-visible description; type hints become the JSON schema. Zero boilerplate.
**Trap**: tool errors crash the agent if uncaught. Use `@tool(handle_tool_errors=True)` or wrap returns in try/except — return the error string to the LLM so it can recover.

---

## 6. Embeddings + vector stores

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

emb = OpenAIEmbeddings(model="text-embedding-3-small")

# Build store
docs = [...]                                         # list[Document]
store = Chroma.from_documents(docs, emb, persist_directory="./chroma_db")

# Add later
store.add_documents([Document(page_content="...", metadata={"source": "x.md"})])

# Retrieve
retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
results = retriever.invoke("query string")           # list[Document]

# Maximal-marginal-relevance (diversity)
retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})

# Filtered
retriever = store.as_retriever(search_kwargs={"k": 4, "filter": {"source": "x.md"}})
```

```python
# Document loaders + splitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = PyPDFLoader("report.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(docs)
```

**Why `RecursiveCharacterTextSplitter`**: splits on paragraph → sentence → word boundaries, preserving semantic structure. Beats fixed-character splitting.
**Trap**: chunk size + overlap depend on embedding model context + retrieval style. Default 500/80 works for most narrative text; tighter for code/structured data.

---

## 7. RAG — basic → advanced

### Basic RAG (LCEL)
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context. If unsure, say "I don't know."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag = (
    {"context":  retriever | format_docs,
     "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)
rag.invoke("What is the calibration tolerance?")
```

### RAG with citations (structured output)
```python
class Answer(BaseModel):
    answer: str
    sources: list[str] = Field(description="Document IDs used")

structured = model.with_structured_output(Answer)

def rag_with_cite(question: str) -> Answer:
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata['id']}] {d.page_content}" for d in docs)
    return structured.invoke(prompt.format(context=context, question=question))
```

### Multi-query retrieval (LLM rewrites the query 3 ways)
```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi = MultiQueryRetriever.from_llm(retriever=retriever, llm=model)
results = multi.invoke("query")                      # union of 3 retrievals, deduped
```

### Parent-document retrieval (embed children, return parents)
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

retriever = ParentDocumentRetriever(
    vectorstore=Chroma(embedding_function=emb),
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
```

**Why parent-doc**: embed small chunks for precise retrieval but return the
LARGER surrounding context to the LLM. Best of both worlds.

### Reranking (Cohere/Voyage cross-encoder)
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

base    = store.as_retriever(search_kwargs={"k": 20})    # over-fetch
reranker = CohereRerank(model="rerank-english-v3.0", top_n=4)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=base,
)
```

**Why rerank**: vector search is fast but coarse. Over-fetch (k=20) then
rerank with a cross-encoder (top_n=4) — precision and recall.

### HyDE (Hypothetical Document Embeddings)
```python
from langchain_core.prompts import ChatPromptTemplate
hyde_prompt = ChatPromptTemplate.from_template("Write a hypothetical paragraph answering: {q}")
hyde_chain  = hyde_prompt | model | StrOutputParser() | retriever
# Embed the hypothetical answer, not the question — often better recall
```

---

## 8. LangGraph basics — `StateGraph`

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, HumanMessage

# 1. STATE — TypedDict is the recommended shape (dataclass also fine;
#    Pydantic BaseModel works but is "less performant" per docs).
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # built-in reducer (appends, doesn't overwrite)
    decision: str

# 2. NODES — pure functions taking state, returning state-updates (dicts)
def classify(state: State) -> dict:
    resp = model.invoke(state["messages"])
    return {"decision": "answer" if "?" in resp.content else "ignore"}

def answer(state: State) -> dict:
    return {"messages": [model.invoke(state["messages"])]}

def ignore(state: State) -> dict:
    return {"messages": [HumanMessage("skipped")]}

# 3. ROUTING — conditional edges based on state
def route(state: State) -> str:
    return "answer_node" if state["decision"] == "answer" else "ignore_node"

# 4. WIRE
graph = StateGraph(State)
graph.add_node("classify",    classify)
graph.add_node("answer_node", answer)
graph.add_node("ignore_node", ignore)
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route, ["answer_node", "ignore_node"])
graph.add_edge("answer_node", END)
graph.add_edge("ignore_node", END)

# 5. COMPILE + INVOKE
app = graph.compile()
result = app.invoke({"messages": [HumanMessage("What is delta?")], "decision": ""})
```

**Why `add_messages`** reducer: appends to the message list instead of overwriting. Without it, every node would clobber the conversation.
**`MessagesState` shortcut** (most common state):
```python
from langgraph.graph import MessagesState

class MyState(MessagesState):           # inherits messages: Annotated[list, add_messages]
    extra_field: str
```

**Trap**: forgetting to return a dict from a node → the framework can't merge into state. Return `{}` if no update.

---

## 9. ReAct agent — `create_react_agent`

The fastest path to a working tool-using agent — use this before hand-rolling one.

```python
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

@tool
def get_price(symbol: str) -> float:
    """Get the current stock price."""
    return prices[symbol]

agent = create_react_agent(model, tools=[get_price])

# Invoke
result = agent.invoke({"messages": [HumanMessage("What's AAPL trading at?")]})
for msg in result["messages"]:
    msg.pretty_print()

# Stream tokens + tool calls
async for ev in agent.astream_events({"messages": [HumanMessage("...")]}, version="v2"):
    if ev["event"] == "on_chat_model_stream":
        print(ev["data"]["chunk"].content, end="")
```

**Why prebuilt**: implements the canonical *think → call tool → observe → think* loop with battle-tested termination logic. Don't reinvent.
**Trap**: `create_react_agent` is a LangGraph graph under the hood — you can `.stream`, `.astream_events`, attach a checkpointer, etc., same as any graph.

---

## 10. Multi-agent — supervisor + handoffs via `Command`

```python
from typing import Literal
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field

# Supervisor decides which sub-agent to route to next
class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"] = Field(description="Next agent")

def supervisor(state: MessagesState) -> Command[Literal["researcher", "writer", "__end__"]]:
    msg = model.with_structured_output(Route).invoke([
        SystemMessage("Route to researcher (gather facts), writer (compose answer), or FINISH."),
        *state["messages"],
    ])
    if msg.next == "FINISH":
        return Command(goto=END)
    return Command(goto=msg.next)

def researcher(state: MessagesState) -> Command[Literal["supervisor"]]:
    resp = research_agent.invoke(state)               # a sub-graph or ReAct agent
    return Command(
        update={"messages": [HumanMessage(content=resp["messages"][-1].content, name="researcher")]},
        goto="supervisor",                            # hand back control
    )

def writer(state: MessagesState) -> Command[Literal["supervisor"]]:
    resp = model.invoke(state["messages"])
    return Command(
        update={"messages": [resp]},
        goto="supervisor",
    )

graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer",     writer)
graph.add_edge(START, "supervisor")
app = graph.compile()
```

**Why `Command`**: combines **state update + routing** in one return value.
**Critical distinction** (per docs):
- `Command(update=..., goto=...)` is **RETURN-only** from node functions.
- `Command(resume=...)` is **INPUT-only** to `invoke()` / `stream()` (resume after `interrupt()`).
Don't try to mix them.
- `Command(goto=node, graph=Command.PARENT)` routes back to a **parent
  graph** from a sub-graph.
**Patterns**:
- **Supervisor**: one orchestrator routes to N specialists. Above.
- **Swarm**: any agent can hand off to any other. Each agent returns `Command(goto="other_agent")`.
- **Hierarchical**: a supervisor whose sub-agents are themselves sub-graphs.

**Trap**: routing loops. Always have a `FINISH` / `END` path the supervisor can choose, and consider `recursion_limit` in the compile config.

---

## 11. HITL — `interrupt()` + checkpointing

```python
from langgraph.checkpoint.memory import InMemorySaver        # was: MemorySaver (renamed)
from langgraph.types import interrupt, Command

def needs_approval(state: MessagesState) -> dict:
    decision = interrupt({                            # PAUSES execution, returns to caller
        "question": "Approve trade?",
        "trade":    state["pending_trade"],
    })
    return {"approved": decision["approved"]}

graph.add_node("approve", needs_approval)
app = graph.compile(checkpointer=InMemorySaver())       # checkpointer is REQUIRED for HITL

# First invocation — graph runs until interrupt(), then pauses
config = {"configurable": {"thread_id": "abc"}}
result = app.invoke({"messages": [...]}, config=config)
print(result["__interrupt__"])                        # → [Interrupt(value={"question": ..., "trade": ...})]

# Human reviews; later, resume with the answer
result = app.invoke(Command(resume={"approved": True}), config=config)
```

**Why this works**: the checkpointer **saves graph state** at every node boundary.
`interrupt()` halts execution; `Command(resume=...)` re-enters at the same node
with the human's decision available as `interrupt()`'s return value.

**Checkpointers** (pick by deployment):
```python
from langgraph.checkpoint.memory import InMemorySaver        # was: MemorySaver (renamed)           # in-process (dev only)
from langgraph.checkpoint.sqlite import SqliteSaver           # local persistent
from langgraph.checkpoint.postgres import PostgresSaver       # production
```

**Trap**: without a checkpointer, `interrupt()` raises. HITL == checkpoint, always.

---

## 12. Streaming + tracing

```python
# astream_events — fine-grained event stream (every token, tool call, node start/end)
async for ev in app.astream_events({"messages": [...]}, version="v2"):
    kind = ev["event"]
    if kind == "on_chat_model_stream":
        print(ev["data"]["chunk"].content, end="", flush=True)
    elif kind == "on_tool_start":
        print(f"\n[tool {ev['name']}({ev['data']['input']})]")
    elif kind == "on_chain_end" and ev["name"] == "LangGraph":
        print("\n[done]")

# astream — coarser: emits state updates per node
async for chunk in app.astream({"messages": [...]}):
    print(chunk)                                     # {"node_name": {"messages": [...]}}

# astream(stream_mode="values") — emits FULL state after each node
async for state in app.astream({"messages": [...]}, stream_mode="values"):
    print(state["messages"][-1].content)
```

```python
# LangSmith tracing — set env vars, get full trace per invocation in the UI
# LANGSMITH_TRACING=true LANGSMITH_API_KEY=... LANGSMITH_PROJECT=my-project
# (no code changes needed — tracing auto-attaches to every Runnable)
```

**Why `astream_events`** (vs `astream`): per-token granularity for chat UIs +
visibility into every tool call. `astream` is one-event-per-node.
**Trap**: `version="v2"` is required for the modern event schema. Old `v1` is deprecated.

---

## 13. Middleware + lifecycle — callbacks, caching, rate limiting, retries

Cross-cutting concerns that wrap any Runnable/graph. All are doc-recommended.

### `RunnableConfig` — settings propagation through every step

```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    "tags":            ["prod", "v2"],
    "metadata":        {"user_id": "alice", "request_id": "abc"},
    "callbacks":       [my_handler],
    "run_name":        "summarise-trade",
    "max_concurrency": 5,                            # for .batch / .abatch
    "recursion_limit": 25,                            # for LangGraph (prevent infinite routing loops)
    "configurable":    {"thread_id": "abc"},          # required for LangGraph HITL
}
chain.invoke({"input": "..."}, config=config)
```

Propagates **automatically** through every sub-Runnable + every LangGraph
node. Tags and metadata show up on each step in LangSmith.

### Callbacks (`BaseCallbackHandler`) — observe everything

```python
from langchain_core.callbacks import BaseCallbackHandler

class LoggingHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM call: {prompts[0][:80]}...")
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"\n→ {serialized['name']}({input_str})")
    def on_chain_error(self, error, **kwargs):
        log.error(f"chain failed: {error}")

chain.invoke({...}, config={"callbacks": [LoggingHandler()]})
```

**Why callbacks**: granular hooks for logging, custom metrics, audit, redaction.
Async variants: `on_llm_start_async`, `on_tool_end_async`, etc.

### Caching — `set_llm_cache` (global, deterministic skip)

```python
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache, RedisCache

set_llm_cache(InMemoryCache())                      # process-local, ephemeral
set_llm_cache(SQLiteCache(database_path="./llm_cache.db"))      # persistent across runs
set_llm_cache(RedisCache(redis_=redis.Redis()))                 # shared across replicas
```

Cache key = model name + temperature + full prompt. Identical re-invocations
return cached output instantly (no API call). **Trap**: cache is **global** —
in multi-tenant systems, prefix prompts with tenant ID so caches don't bleed
across users.

### Rate limiting

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=10,                          # match your provider's tier
    check_every_n_seconds=0.1,
    max_bucket_size=20,                              # short-term burst allowance
)
model = init_chat_model("openai:gpt-4o-mini", rate_limiter=rate_limiter)
```

`InMemoryRateLimiter` is process-local. For distributed rate limiting use
Redis directly via a custom rate limiter.

### Retries + fallbacks (cross-runnable resilience)

```python
import httpx
from openai import RateLimitError

chain = (prompt | model | parser).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
    retry_if_exception_type=(httpx.TimeoutException, RateLimitError),
)

# Fallback chain — if main fails, try the next
resilient = chain.with_fallbacks([
    cheap_model_chain,                               # 1st fallback (cheaper model)
    cached_canned_response_chain,                    # 2nd fallback (last resort)
])
```

### Lifecycle hooks — `with_listeners`

```python
chain.with_listeners(
    on_start=lambda run, cfg: logger.info(f"start {run.id} tags={run.tags}"),
    on_end=lambda run, cfg:   logger.info(f"end   {run.id} ({run.outputs})"),
    on_error=lambda run, cfg: logger.error(f"err   {run.id}: {run.error}"),
)
```

Lightweight hooks for *run-level* events (cheaper than full callbacks).

### Token counting + cost tracking

```python
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    chain.invoke({"input": "..."})
    print(cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost)  # USD
```

For other providers: usage info appears on `AIMessage.usage_metadata`
(`{"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}`).

### LangGraph hooks — pre/post-model (LangGraph ≥ 0.2.30)

```python
from langgraph.prebuilt import create_react_agent

def trim_history(state):
    # Keep only the last 10 messages — cheap context-window management
    return {"messages": state["messages"][-10:]}

def redact_pii(state):
    # Strip sensitive fields from the model output before downstream nodes
    last = state["messages"][-1]
    return {"messages": [redact(last)]}

agent = create_react_agent(
    model, tools,
    pre_model_hook=trim_history,                     # runs BEFORE each model call
    post_model_hook=redact_pii,                      # runs AFTER each model call
)
```

**Why hooks**: cleanest place for context-window trimming, PII redaction,
prompt augmentation. Run on the agent's *internal loop* without re-implementing it.

### Core utility functions worth knowing

```python
# Send — fan-out map-reduce in LangGraph (one node spawns N parallel sub-runs)
from langgraph.types import Send

def fan_out(state) -> list[Send]:
    return [Send("process_item", {"item": x}) for x in state["items"]]

graph.add_conditional_edges("collect", fan_out, ["process_item"])
# Each item runs the "process_item" node in parallel; results merge back into state

# get_buffer_string — format message lists for legacy text prompts
from langchain.messages import get_buffer_string
print(get_buffer_string([HumanMessage("hi"), AIMessage("hello")]))
# → "Human: hi\nAI: hello"

# trim_messages — built-in context-window trimming
from langchain.messages import trim_messages
trimmed = trim_messages(messages, max_tokens=4000, strategy="last", token_counter=model)

# MessagesPlaceholder — inject a message history into a prompt template
from langchain_core.prompts import MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history"),                  # variable filled at invoke time
    ("human", "{question}"),
])
chain.invoke({"history": prior_msgs, "question": "..."})
```

**Trap**: `Send` returns parallel runs that ALL update the same state — your
state reducer must handle concurrent merges (use `add_messages` or a custom
reducer that combines lists).

---

## 14. Memory + persistence

### Short-term (within a thread — automatic with checkpointer)
```python
config = {"configurable": {"thread_id": "user-42"}}
app.invoke({"messages": [HumanMessage("My name is Alice")]}, config=config)
app.invoke({"messages": [HumanMessage("What's my name?")]}, config=config)  # remembers
```
The checkpointer persists state per `thread_id`. Different `thread_id` = different conversation.

### Long-term (across threads — `Store` API)
```python
from langgraph.store.memory import InMemoryStore        # or PostgresStore in prod

store = InMemoryStore()

def remember(state: MessagesState, store) -> dict:
    user_id = state["user_id"]
    store.put(("memories", user_id), key=str(uuid4()), value={"fact": state["fact"]})
    return {}

def recall(state: MessagesState, store) -> dict:
    user_id = state["user_id"]
    items = store.search(("memories", user_id), query=state["query"], limit=5)
    return {"recalled": [it.value["fact"] for it in items]}

app = graph.compile(checkpointer=InMemorySaver(), store=store)
```

**Why split**: short-term = within one conversation (checkpointer);
long-term = across conversations (store, searchable by semantic similarity).
**Trap**: don't store secrets in long-term memory — there's no built-in encryption layer.

---

## 14. Anti-patterns

| Smell | Fix |
|---|---|
| `from langchain.llms import OpenAI` (text-completion, deprecated) | Chat models — `init_chat_model("openai:gpt-4o-mini")` |
| `LLMChain(llm=..., prompt=...)` | `prompt \| model \| StrOutputParser()` (LCEL) |
| `SequentialChain` / `SimpleSequentialChain` | LCEL pipe — `chain1 \| chain2` |
| `PydanticOutputParser` + format instructions in prompt | `model.with_structured_output(Pydantic)` — native tool calling |
| `Tool(name="x", func=lambda ...)` | `@tool` decorator — docstring + type hints become the schema |
| `initialize_agent` / `AgentExecutor` (legacy) | `create_react_agent` from `langgraph.prebuilt` |
| Custom state-machine with `if`/`elif` | `StateGraph` — gets you checkpointing, streaming, tracing for free |
| `add_conditional_edges` + separate state update | `Command(update=..., goto=...)` returns both at once |
| HITL via callbacks / threads.Event | `interrupt()` + checkpointer + `Command(resume=...)` |
| Embed-on-every-query | Build the index once; `vectorstore.add_documents` for incremental |
| Naive RAG, no overlap or rerank | RecursiveCharacterTextSplitter + over-fetch + Cohere rerank |
| `for chunk in chain.stream(...)` in async code | `async for chunk in chain.astream(...)` |
| Bare `lambda x: ...` for async transforms | `RunnableLambda(async_func)` — keeps the async chain async |
| Token-level streaming via `astream` | `astream_events(..., version="v2")` — finer granularity |
| No `thread_id` in HITL graphs | Always pass `config={"configurable": {"thread_id": "..."}}` |
| Tools that raise on bad input | `@tool(handle_tool_errors=True)` — return error text to model |
| `InMemorySaver` in production | `SqliteSaver` / `PostgresSaver` — `InMemorySaver` is in-process only |
| `version="v1"` in `astream_events` | `version="v2"` — v1 is deprecated |
| One mega-prompt for multi-step task | Split into nodes; let the graph orchestrate |
| Hardcoding `temperature=0.7` for structured tasks | `temperature=0` for tool-calling / extraction / classification |

---

## 15. Cross-references

- Worked tutorial: `langchain_langgraph.ipynb`
- Capstones: `capstones/01_filings_rag/`, `02_desk_copilot_mcp/`, `03_earnings_supervisor/`, `04_equity_research_agent/`
- API serving patterns: `../api_engineering/api_engineering_cheatsheet.md`
- MLflow tracking for LLM apps: `../toolkit/mlflow_cheatsheet.md` (use `mlflow.langchain.log_model` to register chains)
- LangChain docs: <https://python.langchain.com/docs/>
- LangGraph docs: <https://langchain-ai.github.io/langgraph/>
- LangSmith: <https://docs.smith.langchain.com/>
