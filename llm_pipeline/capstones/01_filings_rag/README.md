# Capstone 1 — SEC Filings RAG (Compressed)

**Time:** ~2.5h (block 0:30–3:00 of the 1-day sprint)
**Maps to role spec:** RAG; hallucination mitigation via citations.

## The problem (compressed)

Build a notebook that ingests **2–3 SEC 10-K filings**, indexes them in Chroma, and answers **5 sample analyst questions** with chunk-level citations. Cite or it didn't happen — the citation is what separates this from "ChatGPT with extra steps".

## Inputs / outputs

**Inputs**: 2–3 10-K filings from SEC EDGAR (suggest 3 large-cap US banks for comparability — JPM, BAC, GS).

**Outputs**: One notebook (`rag.ipynb`) that:
1. Pulls + chunks the filings (persisted to disk, not re-embedded each run)
2. Loads into Chroma with `OpenAIEmbeddings('text-embedding-3-small')`
3. Defines a retriever-as-tool
4. Asks 5 questions, prints each answer + the chunk citations it relied on

## Quarter-hour milestones (~2.5h)

| Time | Step | Deliverable |
|---|---|---|
| 0:00–0:30 | Corpus | EDGAR pull → cleaned text. Persist to `data/filings/`. |
| 0:30–1:00 | Chunk + embed | `RecursiveCharacterTextSplitter` (chunk_size 1200, overlap 150). Chroma persisted to `data/chroma/`. **Skip re-embedding on subsequent runs.** |
| 1:00–1:30 | Retriever-as-tool | `@tool` wrapping `vector_store.as_retriever(search_kwargs={'k': 4})`. Returns `page_content` only, not full Document objects. |
| 1:30–2:00 | Citation-grounded prompt | Pydantic schema `{answer: str, citations: list[ChunkCitation]}`. System prompt: *"Cite chunk IDs inline. If retrieval is insufficient, say so."* |
| 2:00–2:30 | 5 sample questions | Run + sanity-check by hand. Tag the runs `capstone-1` in LangSmith. |

## Sample questions to seed your eval (use these or vary them)

1. *"What is each company's tier 1 capital ratio as of the most recent fiscal year?"*
2. *"Which company has the highest exposure to commercial real estate?"*
3. *"Summarise the litigation risks disclosed by JPM."*
4. *"Compare the trading-book VaR methodology between BAC and GS."* (multi-doc — likely fails on simple retrieval; a real signal)
5. *"What does GS say about climate-related financial risk?"*

Question 4 is **expected to be weak** — that's the talking point about query-decomposition / multi-doc as production work.

## "Done" criteria (compressed)

- All 5 questions return an answer with at least one citation
- Re-running the notebook does not re-embed (you'd see the cost spike if it did)
- Three LangSmith trace links you can show in interview
- One paragraph in the notebook's last cell: *"What worked, what didn't, what I'd add for production"*

## What got cut (production-scope additions)

Promote any of these back if you spend a second day:

- **30-question hand-authored eval set** with deterministic hit-rate@5 + LLM-as-judge faithfulness evaluator (this is the *single* highest-value addition — it's literally what the role spec means by *meaningful improvements rather than simple prototypes*)
- **Three-column baseline comparison**: no-retrieval vs dense-only vs hybrid+rerank
- **Hybrid retrieval** (BM25 + dense fused) and a Cohere/BGE rerank
- **FastAPI streaming endpoint** with `MemorySaver` keyed by `thread_id`
- **Prompt caching** on the system prompt; latency p50/p95 profiling
- **Query decomposition** for multi-doc questions (the failure mode you'll hit on Q4)
- Full `ANALYSIS.md` with three named failure modes and three trace links

## Anti-patterns to avoid (even in the compressed version)

- **Re-embedding every run** — embeddings are deterministic; persist Chroma. (Cheatsheet §8.)
- **chunk_size = 256** — too small to embed meaningfully.
- **chunk_size = 4000** — so large that retrieval is just "send everything".
- **No citations in the answer.** Drops the entire interview value of the project.
- **Pinned to OpenAI only** — use `init_chat_model` so the provider is one string change.

## Suggested LangChain stack

| Cheatsheet section | Used for |
|---|---|
| §2 Models, structured output | `init_chat_model` + Pydantic `{answer, citations}` |
| §4 Tools | Retriever-as-tool |
| §8 RAG | Chroma + `RecursiveCharacterTextSplitter` |
| §12 Observability | `LANGSMITH_*` env vars + `tags=['capstone-1']` |

## What you'll be able to say in interview (compressed version)

> *"I built a RAG demo over 3 banking 10-Ks. It cites chunks inline so an analyst can audit any answer. I tested 5 questions including one cross-document question that the simple retriever failed — I'd address that with query decomposition. To take this to production I'd add a 30-question eval set with a faithfulness LLM-judge to prove RAG actually beats no-retrieval, then wrap in a FastAPI streaming endpoint."*

The "to take this to production I'd add..." is the part interviewers care about — it shows you know what the gap is.
