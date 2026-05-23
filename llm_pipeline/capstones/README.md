# LLM-Pipeline Capstone Trilogy (1-Day Compressed Sprint)

Three quant-finance LLM capstones compressed into a **single ~8h day** (~2.5h each + 30min setup). The goal is to practice the full range of LangChain/LangGraph patterns on real finance data in one sitting — not to ship production-grade systems.

These are **briefs**, not scaffolds. The point is to build each demo end-to-end yourself.

> **If you want production-grade depth on any one of these**, see the *Stretch / production-scope* section in each project's README. Each capstone has a "what got cut" list — promote any of those back in if you decide to spend a second or third day on a project.

## The three projects (compressed)

| # | Project | Compressed deliverable | Time |
|---|---|---|---|
| **1** | [SEC filings RAG (notebook)](01_filings_rag/README.md) | Notebook that ingests 2–3 filings, retrieves with Chroma, answers 5 sample questions with chunk citations | ~2.5h |
| **2** | [FastMCP desk co-pilot](02_desk_copilot_mcp/README.md) | FastMCP server with 2 tools (1 read + 1 write), ReAct agent, **one** HITL demo | ~2.5h |
| **3** | [2-worker supervisor on transcripts](03_earnings_supervisor/README.md) | Supervisor + classifier + summariser on 3–5 transcripts; one note on cost vs single-shot | ~2.5h |

## Coverage map vs the role spec (compressed scope)

| Role requirement | Where you touch it |
|---|---|
| RAG | Capstone 1 |
| AI agents with tool use | Capstone 2 |
| Multi-step reasoning + planning | Capstone 2 (multi-tool composition) + Capstone 3 (supervisor routing) |
| MCP-style architectures, tool interop | Capstone 2 |
| Classification on real data | Capstone 3 |
| FastMCP/FastAPI | Capstone 2 (FastMCP) |
| LangChain/LangGraph deployed | All three |
| Hallucination mitigation | Capstone 1 (citations) |
| Latency / cost trade-offs | Capstone 3 (model choice mention) |
| **Production reliability + eval** | **Cut for time** — see each project's "what got cut" section |

## Day plan

| Block | Project | Activity |
|---|---|---|
| 0:00–0:30 | Setup | `.env` + `load_dotenv()`, smoke-test `init_chat_model`, set `LANGSMITH_TRACING=true`, `pip install fastmcp` |
| 0:30–3:00 | Capstone 1 | Ingestion → Chroma → retriever-as-tool → 5 sample Q&As with citations |
| 3:00–5:30 | Capstone 2 | FastMCP server (2 tools) → ReAct agent → 1 read demo + 1 HITL-write demo |
| 5:30–8:00 | Capstone 3 | Supervisor + 2 workers on 3 transcripts → cost note vs single-shot |
| 8:00 | Wrap | One paragraph per project: what worked, what surprised you, three LangSmith trace links |

## What "done" looks like (compressed)

By end of day:
- **One notebook per capstone**, each runnable end-to-end against a real input
- **LangSmith traces** for at least 3 runs per capstone (tagged `capstone-1` / `capstone-2` / `capstone-3`)
- **One paragraph of notes per project** — what worked, one surprise, one thing you'd refactor

## What you'll have covered

By the end of the day, working demos of: RAG, agents with tools, an MCP server, and a multi-agent supervisor — all on quant data, all in LangChain 1.x.

## Honest about the compression

A 1-day compressed run means you'll skip:
- Eval sets and baseline comparisons (Capstone 1's main differentiator)
- Persistent checkpointers and multi-server MCP (Capstone 2's reliability story)
- Single-shot baseline + cost engineering (Capstone 3's "when to multi-agent" lesson)
- FastAPI deployment, prompt-injection tests, latency profiling, written `ANALYSIS.md`

If you find yourself with extra hours after the wrap, **promote one item back from any project's "what got cut" list** — Capstone 1's eval set is the highest-value addition.

## Pre-requisites

- `requirements.txt` installed in `.venv`
- `OPENAI_API_KEY` (and ideally `ANTHROPIC_API_KEY`) in `.env`, loaded via `python-dotenv`
- `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true`
- `fastmcp` installed (`pip install fastmcp`) for Capstone 2
- For quant data: existing `quant_finance/capstones/01_rates_portfolio/data/` parquet files (Capstone 2 reuses them)

## Getting started

```bash
cd llm_pipeline/capstones/01_filings_rag
cat README.md          # 2.5h plan, quarter-hour milestones
```

Don't slip past the 2.5h budget on any capstone — cut features, not the whole project. The value is in having touched all three patterns, not in any one being polished.

Good luck.
