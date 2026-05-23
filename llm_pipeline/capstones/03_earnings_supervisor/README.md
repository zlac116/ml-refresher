# Capstone 3 — 2-Worker Supervisor on Earnings Transcripts (Compressed)

**Time:** ~2.5h (block 5:30–8:00 of the 1-day sprint)
**Maps to role spec:** Multi-agent orchestration; classification on real data.

## The problem (compressed)

Build a **supervisor + 2 workers** (classifier + summariser; drop the third worker for time) over **3–5 earnings-call transcripts**. Demonstrate the routing pattern + structured output + shared state. Note (don't fully measure) the cost vs single-shot trade-off.

This is the smallest version that still demonstrates *why supervisor patterns exist*. The full eval-driven comparison is the cut item — name it as the production gap.

## Inputs / outputs

**Inputs**: 3–5 earnings-call transcripts. Quickest source: HuggingFace (`datasets.load_dataset('jlh-ibm/earnings_call')` or similar) — load 3–5 rows, treat raw text as the input. Avoid scraping for time.

**Outputs**:
- `graph.ipynb` — supervisor + 2 workers running on the transcripts
- A printed table with columns: `{ticker, guidance, sentiment, summary}` for each transcript
- One paragraph noting the cost vs a single-shot equivalent prompt

## Architecture (compressed)

```
            ┌──────────────┐
            │  Supervisor  │   chooses next worker or 'finish'
            └──────┬───────┘
        ┌──────────┴──────────┐
        ↓                     ↓
  ┌──────────┐         ┌────────────┐
  │Classifier│         │ Summariser │
  │ gpt-4o-  │         │ gpt-4o-mini│
  │ mini     │         │            │
  └──────────┘         └────────────┘
        └──────────┬──────────┘
                   ↓
              SharedState
```

- **Classifier**: structured output `{guidance: Literal['beat','miss','inline'], sentiment: Literal['positive','neutral','negative']}`
- **Summariser**: 150-word summary with one direct quote
- **Supervisor**: routes to whichever worker hasn't run yet; returns `'finish'` when both fields populated

## Quarter-hour milestones (~2.5h)

| Time | Step | Deliverable |
|---|---|---|
| 0:00–0:20 | Dataset | Load 3–5 transcripts. Trim to ~6k tokens each so calls stay cheap. |
| 0:20–0:45 | Worker schemas + prompts | Two Pydantic schemas, two worker functions. Test each in isolation on 1 transcript. |
| 0:45–1:15 | Supervisor + StateGraph | `Annotated[list, add_messages]` for messages. Router node returns `'classifier'` / `'summariser'` / `'finish'`. `add_conditional_edges`. (Cheatsheet §6, §9.) |
| 1:15–1:45 | Run on all transcripts | Loop the graph over each. Build the result table. Tag runs `capstone-3`. |
| 1:45–2:15 | Single-shot comparison (qualitative) | One prompt that asks for both fields together. Run on the same transcripts. **Don't formally evaluate** — just eyeball one example side-by-side and note: which output looked better, which cost more. |
| 2:15–2:30 | Notes | One paragraph: which architecture would you ship for this task and why. |

## "Done" criteria (compressed)

- Supervisor + 2 workers run successfully on all 3–5 transcripts
- Result table is printed with both fields populated
- Three LangSmith traces tagged `capstone-3`
- One paragraph in the last cell: *"Multi-agent vs single-shot — what I'd ship and why"* (the answer is most likely "single-shot for this task" — that's the lesson)

## What got cut (production-scope additions)

Promote any of these back if you spend a second day:

- **Third worker (risk-flag scanner)** with `{risk_type, evidence_quote, severity}` structured output
- **Hand-labelled eval set of 30 calls** with classification F1 and routing accuracy
- **Quantitative cost comparison** — token counts, $/call, latency-per-call for both architectures
- **Prompt caching** on the long worker system prompts (~60% input-token reduction in practice)
- **Model fallback** — summariser drops to `gpt-4o` only when transcript >8k tokens
- **Faithfulness LLM-judge** using a stronger model — were the quoted lines actually in the transcript?
- Full `ANALYSIS.md` answering *"would I ship single-shot or multi-agent for this task and why"*

## Anti-patterns to avoid (even in the compressed version)

- **Two workers that just call the LLM with slightly different prompts.** If the workers don't have meaningfully different schemas + prompts, collapse them — multi-agent overhead is wasted.
- **No single-shot reference.** Even a qualitative side-by-side is worth more than nothing — it's the talking point.
- **Same model everywhere with no thought** — at minimum, *say* in your notes that both workers are on the cheap model intentionally for cost.
- **Routing logic baked into worker functions** — the supervisor decides what runs next.
- **Cycling forever** — add a hard step counter; supervisor returns `'finish'` once both fields are set.
- **Eval set written by the same model used in the workers** (relevant if you do promote the eval back from "what got cut") — circular.

## Suggested stack

| Cheatsheet section | Used for |
|---|---|
| §2 Models, structured output | `init_chat_model` + `with_structured_output(Pydantic)` per worker |
| §6 LangGraph | `StateGraph`, `Annotated[..., add_messages]`, conditional edges |
| §9 Multi-agent | Supervisor + worker pattern, shared state |
| §12 Observability | LangSmith tags `capstone-3-multi-agent` vs `capstone-3-single-shot` |

## What you'll be able to say in interview

> *"I built a supervisor over two workers — classifier with structured guidance/sentiment output, summariser with a direct-quote requirement. I qualitatively compared to a single-shot prompt asking for both fields together. For this task the single-shot was within a few tokens of the multi-agent on quality and ~half the cost — multi-agent didn't earn its overhead. To make that judgement quantitative I'd hand-label 30 calls and run F1 + cost on both, with prompt caching on the worker system prompts."*

That last sentence — *"I'd make the judgement quantitative by..."* — is the part interviewers are listening for. It shows you know the difference between *I built one* and *I proved it was the right one*.
