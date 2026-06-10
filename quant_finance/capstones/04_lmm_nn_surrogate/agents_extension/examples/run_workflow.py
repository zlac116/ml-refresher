"""CLI entrypoint — ask the agent team a question, watch the workflow run.

Usage:
    uv run python -m examples.run_workflow "calibrate to today's market and price the 5y ATM swaption"

PREREQUISITES:
    1. The LMM surrogate API must be running:
           cd ../api_extension && uv run uvicorn app.main:app --reload --port 8003
    2. .env must contain ANTHROPIC_API_KEY and OPENAI_API_KEY
    3. `uv sync` has been run in this directory

Streams the workflow via the canonical LangGraph 1.2+ pattern:
  - stream_mode=["updates", "values"], version="v2"
  - "updates" chunks print each new message as workers run
  - "values" chunks keep the running merged state for the final report

Deduplicates printed messages by `msg.id` because compiled subgraphs
(create_agent) return their full accumulated message list per chunk;
the parent state's `add_messages` reducer dedupes by id, but the stream
emits everything.

Refs:
  - Streaming: https://docs.langchain.com/oss/python/langgraph/streaming
"""
import argparse

from langchain.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel


console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the multi-agent swaption desk workflow.")
    p.add_argument(
        "question",
        nargs="?",
        default="Fetch today's market quotes, calibrate the LMM surrogate, "
                "then price a 5y ATM swaption (T=5, K=0.045, F=0.045). "
                "Give me a brief report.",
        help="Natural-language question for the agent team.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console.print(Panel(args.question, title="user", border_style="cyan"))

    from app.graph import build_graph
    graph = build_graph()

    initial_state = {"messages": [HumanMessage(content=args.question)]}

    final_state: dict = {}
    printed_ids: set[str] = set()

    for chunk in graph.stream(initial_state, stream_mode=["updates", "values"], version="v2"):
        if chunk["type"] == "updates":
            for node_name, update in chunk["data"].items():
                for msg in update.get("messages", []):
                    if msg.id in printed_ids:
                        continue
                    printed_ids.add(msg.id)
                    role  = msg.type
                    color = {"human": "cyan", "ai": "yellow", "tool": "green"}.get(role, "white")
                    console.print(Panel(
                        str(msg.content)[:2000],
                        title=f"{node_name} ({role})",
                        border_style=color,
                    ))
        elif chunk["type"] == "values":
            final_state = chunk["data"]

    if final_state.get("final_report"):
        console.print(Panel(
            final_state["final_report"],
            title="FINAL REPORT",
            border_style="bold green",
        ))


if __name__ == "__main__":
    main()
