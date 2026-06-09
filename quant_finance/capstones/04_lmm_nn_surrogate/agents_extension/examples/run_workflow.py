"""CLI entrypoint — ask the agent team a question, watch the workflow run.

Usage:
    uv run python -m examples.run_workflow "calibrate to today's market and price the 5y ATM swaption"

PREREQUISITES:
    1. The LMM surrogate API must be running:
           cd ../api_extension && uv run uvicorn app.main:app --reload --port 8003
    2. .env must contain ANTHROPIC_API_KEY (copy from .env.example)
    3. `uv sync` has been run in this directory

What you'll see:
    - The supervisor's tool call (which worker it picked)
    - Each worker's response (tool calls + outputs)
    - The final report

NB: streaming is added in STRETCH ST5. This CORE version uses invoke()
which returns once the entire workflow is complete.
"""
import argparse
import sys

from langchain_core.messages import HumanMessage
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


# ============================================================================
# TODO R1 — implement the main runner.
#
# PATTERN:
#     def main() -> None:
#         args = parse_args()
#         console.print(Panel(args.question, title="user", border_style="cyan"))
#
#         from app.graph import build_graph
#         graph = build_graph()
#
#         initial_state = {
#             "messages":   [HumanMessage(content=args.question)],
#             "step_count": 0,
#         }
#         final_state = graph.invoke(initial_state)
#
#         # Print every message in order so the workflow is auditable.
#         for msg in final_state["messages"]:
#             role  = getattr(msg, "name", None) or msg.type
#             color = {
#                 "human": "cyan", "ai": "yellow", "tool": "green",
#                 "supervisor": "magenta",
#             }.get(role, "white")
#             console.print(Panel(str(msg.content)[:2000], title=role,
#                                 border_style=color))
#
#         # Final report (if the report agent populated it)
#         if final_state.get("final_report"):
#             console.print(Panel(final_state["final_report"], title="FINAL REPORT",
#                                 border_style="bold green"))
# ----------------------------------------------------------------------------
def main() -> None:
    raise NotImplementedError("TODO R1: main runner")


if __name__ == "__main__":
    main()
