"""Smoke tests — run after build to confirm the basics work.

    uv run pytest -v
"""

import pytest


def test_market_data_tool_runs():
    from src.tools.market_data import get_price_summary
    result = get_price_summary.invoke({"ticker": "AAPL"})
    assert "last_price" in result
    assert result["last_price"] > 0


def test_graph_imports():
    from src.graph import graph
    assert graph is not None


@pytest.mark.skipif(
    not __import__("os").environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_graph_runs_to_hitl():
    """End-to-end: graph should pause at the HITL interrupt before finalise."""
    from src.graph import graph
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": "smoke-test"}}
    result = graph.invoke(
        {
            "ticker": "AAPL",
            "question": "Is current valuation reasonable?",
            "messages": [HumanMessage("Is current valuation reasonable?")],
            "fundamentals_notes": "",
            "news_notes": "",
            "filings_notes": "",
            "draft_report": "",
            "approved": False,
            "next_step": None,
        },
        config=config,
    )
    assert "__interrupt__" in result or result.get("draft_report")
