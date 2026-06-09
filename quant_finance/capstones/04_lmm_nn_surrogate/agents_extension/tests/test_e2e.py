"""End-to-end test — drive the full graph against the LIVE surrogate API.

PREREQUISITES:
  - The surrogate API must be running on http://localhost:8003.
    Start it in another terminal:
        cd ../api_extension && uv run uvicorn app.main:app --reload --port 8003
  - ANTHROPIC_API_KEY must be set in .env or the environment.
  - A model must be registered + aliased @candidate (see api_extension README).

These tests are SLOW (a few seconds + LLM round-trips) and BILLABLE
(they call Claude). Mark them so they don't run on `pytest -m fast`.

If you want to skip e2e by default, use:
    @pytest.mark.e2e
    def test_workflow_produces_calibration_and_prices(): ...

and configure pyproject.toml:
    addopts = "-m 'not e2e'"
"""
from __future__ import annotations

import os

import httpx
import pytest


def _api_alive(url: str) -> bool:
    try:
        r = httpx.get(f"{url}/", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def _require_live_api():
    """Skip the whole module if the surrogate API isn't reachable."""
    url = os.environ.get("SURROGATE_API_URL", "http://localhost:8003")
    if not _api_alive(url):
        pytest.skip(f"Surrogate API not reachable at {url}; start it before running e2e tests.")


@pytest.fixture(scope="module", autouse=True)
def _require_anthropic_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; e2e tests require Claude.")


# ============================================================================
# TODO TE1 — full workflow test.
#
# PATTERN:
#     def test_workflow_produces_calibration_and_prices():
#         from app.graph import build_graph
#         from langchain_core.messages import HumanMessage
#
#         graph = build_graph()
#         result = graph.invoke({
#             "messages": [HumanMessage(content=(
#                 "Fetch 2 market quotes, calibrate the LMM surrogate, "
#                 "then price a 1y ATM swaption (T=1, K=0.035, F=0.035)."
#             ))],
#             "step_count": 0,
#         })
#
#         # The workflow should have left calibration + prices in state.
#         assert result.get("calibration") is not None, "no calibration in final state"
#         assert result["calibration"]["success"] is True
#         assert result.get("prices") is not None,      "no prices in final state"
#         assert len(result["prices"]) >= 1
#
#         # And the supervisor should have terminated cleanly (not via step cap).
#         assert result["step_count"] < 12
# ----------------------------------------------------------------------------
