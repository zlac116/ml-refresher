"""End-to-end test — drive the full graph against the LIVE surrogate API.

PREREQUISITES:
  - The surrogate API must be running on http://localhost:8003.
  - ANTHROPIC_API_KEY and OPENAI_API_KEY must be set in .env or environment.

These tests are SLOW (LLM round-trips) and BILLABLE — keep them out of any
fast/CI default suite.

NOTE: not implemented yet — see TE1 scaffolding below.
"""
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
        pytest.skip("ANTHROPIC_API_KEY not set; e2e tests require an LLM.")


# TE1 — full workflow:
#   build_graph().invoke({"messages": [HumanMessage(question)]})
#   assert final state has market_quotes, calibration, prices populated
