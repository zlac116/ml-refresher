"""Unit tests for the three @tool wrappers.

Each tool is invoked via `.func(...)` which bypasses the @tool decorator's
LLM-facing schema and calls the underlying Python function directly. This
lets us pass `runtime` (which would normally be injected by LangGraph) as
a plain stub object.

HTTP-calling tools (calibrate_surrogate, price_swaption) use `respx` to
intercept httpx calls — no live surrogate API needed.

Refs:
  - LangChain tools (Command, ToolRuntime): https://docs.langchain.com/oss/python/langchain/tools
  - respx (httpx mocking, community):       https://lundberg.github.io/respx/
"""
import json
from typing import Any

import httpx
import pytest
import respx
from langchain.messages import AIMessage
from langgraph.types import Command

from app import tools as tools_mod
from app.tools import calibrate_surrogate, fetch_market_quotes, price_swaption


# ============================================================================
# Helpers — fake ToolRuntime for tests that pass via `.func(...)`.
# Using a plain class (NOT dataclass) so we can mutate `state` per-test
# without cross-test contamination.
# ============================================================================
def _fake_runtime(state: dict[str, Any] | None = None, tool_call_id: str = "test-call-001"):
    """Build a fake ToolRuntime stub for `.func(...)` invocation in tests."""
    runtime = type("FakeRuntime", (), {})()
    runtime.tool_call_id = tool_call_id
    # Always include at least one AIMessage in state["messages"] — `_last_ai_message`
    # in tools.py requires one to pair with the ToolMessage in the Command update.
    default_state = {"messages": [AIMessage(content="placeholder calling tool")]}
    runtime.state = {**default_state, **(state or {})}
    return runtime


# ============================================================================
# fetch_market_quotes — reads examples/sample_market.json; no HTTP.
# ============================================================================
class TestFetchMarketQuotes:

    def test_returns_command_with_quotes_in_state(self, env_with_api_key, sample_quotes):
        cmd = fetch_market_quotes.func(num_quotes=2, runtime=_fake_runtime())

        assert isinstance(cmd, Command)
        assert cmd.update["market_quotes"] == sample_quotes
        # Should also emit messages (last AI + ToolMessage with the JSON quotes)
        assert len(cmd.update["messages"]) == 2
        assert cmd.update["messages"][1].tool_call_id == "test-call-001"
        assert json.loads(cmd.update["messages"][1].content) == sample_quotes
        # And uses graph=Command.PARENT so the update reaches parent state
        assert cmd.graph == Command.PARENT

    def test_respects_num_quotes_parameter(self, env_with_api_key):
        cmd = fetch_market_quotes.func(num_quotes=4, runtime=_fake_runtime())
        assert len(cmd.update["market_quotes"]) == 4


# ============================================================================
# calibrate_surrogate — POSTs to /calibrate; mock with respx.
# ============================================================================
_CALIBRATE_RESPONSE = {
    "theta_star":    {"sig_a": 0.18, "sig_c": 0.40, "sabr_alpha": 0.015, "rho_inf": 0.30},
    "cost":          1.5e-7,
    "success":       True,
    "message":       "ok",
    "model_version": 1,
    "verify": {
        "rmse_calib_bp":     11.4,
        "rmse_surrogate_bp": 8.0,
        "rows":              [{"instrument": "T=1", "market": 0.35, "nn": 0.351,
                              "mc": 0.352, "calib_bp": -2.0, "surrogate_bp": -1.0}],
    },
}


class TestCalibrateSurrogate:

    @respx.mock
    def test_default_path_no_x0_in_body(self, env_with_api_key, sample_quotes):
        """No retry_x0 in state → body MUST NOT include `x0`."""
        route = respx.post("http://localhost:8003/calibrate").mock(
            return_value=httpx.Response(200, json=_CALIBRATE_RESPONSE)
        )
        tools_mod._client.cache_clear()  # rebuild httpx client with the mock URL

        cmd = calibrate_surrogate.func(
            quotes=sample_quotes,
            runtime=_fake_runtime(state={"messages": [AIMessage(content="call calibrate")]}),
        )

        assert route.called
        sent_body = json.loads(route.calls.last.request.content)
        assert "instruments" in sent_body
        assert "market_ivs"  in sent_body
        assert "x0" not in sent_body                              # ← the key assertion
        assert cmd.update["calibration"]["success"] is True

    @respx.mock
    def test_uses_retry_x0_from_state(self, env_with_api_key, sample_quotes):
        """Tool-driven retry: state["retry_x0"] is forwarded to the API as `x0`."""
        route = respx.post("http://localhost:8003/calibrate").mock(
            return_value=httpx.Response(200, json=_CALIBRATE_RESPONSE)
        )
        tools_mod._client.cache_clear()

        retry_x0 = [0.20, 0.35, 0.018, 0.40]
        calibrate_surrogate.func(
            quotes=sample_quotes,
            runtime=_fake_runtime(state={
                "messages": [AIMessage(content="call calibrate")],
                "retry_x0": retry_x0,
            }),
        )

        sent_body = json.loads(route.calls.last.request.content)
        assert sent_body["x0"] == retry_x0                        # ← the key assertion

    @respx.mock
    def test_state_market_quotes_overrides_llm_quotes_arg(self, env_with_api_key, sample_quotes):
        """state["market_quotes"] takes precedence over the LLM's `quotes` arg.
        This is the drop_worst handoff: the validator mutates state.market_quotes
        and the tool MUST use the modified list, not whatever the LLM passes.
        """
        route = respx.post("http://localhost:8003/calibrate").mock(
            return_value=httpx.Response(200, json=_CALIBRATE_RESPONSE)
        )
        tools_mod._client.cache_clear()

        # State has only ONE quote (the validator dropped one)
        state_quotes = [sample_quotes[0]]
        # LLM still passes the ORIGINAL two
        calibrate_surrogate.func(
            quotes=sample_quotes,
            runtime=_fake_runtime(state={
                "messages":      [AIMessage(content="call calibrate")],
                "market_quotes": state_quotes,
            }),
        )

        sent_body = json.loads(route.calls.last.request.content)
        assert len(sent_body["instruments"]) == 1                 # state won
        assert len(sent_body["market_ivs"])  == 1


# ============================================================================
# price_swaption — POSTs to /price; mock with respx.
# ============================================================================
_PRICE_RESPONSE = {"ivs": [0.3445], "model_version": 1}


class TestPriceSwaption:

    @respx.mock
    def test_posts_correct_body_shape(self, env_with_api_key):
        route = respx.post("http://localhost:8003/price").mock(
            return_value=httpx.Response(200, json=_PRICE_RESPONSE)
        )
        tools_mod._client.cache_clear()

        params = {"sig_a": 0.18, "sig_c": 0.40, "sabr_alpha": 0.015, "rho_inf": 0.30}
        instruments = [{"T": 1.0, "K": 0.035, "F": 0.035}]
        cmd = price_swaption.func(
            params=params,
            instruments=instruments,
            runtime=_fake_runtime(state={"messages": [AIMessage(content="call price")]}),
        )

        assert route.called
        sent_body = json.loads(route.calls.last.request.content)
        assert sent_body["params"] == params
        assert sent_body["instruments"] == instruments
        assert cmd.update["prices"] == _PRICE_RESPONSE
        assert cmd.graph == Command.PARENT
