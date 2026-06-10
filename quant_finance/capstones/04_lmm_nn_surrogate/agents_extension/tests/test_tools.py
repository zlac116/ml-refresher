"""Unit tests for the three @tool wrappers.

Uses respx to mock httpx — no live surrogate API required. Tests verify:
  - The tool calls the right endpoint with the right body shape
  - The tool returns a Command with the right state update + ToolMessage
  - Errors propagate as expected

Since tools are decorated with `@tool`, invoke them via `.invoke({"arg": ...})`.
For tools that take a `ToolRuntime`, supply a stub via the invoke `config` kwarg.

NOTE: not implemented yet — see TT1/TT2/TT3 scaffolding below.
"""
# TT1 — fetch_market_quotes reads examples/sample_market.json
# TT2 — calibrate_surrogate POSTs to /calibrate, writes state["calibration"]
# TT3 — price_swaption POSTs to /price, writes state["prices"]
