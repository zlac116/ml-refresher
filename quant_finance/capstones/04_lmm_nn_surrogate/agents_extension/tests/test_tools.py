"""Unit tests for the three @tool wrappers.

Uses respx to mock httpx — no live surrogate API required. Tests verify:
  - The tool calls the right endpoint with the right body shape
  - The tool returns the parsed JSON
  - Errors propagate as expected

Tools are decorated with @tool, which wraps them in a StructuredTool.
To call from a test, use `.invoke({"arg": value})` (or `.run({...})`).
"""
from __future__ import annotations

import respx
from httpx import Response


# ============================================================================
# TODO TT1 — test fetch_market_quotes.
#
# PATTERN:
#     def test_fetch_market_quotes_reads_fixture(env_with_api_key):
#         from app.tools import fetch_market_quotes
#         quotes = fetch_market_quotes.invoke({"num_quotes": 2})
#         assert len(quotes) == 2
#         assert {"T", "K", "F", "iv"} <= quotes[0].keys()
# ----------------------------------------------------------------------------


# ============================================================================
# TODO TT2 — test calibrate_surrogate with respx mock.
#
# PATTERN:
#     @respx.mock
#     def test_calibrate_surrogate_posts_correct_shape(env_with_api_key, sample_quotes):
#         from app.tools import _client, calibrate_surrogate
#         _client.cache_clear()       # re-build client with monkey-patched env
#
#         route = respx.post("http://localhost:8003/calibrate").mock(
#             return_value=Response(200, json={
#                 "theta_star":    {"sig_a": 0.18, "sig_c": 0.40,
#                                   "sabr_alpha": 0.015, "rho_inf": 0.30},
#                 "cost":          0.001,
#                 "success":       True,
#                 "message":       "ok",
#                 "model_version": 1,
#                 "verify":        {"rmse_calib_bp": 12.3, "rmse_surrogate_bp": 8.1,
#                                   "rows": []},
#             }),
#         )
#         result = calibrate_surrogate.invoke({"quotes": sample_quotes})
#
#         assert route.called
#         sent = route.calls.last.request.read().decode()
#         assert "instruments" in sent and "market_ivs" in sent
#         assert result["success"] is True
#         assert result["verify"]["rmse_calib_bp"] == 12.3
# ----------------------------------------------------------------------------


# ============================================================================
# TODO TT3 — test price_swaption with respx mock.
#
# PATTERN: similar to TT2 — mock POST /price, assert body shape + return.
# ----------------------------------------------------------------------------
