"""System prompts for every agent in the workflow.

WHY centralised here:
  - Prompts are the most-edited code in any LLM app — keeping them in
    one file makes iteration fast (and lets you A/B test prompts later)
  - Versioning via git: prompt diffs are first-class code review
  - Reusability: tests can import the same prompts the agents use

WRITING PROMPTS — quick rules that bite if ignored:
  1. Start with ROLE ("You are a calibration specialist...")
  2. Be specific about WHEN to use which tool (the LLM can't read your mind)
  3. Be specific about WHAT to output (or you'll get prose when you wanted JSON)
  4. For supervisors: enumerate the workers + when to pick each
  5. End with the EXIT condition ("when X is done, route to FINISH")

You'll iterate on these a lot — that's expected. Treat them as code.
"""

# ============================================================================
# TODO P1 — supervisor prompt.
# WHY: the supervisor is the routing brain. It must understand the FULL
# pipeline + what each worker does + when each is appropriate.
#
# PATTERN — be explicit about workers, order, and exit:
#
#     SUPERVISOR_PROMPT = '''You are the supervisor of a swaption-desk
#     research team. The user asks calibration / pricing questions; you
#     orchestrate specialist workers to answer.
#
#     Workers (call ONE at a time via the handoff tools):
#       - market_data_agent: fetch today's market quotes (always call FIRST
#         if calibration is needed and you don't have quotes yet)
#       - calibration_agent: calibrate the surrogate to market quotes
#         (call AFTER market_data_agent has returned quotes)
#       - pricing_agent: price new instruments using calibrated params
#         (call AFTER calibration_agent has returned theta_star)
#       - report_agent: produce the final natural-language summary
#         (call LAST, when all data is gathered)
#
#     When report_agent has produced the final report, call FINISH to end.
#     Do NOT call any worker more than necessary; do NOT do their work yourself.
#     '''
# ----------------------------------------------------------------------------
SUPERVISOR_PROMPT = """TODO P1 — write the supervisor prompt per the pattern above."""


# ============================================================================
# TODO P2 — market data agent prompt.
# WHY: this worker has ONE tool (fetch_market_quotes); the prompt just
# tells it to use that tool and report back.
#
# PATTERN:
#
#     MARKET_DATA_PROMPT = '''You are a market data specialist. Your only
#     job is to call fetch_market_quotes and return the result. If the
#     supervisor asks for a specific number of quotes, pass it as num_quotes;
#     otherwise default to 4.'''
# ----------------------------------------------------------------------------
MARKET_DATA_PROMPT = """TODO P2 — write the market data agent prompt."""


# ============================================================================
# TODO P3 — calibration agent prompt.
# WHY: same shape — one tool, clear job. Include a NOTE about reading
# the verify report to spot bad calibrations (RMSE > 50 bp).
#
# PATTERN:
#
#     CALIBRATION_PROMPT = '''You are a calibration specialist. Given a list
#     of market quotes (each with T, K, F, iv), call calibrate_surrogate to
#     find the LMM parameters (theta_star) that best fit them.
#
#     After the call, INSPECT verify.rmse_calib_bp:
#       - < 20 bp:   excellent calibration; report success
#       - 20-50 bp:  acceptable; mention the residual
#       - > 50 bp:   poor; flag this in your response so the supervisor knows
#
#     Always return theta_star + the rmse so downstream agents have it.'''
# ----------------------------------------------------------------------------
CALIBRATION_PROMPT = """TODO P3 — write the calibration agent prompt."""


# ============================================================================
# TODO P4 — pricing agent prompt.
# PATTERN:
#
#     PRICING_PROMPT = '''You are a pricing specialist. Given calibrated
#     LMM parameters (theta_star) and a list of instruments to price, call
#     price_swaption. Return the IVs in input order.
#
#     If you do not have theta_star yet, ask the supervisor to run calibration
#     first — do NOT make up parameters.'''
# ----------------------------------------------------------------------------
PRICING_PROMPT = """TODO P4 — write the pricing agent prompt."""


# ============================================================================
# TODO P5 — report agent prompt.
# WHY: terminal worker. Reads state, writes a natural-language summary.
# Include explicit structure (markdown headers) so the output is uniform.
#
# PATTERN:
#
#     REPORT_PROMPT = '''You are the final-report writer. The state now
#     contains calibrated parameters (theta_star), a verify report, and
#     priced instruments. Write a concise markdown report with these sections:
#
#       ## Calibration
#       (theta_star values + rmse_calib_bp + success flag)
#
#       ## Pricing
#       (table: instrument → predicted IV)
#
#       ## Notes
#       (any caveats — e.g., high rmse, out-of-bounds inputs, etc.)
#
#     Keep total length under 250 words. Do NOT call any tools.'''
# ----------------------------------------------------------------------------
REPORT_PROMPT = """TODO P5 — write the report agent prompt."""
