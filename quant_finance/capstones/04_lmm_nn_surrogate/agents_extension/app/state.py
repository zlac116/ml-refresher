"""LangGraph state schema.

Canonical multi-agent supervisor pattern: routing happens via handoff tools
that return `Command(goto=..., graph=Command.PARENT)` from inside the
supervisor agent. The graph does NOT need a `next` field in state — the
Command's `goto` drives routing directly.

State fields:
  - messages         — full conversation history. Reduced with `add_messages`
                       so each node's return APPENDS rather than REPLACES, and
                       duplicate-id messages are deduped automatically.
  - market_quotes    — set by fetch_market_quotes tool; may be MUTATED by the
                       validator's drop-worst strategy.
  - calibration      — set by calibrate_surrogate tool. On give-up, the
                       validator RESTORES this to `best_calibration` so the
                       downstream pricing agent always sees the best result.
  - prices           — set by price_swaption tool.
  - final_report     — set by report_agent's final AIMessage (read at end).

Retry-loop fields (read/written by validator_node, read by calibrate_surrogate):
  - retries          — count of retry attempts so far.
  - retry_x0         — starting point the next calibration attempt should use
                       (set by strategies "perturb" and "random_restart"; the
                       calibrate_surrogate tool reads this from runtime.state).
  - strategies_tried — strategy names already attempted ("perturb",
                       "drop_worst", "random_restart"). Used by the failure-
                       mode picker to avoid repeating a failed strategy.
  - best_calibration — the FULL calibration result (dict) of the lowest-rmse
                       attempt so far. Tracked across retries; written back
                       to `calibration` on give-up.

Refs:
  - State schema:    https://docs.langchain.com/oss/python/langgraph/use-graph-api
  - add_messages:    https://docs.langchain.com/oss/python/langgraph/graph-api#messagesstate
  - TypedDict:       https://docs.python.org/3/library/typing.html#typing.TypedDict
"""
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict, total=False):
    # Conversation + workflow outputs
    messages:      Annotated[list[AnyMessage], add_messages]
    market_quotes: list[dict] | None
    calibration:   dict | None
    prices:        list[dict] | None
    final_report:  str | None

    # Retry-loop bookkeeping
    retries:          int
    retry_x0:         list[float] | None
    strategies_tried: list[str]
    best_calibration: dict | None
