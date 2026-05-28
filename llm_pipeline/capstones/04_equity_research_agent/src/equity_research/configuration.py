"""Runtime configuration for the equity research graph.

This is `context` in LangGraph 0.3+ terminology: values that are fixed at
invocation and don't change during a run. State (in `state.py`) changes
per step; configuration does not.

The supervisor and nodes read configuration via `runtime.context.<field>`.
"""

from dataclasses import dataclass


@dataclass(kw_only=True)
class Configuration:
    """Per-invocation configuration. Set via `config={"configurable": {...}}`."""

    # Model selection — sub-agents, supervisor, finaliser
    subagent_model: str = "openai:gpt-5-nano"
    supervisor_model: str = "openai:gpt-5-mini"
    finaliser_model: str = "openai:gpt-5"
    intent_model: str = "openai:gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Tool budgets
    tavily_max_results: int = 5
    filings_top_k: int = 4

    # Feature flags
    enable_human_approval: bool = True
    """If False, skip the HITL interrupt — useful for tests / smoke runs."""

    # LangSmith project (overrides env var if set)
    langsmith_project: str = "equity-research-agent"
