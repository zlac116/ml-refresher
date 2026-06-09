"""Configuration for the agents extension.

WHY: same Settings-as-single-source pattern as api_extension. Every piece
of config (API URL, model name, API key) lives here behind one Settings
object. Other modules call `get_settings()` and read fields off it.

Env-driven via pydantic-settings — same magic as the parent project:
field `surrogate_api_url` reads env var `SURROGATE_API_URL`, etc.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the agents extension.

    Examples (override at runtime):
        SURROGATE_API_URL=http://staging:8003 uv run python -m examples.run_workflow
        AGENT_MODEL=claude-sonnet-4-6 uv run python -m examples.run_workflow
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # === Agent LLM ===
    anthropic_api_key: str
    agent_model:       str = "claude-haiku-4-5-20251001"

    # === LMM surrogate API (capstone 04 api_extension) ===
    surrogate_api_url: str = "http://localhost:8003"
    api_timeout_sec:   float = 30.0

    # === Workflow ===
    max_supervisor_steps: int = 12
    """Hard cap on supervisor→worker handoffs. Stops runaway loops."""

    # === Optional: LangSmith tracing (stretch ST3) ===
    langsmith_tracing: bool   = False
    langsmith_api_key: str | None = None
    langsmith_project: str    = "lmm-agents-extension"


@lru_cache
def get_settings() -> Settings:
    """Cached so we only parse env once per process."""
    return Settings()
