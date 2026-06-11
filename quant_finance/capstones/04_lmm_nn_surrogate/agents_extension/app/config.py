"""Configuration for the agents extension.

Settings are env-driven via pydantic-settings (reads `.env`). `get_settings()`
returns the cached Settings instance; `get_llm()` returns a cached LLM
constructed via the canonical `init_chat_model` factory — same provider-prefixed
model string for Anthropic or OpenAI ("anthropic:claude-...", "openai:gpt-...").

Refs:
  - pydantic-settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
  - init_chat_model:    https://docs.langchain.com/oss/python/langchain/models
"""
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the agents extension.

    Override at runtime:
        SURROGATE_API_URL=http://staging:8003 uv run python -m examples.run_workflow
        MODEL="anthropic:claude-haiku-4-5-20251001" uv run python -m examples.run_workflow
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # === Agent LLM ===
    # `model` is provider-prefixed per init_chat_model's canonical format.
    # We still hold both API keys so we can switch providers via `model=` alone,
    # without needing different .env files.
    anthropic_api_key: str
    openai_api_key:    str
    model:             str = "openai:gpt-5-mini"

    # === LMM surrogate API (capstone 04 api_extension) ===
    surrogate_api_url: str   = "http://localhost:8003"
    api_timeout_sec:   float = 30.0

    # === Calibration validation parameters ===
    rmse_accept_bp: float = 30.0
    retry_max: int = 3

    # === Optional: LangSmith tracing (stretch ST3) ===
    langsmith_tracing: bool       = False
    langsmith_api_key: str | None = None
    langsmith_project: str        = "lmm-agents-extension"


@lru_cache
def get_settings() -> Settings:
    """Cached Settings — `.env` parsed once per process."""
    return Settings()


@lru_cache
def get_llm() -> BaseChatModel:
    """Cached LLM constructed via the canonical init_chat_model factory.

    pydantic-settings reads .env into the Settings instance but does NOT
    push values into os.environ — so we pass api_key explicitly rather
    than relying on init_chat_model's env-var lookup.
    """
    s = get_settings()
    provider = s.model.split(":", 1)[0]
    api_key  = s.openai_api_key if provider == "openai" else s.anthropic_api_key
    return init_chat_model(
        model=s.model,
        api_key=api_key,
        temperature=0,           # deterministic for tests + reproducibility
    )