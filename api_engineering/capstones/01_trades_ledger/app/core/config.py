"""Application settings, loaded from environment via pydantic-settings.

Centralises all configuration (12-factor). Import the cached `get_settings()`
elsewhere; never read os.environ directly in business code.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # security
    secret_key: str
    access_token_expire_minutes: int = 30
    jwt_algorithm: str = "HS256"

    # database
    database_url: str

    # app
    env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = ""

    # rate limits
    rate_limit_default: str = "100/minute"
    rate_limit_login: str = "5/minute"

    # observability
    otel_exporter_otlp_endpoint: str = ""

    # TODO: add a property `cors_origin_list` -> list[str] that splits
    #       cors_origins on commas (consumed by CORSMiddleware in main.py).


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (also usable as a FastAPI dependency)."""
    # TODO: return Settings()  -- lru_cache makes this a process-wide singleton.
    return Settings()
