"""Shared test fixtures.

Two notable choices:
  - test_tools.py uses RESPX to mock httpx (no live API needed)
  - test_e2e.py uses the LIVE API (uvicorn must be running on the configured port)

This split mirrors the api_extension test structure: unit tests are fast
+ deterministic; e2e tests are slow + realistic.
"""
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Bust the lru_cache on get_settings between tests.

    Without this, env var changes via monkeypatch.setenv take no effect
    after the first test (same trap as in api_extension/tests/test_e2e.py).
    """
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def env_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a fake Anthropic key so Settings() validates.

    Tests that don't actually call Claude can use this.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
    monkeypatch.setenv("SURROGATE_API_URL", "http://localhost:8003")


@pytest.fixture
def sample_quotes() -> list[dict]:
    """Hand-rolled quotes for tests that don't need to hit fetch_market_quotes."""
    return [
        {"T": 1.0, "K": 0.030, "F": 0.035, "iv": 0.3591},
        {"T": 2.0, "K": 0.040, "F": 0.040, "iv": 0.3657},
    ]
