"""Shared test fixtures.

Test layers in this project:
  - test_validator.py  — pure-function tests; no LLM, no httpx, no API
  - test_tools.py      — tools mocked via respx; no live API needed
  - test_e2e.py        — full workflow vs LIVE surrogate API + real LLM

Heavy imports (mlflow, torch, surrogate) are deferred into the fixtures
that need them so the conftest module imports cleanly for the fast layers
even when api_extension / parent-capstone code isn't on sys.path.

Refs:
  - pytest fixtures:        https://docs.pytest.org/en/stable/explanation/fixtures.html
  - pytest monkeypatch:     https://docs.pytest.org/en/stable/how-to/monkeypatch.html
"""
import pytest

# Note: keep this conftest IMPORT-LIGHT. Heavy framework imports (mlflow,
# torch, the parent surrogate module) live inside fixtures, so importing
# this module does not crash tests that don't need those dependencies.


# ============================================================================
# Settings cache reset — autouse so monkeypatch.setenv calls take effect.
# ============================================================================
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


# ============================================================================
# Env fixture for tests that don't actually call an LLM.
# Settings requires BOTH api keys (anthropic + openai); test stubs cover both.
# ============================================================================
@pytest.fixture
def env_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set fake API keys + URL so Settings() validates without hitting any LLM."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
    monkeypatch.setenv("OPENAI_API_KEY",    "sk-test-fake-openai-key")
    monkeypatch.setenv("SURROGATE_API_URL", "http://localhost:8003")


# ============================================================================
# Sample swaption quotes — for tests that don't hit fetch_market_quotes.
# Matches the first two rows of examples/sample_market.json by design.
# ============================================================================
@pytest.fixture
def sample_quotes() -> list[dict]:
    return [
        {"T": 1.0, "K": 0.030, "F": 0.035, "iv": 0.3591},
        {"T": 2.0, "K": 0.040, "F": 0.040, "iv": 0.3657},
    ]


# ============================================================================
# Heavy: train a tiny surrogate + register in a tmp MLflow registry.
# Only needed by INTEGRATION tests that spin up the api_extension's FastAPI
# app via httpx.ASGITransport. Imports deferred so the conftest doesn't pay
# the torch / mlflow import cost when only unit tests run.
# ============================================================================
@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Train + register a tiny surrogate; return the tracking URI."""
    import sys
    from pathlib import Path

    # Make the parent capstone's surrogate module importable.
    # Path: agents_extension/tests/conftest.py -> capstones/04_lmm_nn_surrogate/
    parent_capstone = Path(__file__).resolve().parent.parent.parent
    if str(parent_capstone) not in sys.path:
        sys.path.insert(0, str(parent_capstone))

    import mlflow
    import torch
    from mlflow import MlflowClient

    from surrogate import (  # noqa: E402 (deferred import)
        Surrogate,
        generate_data,
        split_train_val,
        train_surrogate,
    )

    MODEL_NAME = "lmm-surrogate"

    tracking_dir = tmp_path_factory.mktemp("mlruns")
    tracking_uri = f"sqlite:///{tracking_dir}/mlflow.db"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test")

    torch.manual_seed(0)
    X, y = generate_data(200, seed=0)
    X_tr, y_tr, X_va, y_va = split_train_val(X, y, 0.2, seed=0)
    model = Surrogate(7, (16, 16))
    train_surrogate(
        model, X_tr, y_tr, X_va, y_va,
        epochs=20, lr=1e-2, device=torch.device("cpu"),
    )

    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
        )

    MlflowClient().set_registered_model_alias(MODEL_NAME, "candidate", "1")
    return tracking_uri
