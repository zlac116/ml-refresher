"""End-to-end smoke test for the API.

Strategy:
  1. Smoke-train a tiny surrogate (200 rows, 20 epochs) into tmp_path.
  2. Register it + alias it @production in a fresh MLflow tracking dir.
  3. Override the app's Settings to point at that tracking dir.
  4. Use FastAPI TestClient to hit each endpoint.

One test module covers happy paths for all four endpoints + one bounds-
violation test. That's enough surface for the capstone budget. A "real"
test suite would add: model load failure, alias-doesn't-exist failure,
mismatched-length /calibrate request, etc.
"""
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import pytest
import torch
from fastapi.testclient import TestClient
from mlflow import MlflowClient

# Same shim the rest of the package uses.
PARENT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from lmm_nn_capstone import (  # noqa: E402
    N_FEATURES,
    Surrogate,
    generate_data,
    split_train_val,
    train_surrogate,
)

from app.config import Settings, get_settings  # noqa: E402

MODEL_NAME = "lmm-surrogate"


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def trained_registry(tmp_path_factory) -> Path:
    """Train a tiny surrogate and register it @production in a tmp tracking dir.

    Module-scoped because training takes a few seconds and we don't want it
    per-test.

    PATTERN:
        tracking_dir = tmp_path_factory.mktemp("mlruns")
        mlflow.set_tracking_uri(str(tracking_dir))
        mlflow.set_experiment("test")

        torch.manual_seed(0)
        X, y                   = generate_data(200, seed=0)
        X_tr, y_tr, X_va, y_va = split_train_val(X, y, 0.2, seed=0)
        model = Surrogate(N_FEATURES, (16, 16))
        train_surrogate(
            model, X_tr, y_tr, X_va, y_va,
            epochs=20, lr=1e-2, device=torch.device("cpu"),
        )

        with mlflow.start_run():
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )

        # Promote v1 to @production so the API can load it.
        MlflowClient().set_registered_model_alias(MODEL_NAME, "production", "1")
        return tracking_dir
    """
    # TODO T1 — implement per the docstring.
    raise NotImplementedError("TODO T1: trained_registry fixture")


@pytest.fixture
def client(trained_registry: Path) -> TestClient:
    """Boot the FastAPI app pointed at the tmp registry.

    PATTERN:
        # Override get_settings to point at the tmp tracking dir.
        def _settings() -> Settings:
            return Settings(
                mlflow_tracking_uri=str(trained_registry),
                model_name=MODEL_NAME,
                model_alias="production",
            )

        # Import here so the override is in place before app boots.
        from app.main import create_app
        app = create_app()
        app.dependency_overrides[get_settings] = _settings

        with TestClient(app) as c:    # triggers lifespan
            yield c
    """
    # TODO T2 — implement per the docstring.
    raise NotImplementedError("TODO T2: client fixture")


# =============================================================================
# Tests
# =============================================================================
def test_price_endpoint_returns_iv(client: TestClient) -> None:
    """POST /price with valid params returns an IV in a plausible range."""
    # TODO T3 — call /price with:
    #   params:  {"sig_a": 0.18, "sig_c": 0.40, "sabr_alpha": 0.015, "rho_inf": 0.30}
    #   instruments: [{"T": 1.0, "K": 0.030, "F": 0.035}]
    # Assert: 200, response has "ivs" (list of length 1), 0.1 < iv < 1.0.
    # Assert "model_version" == 1.
    raise NotImplementedError("TODO T3: /price happy path")


def test_calibrate_endpoint_recovers_true_params(client: TestClient) -> None:
    """POST /calibrate with synthetic market IVs recovers theta_star near true_params.

    Since the surrogate is tiny (16, 16, 20 epochs) we can't be too strict.
    Just assert the optimizer succeeded and the verify rmses come back.
    """
    # TODO T4 — call /calibrate with the parent capstone's 4 instruments
    # and market_ivs derived from true_params = (0.18, 0.40, 0.015, 0.30).
    # Compute market_ivs in the test (don't hardcode — they depend on
    # mock_lmm_iv being stable).
    # Assert: 200, "success" is True, verify.rmse_calib_bp is a float.
    raise NotImplementedError("TODO T4: /calibrate happy path")


def test_models_list_shows_production_alias(client: TestClient) -> None:
    """GET /models returns one version with the production alias."""
    # TODO T5 — GET /models.
    # Assert: 200, response["name"] == "lmm-surrogate",
    # len(versions) == 1, "production" in versions[0]["aliases"].
    raise NotImplementedError("TODO T5: GET /models")


def test_promote_endpoint_sets_alias(client: TestClient) -> None:
    """POST /promote with a new alias adds that alias to the version."""
    # TODO T6 — POST /models/lmm-surrogate/promote with body
    # {"version": 1, "alias": "staging"}.
    # Assert: 200, then GET /models and check "staging" appears in
    # versions[0]["aliases"].
    raise NotImplementedError("TODO T6: POST /promote")


def test_price_rejects_out_of_bounds_T(client: TestClient) -> None:
    """POST /price with T outside the training range returns 422."""
    # TODO T7 — call /price with T=20.0 (outside [T_LO=0.5, T_HI=10.0]).
    # Assert: 422 (pydantic validation error, NOT 500).
    # Assert: response body mentions "T" somewhere in the error detail.
    raise NotImplementedError("TODO T7: /price bounds violation")
