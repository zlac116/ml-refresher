"""FastAPI app for serving the LMM NN surrogate.

This module is intentionally thin — its only jobs are:

  1. Boot lifespan: load the @production model from MLflow into app.state.
  2. Wire routers: /calibrate, /price, /models.

All real work lives in `services.py` and `registry.py`. Routes are pure
glue between Pydantic schemas and service functions.

Run it with:
    uv run uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

import mlflow
from fastapi import FastAPI

from app.config import get_settings
from app.registry import load_model_by_alias
from app.routes import calibrate, models, price


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run on startup; reverse on shutdown.

    WHY: the model load is the single slowest startup step (materialises a
    PyTorch model from MLflow's flavour metadata + state_dict). Doing it
    once at startup means every request is fast. Doing it per request
    would be catastrophic.

    If the @production alias doesn't resolve (no registered model yet, or
    alias points at nothing), fail fast — letting MlflowException
    propagate out of the lifespan stops uvicorn from booting. That's
    better than a half-working API where /calibrate silently 500s.
    """
    # TODO 6 — wire startup.
    # WHY: this is where the "load from registry" handshake happens.
    # PATTERN:
    #     settings = get_settings()
    #     mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    #     model, version = load_model_by_alias(
    #         settings.model_name, settings.model_alias
    #     )
    #     app.state.model         = model
    #     app.state.model_version = version
    #     print(f"Loaded {settings.model_name} v{version} @{settings.model_alias}")
    yield
    # No shutdown cleanup — torch frees on process exit.


def create_app() -> FastAPI:
    """Factory so tests can build a fresh app per fixture."""
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )

    # TODO 7 — include the three routers.
    # HINT:
    #     app.include_router(calibrate.router)
    #     app.include_router(price.router)
    #     app.include_router(models.router)

    # A tiny root endpoint — useful for liveness probes / curl smoke tests.
    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "service":       settings.api_title,
            "model_version": getattr(app.state, "model_version", None),
        }

    return app


app = create_app()
