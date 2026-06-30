"""Configuration for the API service.

WHY: every config value the app needs (MLflow URI, model name, bounds) is
collected here behind a single `Settings` object. Routes/services don't
read env vars or hard-code paths — they call `get_settings()` (a FastAPI
dependency) and read fields off the returned object. Two benefits:

  1. The whole app's config surface is one file, easy to review.
  2. Tests can override `get_settings` via `app.dependency_overrides`
     to point at a tmp `mlruns/`, without touching the real env.

The LMM training bounds (LMM_PARAM_LO/HI, T_LO/HI, etc.) are mirrored from
the parent capstone. They live here — not imported from the parent — so
that:

  - the app has zero import-time dependency on the parent script's
    module-level code path (which calls argparse if accidentally run),
  - the validation contract is explicit in the API source: someone reading
    `schemas.py` can find the numbers in one hop.

If you change the parent's bounds, change them here too. (A more rigorous
solution would put the bounds in a small shared `lmm_params.py` module
imported by both — out of scope for the 3h budget.)
"""
from functools import lru_cache

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Training-region bounds (mirror surrogate.py — KEEP IN SYNC)
# =============================================================================
LMM_PARAM_NAMES = ("sig_a", "sig_c", "sabr_alpha", "rho_inf")
LMM_PARAM_LO = np.array([0.10, 0.30, 0.005, 0.10])
LMM_PARAM_HI = np.array([0.25, 0.50, 0.025, 0.50])

T_LO, T_HI = 0.5, 10.0
F_LO, F_HI = 0.02, 0.05
LOG_M_LO, LOG_M_HI = -0.3, 0.3


# =============================================================================
# Settings (env-backed)
# =============================================================================
class Settings(BaseSettings):
    """Runtime configuration. Each field is overridable via env var.

    Examples:
        MLFLOW_TRACKING_URI=http://localhost:5000 uv run uvicorn app.main:app
        MODEL_ALIAS=staging uv run uvicorn app.main:app
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MLflow
    # NB: ./mlruns (file-store) is deprecated in MLflow 3.x — must be sqlite or HTTP server.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_name:          str = "lmm-surrogate"
    model_alias:         str = "candidate"     # ← was 'production'; you have @candidate, not @production

    # App
    api_title:   str = "LMM Surrogate API"
    api_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """FastAPI dependency. Cached so we only parse env once per process."""
    return Settings()
