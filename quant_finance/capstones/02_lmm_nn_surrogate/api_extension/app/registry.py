"""MLflow registry wrappers.

The goal of this module is to be the *only* place MLflow types are touched
in the inference path. Routes call these functions, which return plain
Python (or pydantic) types. If we ever swap MLflow for, say, Weights &
Biases or a custom DB, only this file changes.

The MLflow concepts in play here:

  - **Registered model**: a NAMED entry (here `lmm-surrogate`) under which
    multiple versions accumulate.
  - **Version**: each call to `mlflow.pytorch.log_model(...,
    registered_model_name=...)` creates a new version, monotonically
    numbered from 1.
  - **Alias**: a movable named pointer at a version. The "model URI"
    `models:/lmm-surrogate@production` resolves to whichever version
    currently carries the `production` alias. Promotion is just
    `set_registered_model_alias(...)`.

This is the modern MLflow API (>= 2.9). The older Stage API
(Staging/Production/Archived) is DEPRECATED — do NOT use it.
"""
from datetime import datetime, timezone
from typing import Any

import mlflow.pytorch
from mlflow import MlflowClient  # modern re-export (since MLflow 2.0)

from app.schemas import ModelVersion


def load_model_by_alias(name: str, alias: str) -> tuple[Any, int]:
    """Load the model that currently carries `@alias`. Return (model, version).

    WHY: this is the API's "what should I serve right now?" call. We resolve
    the alias inside this function so the rest of the code never has to know
    which numeric version it's actually using — that's purely metadata to
    surface in responses.

    Returns:
        model:    the loaded PyTorch model (already in eval() mode).
        version:  the resolved integer version number, for /metadata.

    Raises:
        mlflow.exceptions.MlflowException if the alias does not resolve.
        That's the right failure mode — the API should NOT boot in a
        degraded state with no model.

    PATTERN:
        uri = f"models:/{name}@{alias}"
        model = mlflow.pytorch.load_model(uri)
        model.eval()

        client  = MlflowClient()
        version = int(client.get_model_version_by_alias(name, alias).version)

        return model, version
    """
    uri = f"models:/{name}@{alias}"
    model = mlflow.pytorch.load_model(uri)
    model.eval()
    
    client = MlflowClient()
    version = int(client.get_model_version_by_alias(name, alias).version)
    return model, version


def list_versions(name: str) -> list[ModelVersion]:
    """Return all versions of `name`, each with its aliases + run_id + created_at.

    WHY: backs GET /models. Lets a caller (UI, admin script) see what's
    available and what's currently aliased where.

    NB: MlflowClient.search_model_versions returns an iterator of MLflow's
    ModelVersion type (different from OUR ModelVersion in schemas.py). We
    translate to the pydantic type here so the route can just `return
    result` without seeing any MLflow imports.

    PATTERN:
        client = MlflowClient()
        out: list[ModelVersion] = []
        for mv in client.search_model_versions(f"name='{name}'"):
            out.append(
                ModelVersion(
                    version=int(mv.version),
                    aliases=list(mv.aliases or []),
                    created_at=datetime.fromtimestamp(
                        mv.creation_timestamp / 1000, tz=timezone.utc
                    ),
                    run_id=mv.run_id,
                )
            )
        # Newest first
        out.sort(key=lambda v: v.version, reverse=True)
        return out
    """
    client = MlflowClient()

    rm = client.get_registered_model(name)
    aliases_by_version: dict[int, list[str]] = {}
    for alias, version_str in rm.aliases.items():
        aliases_by_version.setdefault(int(version_str), []).append(alias)
    
    out: list[ModelVersion] = []
    for mv in client.search_model_versions(f"name='{name}'"):
        out.append(
            ModelVersion(
                version=int(mv.version), 
                aliases=aliases_by_version.get(int(mv.version), []),
                created_at=datetime.fromtimestamp(
                    mv.creation_timestamp / 1000, tz=timezone.utc
                ),
                run_id=mv.run_id
            )
        )
    # Newest first
    out.sort(key=lambda v: v.version, reverse=True)
    return out


def set_alias(name: str, version: int, alias: str) -> None:
    """Move `@alias` to point at `version`. Idempotent.

    WHY: backs POST /models/{name}/promote. Atomic in MLflow's metadata
    store — concurrent calls are safe.

    NB: MLflow types `version` as `str` in the public API, even though
    it's a numeric monotonically-increasing field. Always cast at the
    boundary.

    PATTERN:
        client = MlflowClient()
        client.set_registered_model_alias(
            name=name, alias=alias, version=str(version)
        )
    """
    client = MlflowClient()
    client.set_registered_model_alias(
        name=name, alias=alias, version=str(version)
    )
