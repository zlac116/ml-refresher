"""GET /models, POST /models/{name}/promote — registry management.

NB: /promote updates the registry alias but does NOT re-load the model
into the running API process. That's a deliberate production-correct
choice — restarting the service is the atomic way to swap models. If you
hot-reloaded in-place you'd have a window of inconsistent state, and a
half-loaded model would silently serve garbage. Restart > hot-reload for
this case.
"""
from fastapi import APIRouter, HTTPException

from app.deps import SettingsDep
from app.registry import list_versions, set_alias
from app.schemas import ModelsListResponse, PromoteRequest, PromoteResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsListResponse)
def list_models_endpoint(settings: SettingsDep) -> ModelsListResponse:
    """Return all versions of the configured model name, newest first.

    PATTERN:
        versions = list_versions(settings.model_name)
        return ModelsListResponse(name=settings.model_name, versions=versions)
    """
    versions = list_versions(settings.model_name)
    return ModelsListResponse(name=settings.model_name, versions=versions)


@router.post("/{name}/promote", response_model=PromoteResponse)
def promote_endpoint(
    name: str,
    req: PromoteRequest,
    settings: SettingsDep,
) -> PromoteResponse:
    """Move `@req.alias` to point at version `req.version` of `name`.

    PATTERN:
        # Optional guard: only allow promotion of the configured model.
        # Skip if you want this endpoint to manage any registered model.
        if name != settings.model_name:
            raise HTTPException(404, f"Unknown model: {name}")

        try:
            set_alias(name=name, version=req.version, alias=req.alias)
        except Exception as e:
            raise HTTPException(400, f"Failed to set alias: {e}") from e

        return PromoteResponse(name=name, version=req.version, alias=req.alias)
    """
    if name != settings.model_name:
        raise HTTPException(404, f"Unknown model: {name}")

    try:
        set_alias(name=name, version=req.version, alias=req.alias)
    except Exception as e:
        raise HTTPException(400, f"Failed to set alias: {e}") from e

    return PromoteResponse(name=name, version=req.version, alias=req.alias)
