"""FastAPI dependencies.

A dependency is a small callable that FastAPI runs before the route. The
return value is injected as a parameter. Two benefits:

  1. **Testability**: in tests we can swap a dependency for a stub via
     `app.dependency_overrides[get_model_and_version] = lambda: (fake_model, 0)`
     without monkeypatching globals.
  2. **Composition**: a route declares what it needs, not how to find it.

`Annotated[X, Depends(...)]` is the modern syntax (FastAPI >= 0.95). Routes
import the type alias and use it directly:

    from app.deps import ModelDep

    def my_endpoint(model_and_version: ModelDep):
        model, version = model_and_version
        ...
"""
from typing import Annotated

import torch
from fastapi import Depends, HTTPException, Request

from app.config import Settings, get_settings


def get_model_and_version(request: Request) -> tuple[torch.nn.Module, int]:
    """Pull the model + version that startup loaded into `app.state`.

    WHY a dependency wrapper (instead of routes touching app.state directly)?
    Tests can override THIS function — no need to faff with app.state in
    fixtures.

    Raises 503 if the model wasn't loaded (lifespan crashed or was bypassed).
    That's a cleaner failure mode than `AttributeError: 'State' object has
    no attribute 'model'`.
    """
    model = getattr(request.app.state, "model", None)
    version = getattr(request.app.state, "model_version", None)
    if model is None or version is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check service logs for startup errors.",
        )
    return model, int(version)


# Type aliases — import these in routes for the Annotated[..., Depends(...)]
# pattern that FastAPI's docs recommend.
ModelDep    = Annotated[tuple[torch.nn.Module, int], Depends(get_model_and_version)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
