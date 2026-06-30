"""POST /price — (params, instruments) → predicted IVs.

This is the inference fast path: one forward pass through the surrogate.
"""
from fastapi import APIRouter

from app.deps import ModelDep
from app.schemas import PriceRequest, PriceResponse
from app.services import run_pricing

router = APIRouter(tags=["price"])


@router.post("/price", response_model=PriceResponse)
def price_endpoint(
    req: PriceRequest,
    model_and_version: ModelDep,
) -> PriceResponse:
    """Predict IVs for `req.instruments` at `req.params`.

    PATTERN:
        model, model_version = model_and_version
        ivs                  = run_pricing(model, req.params, req.instruments)
        return PriceResponse(ivs=ivs, model_version=model_version)
    """
    model, model_version = model_and_version
    ivs = run_pricing(model, req.params, req.instruments)
    return PriceResponse(ivs=ivs, model_version=model_version)
