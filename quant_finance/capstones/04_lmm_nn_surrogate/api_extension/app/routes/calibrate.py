"""POST /calibrate — broker quotes → calibrated LMM params + verify report.

The route is glue: schema in, service call, schema out. Bounds validation
already happened inside the pydantic models, so by the time we reach the
service we trust the inputs.
"""
from fastapi import APIRouter

from app.deps import ModelDep
from app.schemas import CalibrateRequest, CalibrateResponse, Params
from app.services import run_calibration

router = APIRouter(tags=["calibrate"])


@router.post("/calibrate", response_model=CalibrateResponse)
def calibrate_endpoint(
    req: CalibrateRequest,
    model_and_version: ModelDep,
) -> CalibrateResponse:
    """Calibrate the LMM model against `req.instruments` and `req.market_ivs`.

    PATTERN:
        model, model_version = model_and_version
        result = run_calibration(model, req.instruments, req.market_ivs)
        return CalibrateResponse(
            theta_star=Params(**result["theta_star"]),
            cost=result["cost"],
            success=result["success"],
            message=result["message"],
            model_version=model_version,
            verify=result["verify"],
        )
    """
    model, model_version = model_and_version
    result = run_calibration(model, req.instruments, req.market_ivs)
    return CalibrateResponse(
        theta_star=Params(**result["theta_star"]),
        cost=result["cost"],
        success=result["success"],
        message=result["message"],
        model_version=model_version,
        verify=result["verify"],
    )
    
