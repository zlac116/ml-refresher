"""Operational endpoints: liveness + readiness (mounted at root, not /api/v1).

/metrics is exposed separately by the Prometheus instrumentator in
core/observability.py (stretch).
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.session import get_session

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health():
    """Liveness: the process is up. Return 200 with a tiny payload. No DB check."""
    # TODO: return {"status": "ok"}
    return {"status": "ok"}


@router.get("/ready")
async def ready(session: AsyncSession = Depends(get_session)):
    """Readiness: can we serve traffic? Run a trivial `SELECT 1`.

    Return 200 if the DB responds; otherwise return/raise a 503.
    """
    # TODO: try a SELECT 1 via the session; map failure to 503.
    try:
        r = (await session.execute(text("SELECT 1"))).scalar()
        if r == 1:
            return {"status": "ok"}
    except Exception:
        pass
    raise HTTPException(status_code=503, detail="database unavailable")
