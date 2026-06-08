"""Trade + portfolio routes.

Every route requires an authenticated user (get_current_user). Every single-trade
route is authorized by ownership *inside the service* (service._get_owned), so a
caller can only ever touch their own trades.
"""
from fastapi import APIRouter, Depends, Query

from app import schemas
from app.api.deps import get_current_user, get_trade_service

router = APIRouter(tags=["trades"])


@router.post("/trades", response_model=schemas.TradeRead, status_code=201)
async def open_trade(
    data: schemas.TradeCreate,
    user=Depends(get_current_user),
    service=Depends(get_trade_service),
):
    # TODO: return await service.open_trade(user.id, data)
    return await service.open_trade(user.id, data)


@router.get("/trades", response_model=list[schemas.TradeRead])
async def list_trades(
    status: schemas.TradeStatus | None = None,
    symbol: str | None = None,
    limit: int = Query(50, ge=1, le=100),   # cap page size -> OWASP API4
    offset: int = Query(0, ge=0),
    user=Depends(get_current_user),
    service=Depends(get_trade_service),
):
    # TODO: return await service.list_trades(user.id, status=status, symbol=symbol,
    #                                        limit=limit, offset=offset)
    return await service.list_trades(
        user.id,
        status=status,
        symbol=symbol,
        limit=limit,
        offset=offset
    )


@router.get("/trades/{trade_id}", response_model=schemas.TradeRead)
async def get_trade(
    trade_id: int,
    user=Depends(get_current_user),
    service=Depends(get_trade_service),
):
    # TODO: return await service.get_trade(trade_id, user.id)  # 404 if not owned
    return await service.get_trade(trade_id, user.id)  # 404 if not owned


@router.patch("/trades/{trade_id}", response_model=schemas.TradeRead)
async def update_trade(
    trade_id: int,
    data: schemas.TradeUpdate,
    user=Depends(get_current_user),
    service=Depends(get_trade_service),
):
    # TODO: return await service.update_trade(trade_id, user.id, data)
    return await service.update_trade(trade_id, user.id, data)


@router.post("/trades/{trade_id}/close", response_model=schemas.TradeRead)
async def close_trade(
    trade_id: int,
    data: schemas.TradeClose,
    user=Depends(get_current_user),
    service=Depends(get_trade_service),
):
    # TODO: return await service.close_trade(trade_id, user.id, data)
    return await service.close_trade(trade_id, user.id, data)


@router.delete("/trades/{trade_id}", status_code=204)
async def delete_trade(
    trade_id: int,
    user=Depends(get_current_user),
    service=Depends(get_trade_service),
):
    # TODO: await service.delete_trade(trade_id, user.id)
    await service.delete_trade(trade_id, user.id)


@router.get("/portfolio/summary", response_model=schemas.PortfolioSummary)
async def portfolio_summary(
    user=Depends(get_current_user),
    service=Depends(get_trade_service),
):
    # TODO: return await service.portfolio_summary(user.id)   # stretch
    return await service.portfolio_summary(user.id)   # stretch
