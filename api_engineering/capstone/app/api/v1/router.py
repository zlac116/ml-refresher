"""Aggregate the v1 routers under one APIRouter (mounted at /api/v1 in main.py)."""
from fastapi import APIRouter

from app.api.v1 import auth, trades

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(trades.router)
