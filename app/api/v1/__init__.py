"""v1 API router — single aggregation point for every v1 endpoint module."""

from fastapi import APIRouter

from app.api.v1 import health, predict, tickers

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(tickers.router)
api_router.include_router(predict.router)

__all__ = ["api_router"]
