"""`/predict` endpoints — single + batch inference, served from cache when possible."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import PredictionServiceDep
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionOut,
)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get(
    "/{symbol}",
    response_model=PredictionOut,
    summary="Latest prediction for one symbol",
    responses={404: {"description": "No prediction available for that symbol"}},
)
async def predict_one(
    symbol: str,
    service: PredictionServiceDep,
    model: Annotated[str, Query(description="Registered model name")] = "xgboost",
    force_refresh: Annotated[bool, Query(description="Bypass the Redis cache")] = False,
) -> PredictionOut:
    items, _ = await service.predict([symbol], model_name=model, force_refresh=force_refresh)
    if not items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction available for symbol '{symbol}'",
        )
    return items[0]


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Latest predictions for many symbols (1-500)",
)
async def predict_batch(
    body: BatchPredictionRequest,
    service: PredictionServiceDep,
    force_refresh: Annotated[bool, Query(description="Bypass the Redis cache")] = False,
) -> BatchPredictionResponse:
    items, cache_hit = await service.predict(
        body.symbols, model_name=body.model_name, force_refresh=force_refresh
    )
    return BatchPredictionResponse(items=items, model_name=body.model_name, cached=cache_hit)
