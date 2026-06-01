"""Prediction request/response schemas."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class PredictionOut(BaseModel):
    """A single forward-looking prediction for one symbol."""

    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(..., examples=["7203.JP"])
    prediction_date: date = Field(..., description="As-of date for the feature snapshot")
    horizon_days: int = Field(..., description="Trading-day window the label looks ahead")
    model_name: str = Field(..., examples=["xgboost"])
    predicted_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_class: int = Field(..., ge=0, le=1, description="1 = up, 0 = down")


class BatchPredictionRequest(BaseModel):
    """Body for `POST /predict/batch`."""

    symbols: list[str] = Field(..., min_length=1, max_length=500)
    model_name: str = Field(default="xgboost")


class BatchPredictionResponse(BaseModel):
    items: list[PredictionOut]
    model_name: str
    cached: bool = False
