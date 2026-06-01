"""Ticker resource schemas."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class TickerOut(BaseModel):
    """Public view of one ticker."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str = Field(..., examples=["7203.JP"])
    name: str | None = None
    market: str = Field(..., examples=["jp"])
    asset_type: str = Field(..., examples=["stock"])
    sector_code: str | None = None
    listed_at: date | None = None
    delisted_at: date | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime
