"""`/tickers` endpoints — list, search, lookup."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import TickerServiceDep
from app.schemas.common import Page, Pagination
from app.schemas.ticker import TickerOut
from app.services.ticker_service import TickerNotFoundError
from src.db.ticker import AssetType, Market

router = APIRouter(prefix="/tickers", tags=["tickers"])


@router.get(
    "",
    response_model=Page[TickerOut],
    summary="List tickers (paginated, optional filters)",
)
async def list_tickers(
    service: TickerServiceDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
    market: Market | None = None,
    asset_type: AssetType | None = None,
    active_only: bool = True,
    search: Annotated[str | None, Query(max_length=32, description="Match against symbol")] = None,
) -> Page[TickerOut]:
    items, total = await service.list_tickers(
        page=page,
        page_size=page_size,
        market=market,
        asset_type=asset_type,
        active_only=active_only,
        search=search,
    )
    return Page[TickerOut](
        items=[TickerOut.model_validate(t) for t in items],
        pagination=Pagination(total=total, page=page, page_size=page_size),
    )


@router.get(
    "/{symbol}",
    response_model=TickerOut,
    responses={404: {"description": "Ticker not found"}},
    summary="Look up a single ticker by symbol",
)
async def get_ticker(symbol: str, service: TickerServiceDep) -> TickerOut:
    try:
        ticker = await service.get_by_symbol(symbol)
    except TickerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return TickerOut.model_validate(ticker)
