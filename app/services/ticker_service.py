"""Business logic for the ticker resource.

Services own DB access and any cross-cutting concerns (caching, validation
beyond the schema). Endpoints stay thin: they parse the request, call the
service, serialise the response. This separation lets us reuse the same
service from Prefect flows, CLI scripts, or tests without spinning up
FastAPI's request machinery.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.ticker import AssetType, Market, Ticker


class TickerNotFoundError(LookupError):
    """Raised when a ticker symbol does not exist in the catalogue."""


class TickerService:
    """All queries / mutations on the `tickers` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_symbol(self, symbol: str) -> Ticker:
        result = await self._session.execute(select(Ticker).where(Ticker.symbol == symbol.upper()))
        ticker = result.scalar_one_or_none()
        if ticker is None:
            raise TickerNotFoundError(f"Ticker '{symbol}' not found")
        return ticker

    async def list_tickers(
        self,
        page: int = 1,
        page_size: int = 50,
        market: Market | None = None,
        asset_type: AssetType | None = None,
        active_only: bool = True,
        search: str | None = None,
    ) -> tuple[list[Ticker], int]:
        """Paginated listing with optional filters. Returns (items, total)."""
        query = select(Ticker)
        if market is not None:
            query = query.where(Ticker.market == market)
        if asset_type is not None:
            query = query.where(Ticker.asset_type == asset_type)
        if active_only:
            query = query.where(Ticker.is_active.is_(True))
        if search:
            pattern = f"%{search.upper()}%"
            query = query.where(Ticker.symbol.ilike(pattern))

        # Total count (uses the same WHERE clause).
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self._session.execute(count_query)).scalar_one()

        page = max(page, 1)
        page_size = max(min(page_size, 500), 1)
        offset = (page - 1) * page_size
        rows_query = query.order_by(Ticker.symbol).offset(offset).limit(page_size)
        rows = (await self._session.execute(rows_query)).scalars().all()
        return list(rows), int(total)
