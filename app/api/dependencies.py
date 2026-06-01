"""FastAPI dependency-injection providers.

Endpoints declare what they need via `Depends(...)` and FastAPI wires it up:

    @router.get("/")
    async def list_tickers(
        service: Annotated[TickerService, Depends(get_ticker_service)],
    ): ...

This isolates endpoints from concrete construction details, which is what
Dependency Inversion is about — endpoints depend on a `TickerService`
abstraction, not on `AsyncSession` or `Redis` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.prediction_service import PredictionService
from app.services.ticker_service import TickerService
from src.config.settings import Settings, get_settings
from src.utils.db import get_async_session
from src.utils.redis_client import get_redis


# ── Settings ────────────────────────────────────────────────────────────────
def settings_dep() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dep)]


# ── Database session ────────────────────────────────────────────────────────
DBSession = Annotated[AsyncSession, Depends(get_async_session)]


# ── Redis client ────────────────────────────────────────────────────────────
# `Redis` is generic in stubs but NOT at runtime — subscripting it in a
# module-level `Annotated[...]` would raise `TypeError`. We split the alias:
# the type-checker sees `Redis[str]` (so it can verify usage); the runtime
# sees plain `Redis` (so the import doesn't blow up).
def redis_dep() -> Redis:  # type: ignore[type-arg]
    return get_redis()


if TYPE_CHECKING:
    RedisDep = Annotated[Redis[str], Depends(redis_dep)]
else:
    RedisDep = Annotated[Redis, Depends(redis_dep)]


# ── Services ────────────────────────────────────────────────────────────────
def get_ticker_service(session: DBSession) -> TickerService:
    return TickerService(session)


def get_prediction_service(redis: RedisDep, settings: SettingsDep) -> PredictionService:
    return PredictionService(redis=redis, cache_ttl_seconds=settings.redis.cache_ttl_seconds)


TickerServiceDep = Annotated[TickerService, Depends(get_ticker_service)]
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
