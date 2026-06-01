"""Async Redis client backed by a connection pool.

Redis is used as:
  * cache for latest prices and recent predictions (cuts DB load on the API)
  * cache for expensive computations (technical indicator pre-aggregations)
  * future message broker for Prefect / background tasks

Only an async client is exposed — the API is async and any non-async caller
can drive it with `asyncio.run(...)`. There is no need for a sync Redis client.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Final, cast

from loguru import logger
from redis.asyncio import ConnectionPool, Redis

from src.config.settings import get_settings

_settings = get_settings()


# Single shared connection pool. `decode_responses=True` makes get/set return
# `str` instead of `bytes`, which is what 99% of caller code wants.
#
# Note: `ConnectionPool` is generic over the connection *class* (TCP, UDS, …)
# rather than the response type, so we parametrise it with `Any` — the
# response decoding is what matters at the `Redis[str]` boundary downstream.
_pool: Final[ConnectionPool[Any]] = ConnectionPool.from_url(
    str(_settings.redis.url),
    decode_responses=True,
    max_connections=20,
    socket_keepalive=True,
    health_check_interval=30,
)


def get_redis() -> Redis[str]:
    """Return a Redis client bound to the shared pool.

    Cheap to call repeatedly; the pool is what holds the actual connections.
    Use this directly in async code or as a FastAPI dependency.
    """
    # `decode_responses=True` was set on the pool, so responses come back as
    # `str`. The stubs can't trace that through `connection_pool=`, hence cast.
    return cast("Redis[str]", Redis(connection_pool=_pool))


@asynccontextmanager
async def redis_client() -> AsyncIterator[Redis[str]]:
    """Context manager that returns a client and cleans up after itself."""
    client = get_redis()
    try:
        yield client
    finally:
        # `aclose` exists at runtime in redis-py 5.x but isn't in the stubs
        # for some versions; the call is intentional, hence the ignore.
        await client.aclose()  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# Health checks
# ─────────────────────────────────────────────────────────────────────────────
async def check_redis_connection() -> bool:
    """Return True if Redis responds to PING. Never raises."""
    try:
        async with redis_client() as client:
            return bool(await client.ping())
    except Exception as exc:
        logger.error("Redis connection check failed | url={} | error={}", _settings.redis.url, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Caching helpers — small wrappers over common patterns
# ─────────────────────────────────────────────────────────────────────────────
async def cache_get_json(key: str) -> Any | None:
    """GET and deserialise a JSON-encoded value. Returns None on miss."""
    async with redis_client() as client:
        raw = await client.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Cache value at {} is not valid JSON — returning raw string", key)
        return raw


async def cache_set_json(
    key: str,
    value: Any,
    ttl_seconds: int | None = None,
) -> None:
    """SET a value as JSON with an optional TTL (defaults to settings.cache_ttl)."""
    ttl = ttl_seconds if ttl_seconds is not None else _settings.redis.cache_ttl_seconds
    payload = json.dumps(value, default=str, ensure_ascii=False)
    async with redis_client() as client:
        await client.set(key, payload, ex=ttl)


async def cache_get_or_set(
    key: str,
    loader: Callable[[], Awaitable[Any]],
    ttl_seconds: int | None = None,
) -> Any:
    """Return the cached value at `key`, or call `loader()`, store, and return its result.

    Standard cache-aside pattern. `loader` must be an async callable that
    produces a JSON-serialisable value.
    """
    cached = await cache_get_json(key)
    if cached is not None:
        return cached

    fresh = await loader()
    await cache_set_json(key, fresh, ttl_seconds=ttl_seconds)
    return fresh


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle — called from FastAPI lifespan
# ─────────────────────────────────────────────────────────────────────────────
async def close_redis_pool() -> None:
    """Close the shared connection pool. Call on application shutdown."""
    logger.info("Closing Redis connection pool")
    await _pool.aclose()  # type: ignore[attr-defined]
