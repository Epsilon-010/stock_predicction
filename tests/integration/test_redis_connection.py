"""Integration tests for the Redis cache layer.

Requires `make up` (Redis running). Marked `integration` so `make test-fast`
skips them.
"""

from __future__ import annotations

import pytest
from redis.asyncio import Redis

from src.utils.redis_client import (
    cache_get_json,
    cache_get_or_set,
    cache_set_json,
    check_redis_connection,
)

pytestmark = pytest.mark.integration


async def test_redis_connection_works() -> None:
    assert await check_redis_connection() is True


async def test_set_and_get_string(redis: Redis) -> None:
    await redis.set("test:string", "hello", ex=10)
    value = await redis.get("test:string")
    assert value == "hello"


async def test_json_roundtrip() -> None:
    payload = {"ticker": "7203.JP", "close": 2987.0, "volume": 16498800}
    await cache_set_json("test:bar:7203.JP", payload, ttl_seconds=10)
    fetched = await cache_get_json("test:bar:7203.JP")
    assert fetched == payload


async def test_cache_miss_returns_none() -> None:
    assert await cache_get_json("test:does:not:exist") is None


async def test_cache_get_or_set_runs_loader_once() -> None:
    calls = {"n": 0}

    async def loader() -> dict[str, int]:
        calls["n"] += 1
        return {"value": 42}

    first = await cache_get_or_set("test:lazy", loader, ttl_seconds=10)
    second = await cache_get_or_set("test:lazy", loader, ttl_seconds=10)

    assert first == {"value": 42}
    assert second == {"value": 42}
    assert calls["n"] == 1, "loader should only run once — second call should hit cache"
