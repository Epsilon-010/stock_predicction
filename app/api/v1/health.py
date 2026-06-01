"""`/health` endpoint — used by Docker, k8s and the load balancer.

Returns 200 if the API is up at all; the body breaks down per-component
status (DB, Redis) so on-call can see which dependency is degraded.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import APIRouter

from app.api.dependencies import SettingsDep
from app.schemas.health import ComponentHealth, HealthResponse
from src.utils.db import check_async_connection
from src.utils.redis_client import check_redis_connection

router = APIRouter(tags=["health"])

HealthCheck = Callable[[], Awaitable[bool]]


async def _check_component(name: str, coro_fn: HealthCheck) -> ComponentHealth:
    start = time.perf_counter()
    try:
        ok = await coro_fn()
    except Exception as exc:
        return ComponentHealth(
            name=name,
            status="down",
            latency_ms=(time.perf_counter() - start) * 1000,
            detail=str(exc)[:200],
        )
    return ComponentHealth(
        name=name,
        status="ok" if ok else "down",
        latency_ms=(time.perf_counter() - start) * 1000,
    )


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    components = [
        await _check_component("postgres", check_async_connection),
        await _check_component("redis", check_redis_connection),
    ]
    overall: Literal["ok", "degraded"] = (
        "ok" if all(c.status == "ok" for c in components) else "degraded"
    )
    return HealthResponse(
        status=overall,
        version=settings.app.version,
        environment=settings.app.env.value,
        components=components,
    )
