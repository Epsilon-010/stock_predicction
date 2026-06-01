"""Shared utilities: database engines and Redis client."""

from src.utils.db import (
    AsyncSessionLocal,
    SyncSessionLocal,
    async_engine,
    check_async_connection,
    check_sync_connection,
    dispose_engines,
    get_async_session,
    get_async_session_ctx,
    get_sync_session,
    sync_engine,
)
from src.utils.redis_client import (
    cache_get_json,
    cache_get_or_set,
    cache_set_json,
    check_redis_connection,
    close_redis_pool,
    get_redis,
    redis_client,
)

__all__ = [  # noqa: RUF022 (grouped by section, not alphabetical)
    # ── db ──
    "AsyncSessionLocal",
    "SyncSessionLocal",
    "async_engine",
    "check_async_connection",
    "check_sync_connection",
    "dispose_engines",
    "get_async_session",
    "get_async_session_ctx",
    "get_sync_session",
    "sync_engine",
    # ── redis ──
    "cache_get_json",
    "cache_get_or_set",
    "cache_set_json",
    "check_redis_connection",
    "close_redis_pool",
    "get_redis",
    "redis_client",
]
