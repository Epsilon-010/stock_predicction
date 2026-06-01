"""Pytest configuration shared by all tests.

Setup that runs once per test session:
  * Forces `APP_ENV=test` so settings load the test profile
  * Initialises Loguru with the test sink (quiet, WARNING+ only)
  * Provides reusable async-DB / Redis fixtures for integration tests
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

import pytest

# Force the test environment **before** any project module is imported.
os.environ.setdefault("APP_ENV", "test")

from src.config.settings import get_settings

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session


@pytest.fixture(scope="session", autouse=True)
def _initialise_logging() -> None:
    """Set up Loguru once per test session (quiet output)."""
    from src.config.logging_config import setup_logging

    setup_logging()


@pytest.fixture(scope="session")
def settings():
    """Project Settings singleton — refreshed for the test session."""
    get_settings.cache_clear()
    return get_settings()


# ── Database fixtures (integration tests) ─────────────────────────────────────
@pytest.fixture
def sync_session() -> Iterator[Session]:
    """Yield a sync SQLAlchemy session. Rolls back on test exit."""
    from src.utils.db import SyncSessionLocal

    session = SyncSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
async def async_session() -> AsyncIterator[AsyncSession]:
    """Yield an async SQLAlchemy session. Rolls back on test exit."""
    from src.utils.db import AsyncSessionLocal

    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.rollback()
        await session.close()


# ── Redis fixture (integration tests) ─────────────────────────────────────────
@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    """Yield an async Redis client. Flushes test keys at teardown."""
    from src.utils.redis_client import get_redis

    client = get_redis()
    test_prefix = "test:"
    try:
        yield client
    finally:
        # Best-effort cleanup of keys created by the test.
        async for key in client.scan_iter(match=f"{test_prefix}*"):
            await client.delete(key)
        await client.aclose()
