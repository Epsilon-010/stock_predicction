"""Hybrid async + sync SQLAlchemy 2.0 engines for the project.

Two engines share the same `DATABASE_URL` from settings but use different
drivers and serve different workloads:

  * **async engine (asyncpg)** — used by the FastAPI request handlers, the
    async ETL extract tasks, and any code under `async def`.
  * **sync engine (psycopg2)** — used by Alembic migrations, bulk loads with
    `COPY FROM STDIN` (silver → gold ETL), CLI scripts, and notebooks.

Both engines are module-level singletons created lazily (SQLAlchemy engines
don't open a TCP connection until first use), so importing this module is cheap
even if the database is offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Final

from loguru import logger
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import get_settings

_settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Sync engine — Alembic, bulk loads, scripts
# ─────────────────────────────────────────────────────────────────────────────
sync_engine: Final[Engine] = create_engine(
    _settings.db.sync_url,
    pool_size=_settings.db.pool_size,
    max_overflow=_settings.db.max_overflow,
    pool_timeout=_settings.db.pool_timeout,
    pool_pre_ping=True,  # validate connections before checkout (handles drops)
    pool_recycle=3600,  # recycle connections older than 1h (avoids stale TCP)
    echo=_settings.db.echo,
)

SyncSessionLocal: Final[sessionmaker[Session]] = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Async engine — FastAPI, async ETL
# ─────────────────────────────────────────────────────────────────────────────
async_engine: Final[AsyncEngine] = create_async_engine(
    _settings.db.async_url,
    pool_size=_settings.db.pool_size,
    max_overflow=_settings.db.max_overflow,
    pool_timeout=_settings.db.pool_timeout,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=_settings.db.echo,
)

AsyncSessionLocal: Final[async_sessionmaker[AsyncSession]] = async_sessionmaker(
    bind=async_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # required for async — avoids implicit refresh after commit
)


# ─────────────────────────────────────────────────────────────────────────────
# Session context managers
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def get_sync_session() -> Iterator[Session]:
    """Yield a sync session — for scripts, ETL, Alembic helpers.

    Commits on success, rolls back on any exception, always closes.
    """
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_session_ctx() -> AsyncIterator[AsyncSession]:
    """Yield an async session — for async scripts and the Prefect ETL.

    The FastAPI dependency (see `app/api/dependencies.py` later) uses a thin
    wrapper around this same pattern.
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency — `Depends(get_async_session)` in route handlers.

    Yields a session bound to the request lifecycle. FastAPI calls `aclose()`
    automatically when the response is sent.
    """
    async with get_async_session_ctx() as session:
        yield session


# ─────────────────────────────────────────────────────────────────────────────
# Health checks (used by /health endpoint and Docker healthcheck)
# ─────────────────────────────────────────────────────────────────────────────
def check_sync_connection() -> bool:
    """Return True if the sync engine can reach the DB. Never raises."""
    try:
        with sync_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error(
            "Sync DB connection check failed | url={} | error={}", _settings.db.safe_url, exc
        )
        return False


async def check_async_connection() -> bool:
    """Return True if the async engine can reach the DB. Never raises."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error(
            "Async DB connection check failed | url={} | error={}", _settings.db.safe_url, exc
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle — called by FastAPI lifespan + scripts on shutdown
# ─────────────────────────────────────────────────────────────────────────────
async def dispose_engines() -> None:
    """Close all pooled connections — call on application shutdown."""
    logger.info("Disposing database engines")
    await async_engine.dispose()
    sync_engine.dispose()
