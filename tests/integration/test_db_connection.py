"""Integration tests for the database layer.

Requires `make up && make db-migrate` (Postgres + TimescaleDB running and
migrations applied). Mark all tests in this file as `integration` so that
`make test-fast` skips them.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.db import Ticker
from src.utils.db import (
    check_async_connection,
    check_sync_connection,
    sync_engine,
)

pytestmark = pytest.mark.integration


# ── Connectivity ─────────────────────────────────────────────────────────────
def test_sync_connection_works() -> None:
    assert check_sync_connection() is True


async def test_async_connection_works() -> None:
    assert await check_async_connection() is True


# ── Postgres extensions ──────────────────────────────────────────────────────
def test_timescaledb_extension_installed() -> None:
    with sync_engine.connect() as conn:
        row = conn.execute(
            text("SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb'")
        ).first()
    assert row is not None, "TimescaleDB extension is not installed"


def test_pg_trgm_extension_installed() -> None:
    with sync_engine.connect() as conn:
        row = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")).first()
    assert row is not None, "pg_trgm extension is not installed"


def test_mlflow_database_exists() -> None:
    # mlflow lives in a sibling DB on the same Postgres instance.
    with sync_engine.connect() as conn:
        row = conn.execute(text("SELECT datname FROM pg_database WHERE datname = 'mlflow'")).first()
    assert row is not None, "MLflow database was not created on init"


# ── Schema (migrations applied) ──────────────────────────────────────────────
def test_tables_exist(sync_session: Session) -> None:
    rows = sync_session.execute(
        text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' "
            "  AND tablename IN ('tickers', 'ohlcv', 'predictions') "
            "ORDER BY tablename"
        )
    ).all()
    names = [r[0] for r in rows]
    assert names == ["ohlcv", "predictions", "tickers"]


def test_ohlcv_is_hypertable(sync_session: Session) -> None:
    row = sync_session.execute(
        text(
            "SELECT hypertable_name FROM timescaledb_information.hypertables "
            "WHERE hypertable_name = 'ohlcv'"
        )
    ).first()
    assert row is not None, "ohlcv was not converted to a hypertable"


def test_predictions_is_hypertable(sync_session: Session) -> None:
    row = sync_session.execute(
        text(
            "SELECT hypertable_name FROM timescaledb_information.hypertables "
            "WHERE hypertable_name = 'predictions'"
        )
    ).first()
    assert row is not None, "predictions was not converted to a hypertable"


# ── ORM models ───────────────────────────────────────────────────────────────
async def test_can_insert_and_query_ticker(async_session: AsyncSession) -> None:
    from src.db import AssetType, Market

    ticker = Ticker(
        symbol="TEST_7203.JP",
        name="Test Toyota",
        market=Market.JP,
        asset_type=AssetType.STOCK,
        sector_code="7000",
    )
    async_session.add(ticker)
    await async_session.flush()

    assert ticker.id is not None
    assert ticker.created_at is not None

    # Rollback at teardown by the fixture — no permanent state created.
