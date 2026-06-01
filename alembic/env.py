"""Alembic environment — runs migrations against the project's Postgres DB.

Key differences from the stock Alembic template:
  * The DB URL is pulled from `src/config/settings.py`, not from `alembic.ini`,
    so we have a single source of truth across dev/staging/prod.
  * The sync engine from `src/utils/db.py` is reused — same pool config as the
    rest of the project, no double connection setup.
  * `target_metadata` comes from `src/db.Base` after importing all models, so
    `alembic revision --autogenerate` sees every table.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool

# Importing the package side-effects all models into Base.metadata.
from src.config.settings import get_settings
from src.db import Base  # noqa: F401 — needed so models register on Base.metadata
from src.utils.db import sync_engine

# ── Alembic Config object ────────────────────────────────────────────────────
config = context.config

# Inject the URL from settings so alembic.ini stays env-agnostic.
config.set_main_option("sqlalchemy.url", get_settings().db.sync_url)

# Configure Python logging from the alembic.ini section.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _include_object(object, name, type_, reflected, compare_to):  # noqa: ANN001
    """Skip TimescaleDB-internal chunk tables when diffing.

    After `SELECT create_hypertable('ohlcv', 'date')`, TimescaleDB creates
    chunk tables under the `_timescaledb_internal` schema. We don't manage
    those with Alembic — they're an implementation detail of the hypertable.
    """
    if type_ == "table" and getattr(object, "schema", None) == "_timescaledb_internal":
        return False
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to the DB ("alembic upgrade head --sql")."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against the live database."""
    # Reuse the project's sync engine; NullPool avoids holding extra
    # connections open between Alembic invocations.
    connectable = sync_engine.execution_options(poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=_include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
