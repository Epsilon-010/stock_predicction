"""Initial schema: tickers, ohlcv hypertable, predictions hypertable.

Revision ID: 0001
Revises:
Create Date: 2026-05-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade
# ─────────────────────────────────────────────────────────────────────────────
def upgrade() -> None:
    # ── Enum types ──────────────────────────────────────────────────────────
    market_enum = sa.Enum("jp", "us", name="market_enum")
    asset_type_enum = sa.Enum(
        "stock", "etf", "index", "future", "option", "bond",
        name="asset_type_enum",
    )

    # ── tickers ─────────────────────────────────────────────────────────────
    op.create_table(
        "tickers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column("market", market_enum, nullable=False),
        sa.Column("asset_type", asset_type_enum, nullable=False),
        sa.Column("sector_code", sa.String(length=16), nullable=True),
        sa.Column("listed_at", sa.Date(), nullable=True),
        sa.Column("delisted_at", sa.Date(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_tickers"),
        sa.UniqueConstraint("symbol", name="uq_tickers_symbol"),
    )
    op.create_index("ix_tickers_symbol", "tickers", ["symbol"])
    op.create_index("ix_tickers_is_active", "tickers", ["is_active"])
    op.create_index("ix_tickers_market_asset_type", "tickers", ["market", "asset_type"])
    op.create_index("ix_tickers_market_active", "tickers", ["market", "is_active"])

    # ── ohlcv (will become hypertable) ──────────────────────────────────────
    op.create_table(
        "ohlcv",
        sa.Column("ticker_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("open", sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column("high", sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column("low", sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column("close", sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column("volume", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("open_interest", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["ticker_id"], ["tickers.id"],
            name="fk_ohlcv_ticker_id_tickers",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("ticker_id", "date", name="pk_ohlcv"),
    )
    op.create_index("ix_ohlcv_ticker_date_desc", "ohlcv", ["ticker_id", "date"])

    # Convert into a TimescaleDB hypertable partitioned by `date`.
    # One chunk per month is a good default for daily bars (~22 rows/ticker/chunk).
    op.execute(
        "SELECT create_hypertable("
        "  'ohlcv', 'date',"
        "  chunk_time_interval => INTERVAL '1 month',"
        "  if_not_exists => TRUE"
        ")"
    )

    # ── predictions (will become hypertable) ────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("ticker_id", sa.Integer(), nullable=False),
        sa.Column("prediction_date", sa.Date(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("predicted_probability", sa.Float(), nullable=False),
        sa.Column("predicted_class", sa.Integer(), nullable=False),
        sa.Column("actual_class", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["ticker_id"], ["tickers.id"],
            name="fk_predictions_ticker_id_tickers",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "ticker_id", "prediction_date", "horizon_days", "model_name",
            name="pk_predictions",
        ),
    )
    op.create_index(
        "ix_predictions_date_model", "predictions", ["prediction_date", "model_name"]
    )

    # Hypertable on prediction_date. Smaller volume than ohlcv → larger chunks.
    op.execute(
        "SELECT create_hypertable("
        "  'predictions', 'prediction_date',"
        "  chunk_time_interval => INTERVAL '3 months',"
        "  if_not_exists => TRUE"
        ")"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Downgrade
# ─────────────────────────────────────────────────────────────────────────────
def downgrade() -> None:
    op.drop_index("ix_predictions_date_model", table_name="predictions")
    op.drop_table("predictions")

    op.drop_index("ix_ohlcv_ticker_date_desc", table_name="ohlcv")
    op.drop_table("ohlcv")

    op.drop_index("ix_tickers_market_active", table_name="tickers")
    op.drop_index("ix_tickers_market_asset_type", table_name="tickers")
    op.drop_index("ix_tickers_is_active", table_name="tickers")
    op.drop_index("ix_tickers_symbol", table_name="tickers")
    op.drop_table("tickers")

    op.execute("DROP TYPE IF EXISTS asset_type_enum")
    op.execute("DROP TYPE IF EXISTS market_enum")
