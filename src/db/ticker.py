"""Ticker — master catalogue of every instrument we ingest.

One row per symbol. Symbol is the unique business key; the integer `id` is the
internal FK target used by `ohlcv` and `predictions` (smaller index, faster joins).
"""

from __future__ import annotations

import enum
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, Enum, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from src.db.ohlcv import OHLCV
    from src.db.prediction import Prediction


class Market(enum.StrEnum):
    """Country / exchange the ticker belongs to."""

    JP = "jp"
    US = "us"


class AssetType(enum.StrEnum):
    """High-level asset class — drives feature engineering and routing."""

    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    FUTURE = "future"
    OPTION = "option"
    BOND = "bond"


class Ticker(Base, TimestampMixin):
    """A tradable instrument with its identifying metadata."""

    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Business key. e.g. "7203.JP" (Toyota), "^TPX" (TOPIX index).
    symbol: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)

    # Human-readable name — nullable because the bronze .txt files don't carry it.
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    market: Mapped[Market] = mapped_column(
        Enum(Market, name="market_enum", native_enum=True),
        nullable=False,
    )
    asset_type: Mapped[AssetType] = mapped_column(
        Enum(AssetType, name="asset_type_enum", native_enum=True),
        nullable=False,
    )

    # For JP equities, derived from the TSE 4-digit numbering scheme
    # (7xxx = transport, 6xxx = electronics, etc.).
    sector_code: Mapped[str | None] = mapped_column(String(16), nullable=True)

    listed_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    delisted_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, server_default="true", nullable=False, index=True
    )

    # ── Relationships ───────────────────────────────────────────────────────
    ohlcv: Mapped[list[OHLCV]] = relationship(
        back_populates="ticker",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="ticker",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_tickers_market_asset_type", "market", "asset_type"),
        Index("ix_tickers_market_active", "market", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Ticker {self.symbol} ({self.asset_type.value}, {self.market.value})>"
