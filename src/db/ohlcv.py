"""OHLCV — daily price bars, the heart of the dataset.

This table is converted to a **TimescaleDB hypertable** partitioned by `date`
in the initial migration. From the ORM's perspective it looks like any other
table; the partitioning is transparent.

PK is composite `(ticker_id, date)` because:
  * hypertables require the partition column (date) to be in the primary key
  * `(ticker_id, date)` is also the natural business key — we never want two
    bars for the same ticker on the same day
"""

from __future__ import annotations

from datetime import date as date_t
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Date, ForeignKey, Index, Numeric
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base

if TYPE_CHECKING:
    from src.db.ticker import Ticker


class OHLCV(Base):
    """A single end-of-day bar for one instrument.

    Numeric(18, 6) holds prices up to ~10^12 with 6 decimal places — comfortably
    fits both yen-denominated equities (~thousands) and crypto-style prices
    (fractional cents) in the same column.
    """

    __tablename__ = "ohlcv"

    ticker_id: Mapped[int] = mapped_column(
        ForeignKey("tickers.id", ondelete="CASCADE"),
        primary_key=True,
    )
    date: Mapped[date_t] = mapped_column(Date, primary_key=True)

    open: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)

    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    open_interest: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # ── Relationship ────────────────────────────────────────────────────────
    ticker: Mapped[Ticker] = relationship(back_populates="ohlcv")

    __table_args__ = (
        # Reverse-chronological lookup ("last N bars for ticker X") is the most
        # common access pattern, so the index leads with ticker_id and orders
        # date descending. TimescaleDB chunk pruning still kicks in regardless.
        Index("ix_ohlcv_ticker_date_desc", "ticker_id", "date", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        return f"<OHLCV ticker={self.ticker_id} date={self.date} close={self.close}>"
