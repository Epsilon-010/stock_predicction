"""Prediction — model outputs persisted for analysis, monitoring, and backtest.

Also converted to a TimescaleDB hypertable in the initial migration, partitioned
by `prediction_date`. The composite PK encodes the business key
`(ticker, date, horizon, model)` so re-running a model with the same parameters
is idempotent (a re-run UPSERTs into the same row).
"""

from __future__ import annotations

from datetime import date as date_t
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db.base import Base

if TYPE_CHECKING:
    from src.db.ticker import Ticker


class Prediction(Base):
    """A single forward-looking prediction emitted by a model."""

    __tablename__ = "predictions"

    # ── Composite primary key (business key) ────────────────────────────────
    ticker_id: Mapped[int] = mapped_column(
        ForeignKey("tickers.id", ondelete="CASCADE"),
        primary_key=True,
    )
    prediction_date: Mapped[date_t] = mapped_column(Date, primary_key=True)
    horizon_days: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), primary_key=True)

    # ── Payload ─────────────────────────────────────────────────────────────
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)

    # Filled in by a batch job once the horizon has elapsed and we know the
    # true outcome. Enables online accuracy / drift monitoring.
    actual_class: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ── Relationship ────────────────────────────────────────────────────────
    ticker: Mapped[Ticker] = relationship(back_populates="predictions")

    __table_args__ = (
        # Common query: "all predictions emitted on date X across the universe".
        Index("ix_predictions_date_model", "prediction_date", "model_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction model={self.model_name} ticker={self.ticker_id} "
            f"date={self.prediction_date} p={self.predicted_probability:.3f}>"
        )
