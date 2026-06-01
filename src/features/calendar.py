"""Calendar / seasonality features derived from the `date` column.

These features capture well-documented market anomalies:
  * Monday effect, turn-of-month effect, January effect, …
  * End-of-quarter window dressing
  * Post-holiday volatility

Cheap to compute, often useful, and never the source of look-ahead leakage
(today's date is known at prediction time).
"""

from __future__ import annotations

import polars as pl


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """Append date-derived columns to a DataFrame with a `date` column."""
    return df.with_columns(
        pl.col("date").dt.weekday().alias("day_of_week"),  # 1=Mon … 7=Sun
        pl.col("date").dt.day().alias("day_of_month"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.quarter().alias("quarter"),
        pl.col("date").dt.week().alias("week_of_year"),
        pl.col("date").dt.year().alias("year"),
        # Boolean flags for "edge" days that tend to behave differently.
        (pl.col("date").dt.day() == 1).alias("is_month_start"),
        (pl.col("date") == pl.col("date").dt.month_end()).alias("is_month_end"),
        pl.col("date").dt.month().is_in([3, 6, 9, 12]).alias("is_quarter_end_month"),
    )
