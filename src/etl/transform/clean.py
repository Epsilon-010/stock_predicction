from __future__ import annotations

import polars as pl
from loguru import logger


def clean_ohlcv(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Apply validation rules to a single ticker's daily bars."""
    if df.is_empty():
        return df

    n_in = df.height

    df = df.drop_nulls(subset=["date", "open", "high", "low", "close"])

    df = df.filter(
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & (pl.col("high") >= pl.col("low"))
    )

    df = df.unique(subset=["date"], keep="first", maintain_order=False)
    df = df.sort("date")

    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").fill_null(0))

    n_out = df.height
    if n_in != n_out:
        logger.debug(
            "{}: cleaned {} → {} rows ({} dropped)",
            symbol,
            n_in,
            n_out,
            n_in - n_out,
        )

    return df
