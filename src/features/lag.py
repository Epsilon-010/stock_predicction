"""Lagged feature generation.

Given a column `x`, produce `x_lag_1`, `x_lag_2`, … so the model can see the
recent trajectory, not just the current snapshot. Lagged returns are
particularly informative — autocorrelation patterns in returns are weak but
real, especially over short windows.
"""

from __future__ import annotations

import polars as pl


def add_lagged_columns(
    df: pl.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20),
) -> pl.DataFrame:
    """Append `{col}_lag_{n}` for each `(col, n)` combination."""
    exprs = [
        pl.col(col).shift(n).alias(f"{col}_lag_{n}")
        for col in columns
        for n in lags
        if col in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def add_default_lags(df: pl.DataFrame) -> pl.DataFrame:
    """The lag set used by the default pipeline.

    Lags 1-d returns and the main short-window technical signals so the model
    can learn momentum-reversal dynamics.
    """
    cols = [
        c for c in ("returns_1d", "rsi_14", "macd_hist", "volume_zscore_20d") if c in df.columns
    ]
    return add_lagged_columns(df, columns=cols, lags=(1, 2, 3, 5))
