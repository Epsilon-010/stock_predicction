"""Label generation — the *target* the model learns to predict.

Two label types:

  * `label_return_{N}d`    : raw forward return over the next N trading days.
  * `label_direction_{N}d` : 1 if forward return > `threshold`, else 0.

Both are computed with `shift(-N)` on close prices, which means the *last* N
rows per ticker will have null labels — those rows are dropped before training
to avoid the model seeing examples with no answer.

Look-ahead leakage warning: NEVER use these columns as features. They are
literally the future. The build pipeline keeps them in a column prefixed
`label_` so it's impossible to accidentally include them via `select(...)`.
"""

from __future__ import annotations

import polars as pl


def add_forward_returns(
    df: pl.DataFrame,
    horizon_days: int = 5,
) -> pl.DataFrame:
    """`label_return_{N}d` = (close[t+N] / close[t]) - 1."""
    fwd_close = pl.col("close").shift(-horizon_days)
    return df.with_columns(
        ((fwd_close / pl.col("close")) - 1).alias(f"label_return_{horizon_days}d")
    )


def add_direction_label(
    df: pl.DataFrame,
    horizon_days: int = 5,
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Binary direction at horizon. Adds `label_direction_{N}d` ∈ {0, 1}."""
    return_col = f"label_return_{horizon_days}d"
    if return_col not in df.columns:
        df = add_forward_returns(df, horizon_days=horizon_days)

    return df.with_columns(
        pl.when(pl.col(return_col).is_null())
        .then(None)
        .when(pl.col(return_col) > threshold)
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias(f"label_direction_{horizon_days}d")
    )


def add_labels(
    df: pl.DataFrame,
    horizon_days: int = 5,
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Convenience: forward returns + direction label in one call."""
    df = add_forward_returns(df, horizon_days=horizon_days)
    return add_direction_label(df, horizon_days=horizon_days, threshold=threshold)
