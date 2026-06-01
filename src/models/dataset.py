"""Load the processed feature matrix and split it for training.

The processed parquet is a long DataFrame (one row per ticker × date), with:

  * identifiers: `symbol`, `date`
  * raw bars:    `open`, `high`, `low`, `close`, `volume`, `open_interest`
                 (kept for traceability but NEVER used as features)
  * features:    everything numeric that's neither identifier, raw bar, nor label
  * labels:      every column starting with `label_`

The helpers below centralise feature selection and date-based splitting so
that every model trains on exactly the same X/y contract.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger

# Columns we *never* feed to the model as inputs.
_RAW_BAR_COLUMNS: frozenset[str] = frozenset(
    {"open", "high", "low", "close", "volume", "open_interest"}
)
_IDENTIFIER_COLUMNS: frozenset[str] = frozenset({"date", "symbol", "ticker"})


def load_features(features_path: Path) -> pl.DataFrame:
    """Read the processed feature parquet."""
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features not found at {features_path} — run `make features` first."
        )
    df = pl.read_parquet(features_path)
    logger.info(
        "Loaded features | rows={} | cols={} | path={}",
        df.height,
        len(df.columns),
        features_path,
    )
    return df


def select_feature_columns(df: pl.DataFrame, label_col: str) -> list[str]:
    """Pick the columns that should be fed to the model as X.

    Rules:
      * numeric dtype
      * not an identifier (`date`, `symbol`, `ticker`)
      * not a raw OHLCV column
      * not a label
    """
    excluded = _RAW_BAR_COLUMNS | _IDENTIFIER_COLUMNS | {label_col}
    return [
        c
        for c in df.columns
        if c not in excluded and not c.startswith("label_") and df.schema[c].is_numeric()
    ]


def split_by_date(
    df: pl.DataFrame,
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return `(train_df, val_df)` filtered by the date column."""
    train_df = df.filter((pl.col("date") >= train_start) & (pl.col("date") <= train_end))
    val_df = df.filter((pl.col("date") >= val_start) & (pl.col("date") <= val_end))
    return train_df, val_df


def to_xy(
    df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[pl.DataFrame, pl.Series]:
    """Slice into (X, y), dropping rows where the label is null."""
    df = df.drop_nulls(subset=[label_col])
    return df.select(feature_cols), df[label_col]
