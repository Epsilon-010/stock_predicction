"""End-to-end feature-engineering pipeline: silver Parquet → processed Parquet.

The pipeline:

  1. Discover silver-layer parquets under `data/interim/ohlcv/market=jp/...`.
  2. For each ticker, compute per-ticker features (technical, calendar, lag,
     labels) — these MUST be computed in isolation per ticker to avoid leaking
     rolling-window state across instruments.
  3. Concatenate all tickers into one long DataFrame.
  4. Compute cross-sectional features (rank per date) on the combined frame.
  5. Drop rows with null labels (last `horizon_days` per ticker — no future).
  6. Write `data/processed/features.parquet` (Hive-partitioned by year for
     efficient scan filtering during training).

The output schema is stable and self-describing: any column starting with
`label_` is a target; everything else is safe to use as a feature.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from src.db import AssetType, Market
from src.features.calendar import add_calendar_features
from src.features.cross_sectional import add_cross_sectional_ranks, add_market_relative_returns
from src.features.labels import add_labels
from src.features.lag import add_default_lags
from src.features.technical import add_all_technical


# ─────────────────────────────────────────────────────────────────────────────
# Silver-layer discovery
# ─────────────────────────────────────────────────────────────────────────────
def iter_silver_parquets(
    silver_root: Path,
    market: Market,
    asset_types: set[AssetType],
) -> list[Path]:
    """Return all silver parquets matching the given market + asset types."""
    paths: list[Path] = []
    market_dir = silver_root / "ohlcv" / f"market={market.value}"
    for at in asset_types:
        paths.extend((market_dir / f"asset_type={at.value}").glob("*.parquet"))
    return sorted(paths)


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_ticker_features(
    df: pl.DataFrame,
    horizon_days: int = 5,
    direction_threshold: float = 0.0,
) -> pl.DataFrame:
    """Apply the per-ticker stack: technical → calendar → lag → labels."""
    df = df.sort("date")
    df = add_all_technical(df)
    df = add_calendar_features(df)
    df = add_default_lags(df)
    df = add_labels(df, horizon_days=horizon_days, threshold=direction_threshold)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_features(
    silver_root: Path,
    processed_root: Path,
    market: Market = Market.JP,
    asset_types: set[AssetType] | None = None,
    horizon_days: int = 5,
    direction_threshold: float = 0.0,
    min_history_days: int = 250,
    market_index_symbol: str = "^TPX",
    limit: int | None = None,
) -> Path:
    """Build the full feature matrix and persist to `processed/features.parquet`.

    Args:
        silver_root:        Root of silver Parquet layer (e.g. `data/interim`).
        processed_root:     Where to write the processed parquet.
        market:             Market to process.
        asset_types:        Subset of asset types; default `{STOCK, INDEX}`.
        horizon_days:       Forward window for labels.
        direction_threshold: Threshold for binarising forward returns.
        min_history_days:   Skip tickers with fewer than N bars.
        market_index_symbol: Used for excess-return computation.
        limit:              Process only first N tickers (dev iteration).

    Returns:
        Path to the written parquet file.
    """
    asset_types = asset_types or {AssetType.STOCK, AssetType.INDEX}
    paths = iter_silver_parquets(silver_root, market, asset_types)
    if not paths:
        raise FileNotFoundError(
            f"No silver parquets found under {silver_root} for market={market.value}"
        )

    logger.info("Building features for {} tickers from {}", len(paths), silver_root)

    per_ticker_dfs: list[pl.DataFrame] = []
    n_kept = 0
    n_skipped_short = 0

    for path in paths:
        if limit is not None and n_kept >= limit:
            break
        df = pl.read_parquet(path)
        if df.height < min_history_days:
            n_skipped_short += 1
            continue
        df = build_ticker_features(
            df, horizon_days=horizon_days, direction_threshold=direction_threshold
        )
        per_ticker_dfs.append(df)
        n_kept += 1
        if n_kept % 500 == 0:
            logger.info("Processed {}/{} tickers", n_kept, len(paths))

    if not per_ticker_dfs:
        raise RuntimeError(f"No tickers passed the min_history_days={min_history_days} filter")

    logger.info(
        "Per-ticker features done | kept={} | skipped_short={}",
        n_kept,
        n_skipped_short,
    )

    # Combine — `diagonal_relaxed` tolerates schema drift (e.g. an index has no
    # OBV because volume is 0).
    combined = pl.concat(per_ticker_dfs, how="diagonal_relaxed")

    # Cross-sectional features rely on the combined DataFrame.
    combined = add_cross_sectional_ranks(combined)
    combined = add_market_relative_returns(combined, market_symbol=market_index_symbol)

    # Drop rows that have no label (last N per ticker).
    label_col = f"label_direction_{horizon_days}d"
    combined = combined.drop_nulls(subset=[label_col])

    out_path = processed_root / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path, compression="zstd", statistics=True)

    logger.info(
        "Features written | rows={} | cols={} | path={}",
        combined.height,
        len(combined.columns),
        out_path,
    )
    return out_path
