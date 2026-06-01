"""Persist cleaned OHLCV bars to the silver layer (Parquet, Hive-partitioned).

Output layout:

    data/interim/ohlcv/market=<m>/asset_type=<t>/<safe_symbol>.parquet
    data/interim/tickers/tickers.parquet

`safe_symbol` strips characters that are awkward in filesystems
(`^TPX` → `TPX`, `7203.JP` → `7203_JP`). The original symbol is preserved
as a column inside the parquet itself, so the mapping is non-destructive.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from src.db import AssetType, Market
from src.etl.extract.from_stooq import TickerMetadata


def _safe_symbol(symbol: str) -> str:
    """Sanitise a ticker symbol for use in a filename."""
    return symbol.replace("^", "").replace(".", "_")


def silver_path_for_ticker(
    interim_root: Path,
    market: Market,
    asset_type: AssetType,
    symbol: str,
) -> Path:
    """Compute the canonical silver-layer path for one ticker."""
    return (
        interim_root
        / "ohlcv"
        / f"market={market.value}"
        / f"asset_type={asset_type.value}"
        / f"{_safe_symbol(symbol)}.parquet"
    )


def write_ohlcv_parquet(
    df: pl.DataFrame,
    metadata: TickerMetadata,
    interim_root: Path,
) -> Path:
    """Write one ticker's cleaned bars to the silver layer."""
    out_path = silver_path_for_ticker(
        interim_root,
        metadata.market,
        metadata.asset_type,
        metadata.symbol,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve the original symbol on every row — handy for cross-ticker reads.
    df_to_write = df.with_columns(pl.lit(metadata.symbol).alias("symbol"))
    df_to_write.write_parquet(out_path, compression="zstd", statistics=True)
    return out_path


def write_tickers_parquet(
    tickers: list[TickerMetadata],
    interim_root: Path,
) -> Path:
    """Persist the ticker catalogue used by the load step."""
    out_path = interim_root / "tickers" / "tickers.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame(
        {
            "symbol": [t.symbol for t in tickers],
            "market": [t.market.value for t in tickers],
            "asset_type": [t.asset_type.value for t in tickers],
            "sector_code": [t.sector_code for t in tickers],
            "listed_at": [t.listed_at for t in tickers],
        }
    )
    df.write_parquet(out_path, compression="zstd")
    logger.info("Wrote {} ticker rows → {}", df.height, out_path)
    return out_path
