from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date as date_t
from pathlib import Path
from typing import NamedTuple

import polars as pl
from loguru import logger

from src.db import AssetType, Market

# Maps Stooq folder names → AssetType.
_FOLDER_TO_ASSET_TYPE: dict[str, AssetType] = {
    "tse stocks": AssetType.STOCK,
    "tse etfs": AssetType.ETF,
    "tse indices": AssetType.INDEX,
    "tse futures": AssetType.FUTURE,
    "tse options": AssetType.OPTION,
    "tse corporate bonds": AssetType.BOND,
    "tse treasury bonds": AssetType.BOND,
}

# Rename Stooq's bracketed header into snake_case columns.
_STOOQ_RENAME: dict[str, str] = {
    "<TICKER>": "ticker",
    "<PER>": "period",
    "<DATE>": "date_raw",
    "<TIME>": "time_raw",
    "<OPEN>": "open",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
    "<VOL>": "volume",
    "<OPENINT>": "open_interest",
}

# JP equities use 4-character codes (`7203`, `130a`); the leading digit
# tells us the sector. We require the first char to be a digit but allow
# the rest to mix digits + letters (post-2024 issuances do this).
_NUMERIC_TICKER_RE = re.compile(r"^(\d)[\da-z]{3}", re.IGNORECASE)


class TickerMetadata(NamedTuple):
    """Metadata derived from a single .txt file's path and first-bar date."""

    symbol: str
    market: Market
    asset_type: AssetType
    sector_code: str | None
    listed_at: date_t | None
    file_path: Path


# ─────────────────────────────────────────────────────────────────────────────
# Path / symbol helpers (pure functions — unit-tested)
# ─────────────────────────────────────────────────────────────────────────────
def symbol_from_path(path: Path) -> str:
    """`1305.jp.txt` → `1305.JP`; `^tpx.txt` → `^TPX`."""
    name = path.name
    if name.lower().endswith(".txt"):
        name = name[:-4]
    return name.upper()


def asset_type_from_path(path: Path, raw_root: Path) -> AssetType | None:
    """Identify the asset-type folder under `raw_root/<market>/`.

    `path`     = `data/raw/jp/tse stocks/1/7203.jp.txt`
    `raw_root` = `data/raw`
    → returns `AssetType.STOCK` (from "tse stocks").
    """
    try:
        rel = path.relative_to(raw_root).parts
    except ValueError:
        return None
    if len(rel) < 2:
        return None
    return _FOLDER_TO_ASSET_TYPE.get(rel[1].lower())


def sector_from_symbol(symbol: str, asset_type: AssetType) -> str | None:
    """JP 4-digit equities → sector bucket (`7203` → `"7000"`). Else None."""
    if asset_type != AssetType.STOCK:
        return None
    match = _NUMERIC_TICKER_RE.match(symbol)
    if not match:
        return None
    return f"{match.group(1)}000"


# ─────────────────────────────────────────────────────────────────────────────
# File iteration
# ─────────────────────────────────────────────────────────────────────────────
def iter_stooq_files(
    market_dir: Path,
    raw_root: Path,
    asset_types: set[AssetType] | None = None,
) -> Iterator[Path]:

    if not market_dir.exists():
        logger.warning("Market directory not found: {}", market_dir)
        return

    for path in market_dir.rglob("*.txt"):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        asset_type = asset_type_from_path(path, raw_root)
        if asset_type is None:
            continue
        if asset_types is not None and asset_type not in asset_types:
            continue
        yield path


# ─────────────────────────────────────────────────────────────────────────────
# Read + normalise a single file
# ─────────────────────────────────────────────────────────────────────────────
def read_stooq_file(path: Path) -> pl.DataFrame | None:
    """Parse one `.txt` and return a normalised DataFrame.

    Output columns:
        ticker, date, open, high, low, close, volume, open_interest

    Returns None on parse failure or empty file.
    """
    try:
        df = pl.read_csv(path, has_header=True)
    except Exception as exc:
        logger.error("Failed to read {}: {}", path.name, exc)
        return None

    if df.is_empty():
        return None

    # Normalise column names (only the ones we recognise).
    rename_map = {c: _STOOQ_RENAME[c] for c in df.columns if c in _STOOQ_RENAME}
    df = df.rename(rename_map)

    required = {"ticker", "date_raw", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("{} missing required columns: {}", path.name, missing)
        return None

    # Parse YYYYMMDD → Date.
    df = df.with_columns(
        pl.col("date_raw").cast(pl.String).str.strptime(pl.Date, "%Y%m%d").alias("date"),
    ).drop(["date_raw", "time_raw", "period"], strict=False)

    keep = ["ticker", "date", "open", "high", "low", "close", "volume", "open_interest"]
    return df.select([c for c in keep if c in df.columns])


# ─────────────────────────────────────────────────────────────────────────────
# Top-level extract — generator yielding (metadata, dataframe) per file
# ─────────────────────────────────────────────────────────────────────────────
def extract_market(
    raw_root: Path,
    market: Market,
    asset_types: set[AssetType] | None = None,
    limit: int | None = None,
) -> Iterator[tuple[TickerMetadata, pl.DataFrame]]:
    """Walk every Stooq file for `market` and yield (metadata, dataframe).

    Args:
        raw_root: Parent of the market directory (typically `data/raw`).
        market:  Which market to ingest.
        asset_types: Filter to a subset of asset types; None = all.
        limit:   Stop after this many files (useful for dev iteration).
    """
    market_dir = raw_root / market.value
    logger.info(
        "Extracting market={} from {} | asset_types={}",
        market.value,
        market_dir,
        sorted(t.value for t in asset_types) if asset_types else "all",
    )

    n_yielded = 0
    n_skipped = 0

    for path in iter_stooq_files(market_dir, raw_root, asset_types=asset_types):
        if limit is not None and n_yielded >= limit:
            break

        df = read_stooq_file(path)
        if df is None or df.is_empty():
            n_skipped += 1
            continue

        asset_type = asset_type_from_path(path, raw_root)
        assert asset_type is not None  # iter_stooq_files filtered these out

        symbol = symbol_from_path(path)
        sector = sector_from_symbol(symbol, asset_type)
        listed_at = df.get_column("date").min()

        metadata = TickerMetadata(
            symbol=symbol,
            market=market,
            asset_type=asset_type,
            sector_code=sector,
            listed_at=listed_at,  # type: ignore[arg-type]
            file_path=path,
        )

        n_yielded += 1
        if n_yielded % 500 == 0:
            logger.info("Extracted {} tickers so far ({} skipped)", n_yielded, n_skipped)

        yield metadata, df

    logger.info("Extract complete | yielded={} | skipped={}", n_yielded, n_skipped)
