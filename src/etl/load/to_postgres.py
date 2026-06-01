"""Silver (Parquet) → gold (Postgres) bulk loader.

Why sync (psycopg2) and not async:

  * `COPY FROM STDIN` is the fastest way to push millions of rows into
    Postgres — orders of magnitude faster than parameterised INSERTs.
  * Bulk loads are batch jobs, not request-bound — there is nothing to gain
    from event-loop concurrency here. A sync transaction per ticker is
    simpler and gives clean failure isolation.

Idempotency: every operation uses `INSERT … ON CONFLICT DO UPDATE`, so
re-running the loader on the same data converges to the same end state.
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
from loguru import logger
from sqlalchemy import text

from src.db import AssetType, Market
from src.utils.db import SyncSessionLocal, sync_engine

# ─────────────────────────────────────────────────────────────────────────────
# Tickers — small table, ORM-level UPSERT is plenty fast
# ─────────────────────────────────────────────────────────────────────────────
_UPSERT_TICKER_SQL = text(
    """
    INSERT INTO tickers (symbol, market, asset_type, sector_code, listed_at)
    VALUES (:symbol, :market, :asset_type, :sector_code, :listed_at)
    ON CONFLICT (symbol) DO UPDATE SET
        market       = EXCLUDED.market,
        asset_type   = EXCLUDED.asset_type,
        sector_code  = EXCLUDED.sector_code,
        listed_at    = LEAST(tickers.listed_at, EXCLUDED.listed_at),
        is_active    = TRUE,
        updated_at   = NOW()
    RETURNING id, symbol
    """
)


def upsert_tickers_from_parquet(tickers_parquet: Path) -> dict[str, int]:
    """Load the ticker catalogue. Returns `{symbol: ticker_id}`."""
    df = pl.read_parquet(tickers_parquet)
    symbol_to_id: dict[str, int] = {}

    with SyncSessionLocal() as session:
        for row in df.iter_rows(named=True):
            result = session.execute(_UPSERT_TICKER_SQL, row).first()
            assert result is not None
            symbol_to_id[result.symbol] = result.id
        session.commit()

    logger.info("Upserted {} tickers", len(symbol_to_id))
    return symbol_to_id


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV — bulk path via COPY FROM STDIN + staging table
# ─────────────────────────────────────────────────────────────────────────────
_COPY_STAGE_SQL = (
    "COPY _ohlcv_stage "
    "(ticker_id, date, open, high, low, close, volume, open_interest) "
    "FROM STDIN WITH (FORMAT csv, NULL '')"
)

_MERGE_FROM_STAGE_SQL = """
    INSERT INTO ohlcv (ticker_id, date, open, high, low, close, volume, open_interest)
    SELECT ticker_id, date, open, high, low, close, volume, open_interest
    FROM _ohlcv_stage
    ON CONFLICT (ticker_id, date) DO UPDATE SET
        open          = EXCLUDED.open,
        high          = EXCLUDED.high,
        low           = EXCLUDED.low,
        close         = EXCLUDED.close,
        volume        = EXCLUDED.volume,
        open_interest = EXCLUDED.open_interest
"""


def bulk_copy_ohlcv(
    parquet_path: Path,
    symbol_to_id: dict[str, int],
) -> int:
    """Load one ticker's parquet → Postgres. Returns rows merged into `ohlcv`."""
    df = pl.read_parquet(parquet_path)
    if df.is_empty():
        return 0

    symbol = df["symbol"][0]
    ticker_id = symbol_to_id.get(symbol)
    if ticker_id is None:
        logger.warning("{}: ticker not in catalogue, skipping", symbol)
        return 0

    # Add ticker_id as the first column and order for COPY.
    df = df.with_columns(pl.lit(ticker_id).alias("ticker_id")).select(
        ["ticker_id", "date", "open", "high", "low", "close", "volume", "open_interest"]
    )

    # Serialise to in-memory CSV (no header, NULL as empty string).
    buf = io.StringIO()
    df.write_csv(buf, include_header=False, null_value="")
    buf.seek(0)

    # Raw psycopg2 connection — required for copy_expert.
    raw_conn = sync_engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE _ohlcv_stage (LIKE ohlcv INCLUDING DEFAULTS) ON COMMIT DROP"
            )
            cur.copy_expert(_COPY_STAGE_SQL, buf)
            cur.execute(_MERGE_FROM_STAGE_SQL)
            n_rows = cur.rowcount
        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise
    finally:
        raw_conn.close()

    return n_rows


def load_silver_to_postgres(
    interim_root: Path,
    market: Market | None = None,
    asset_types: set[AssetType] | None = None,
) -> dict[str, int]:
    """Upsert the ticker catalogue, then COPY every silver-layer parquet.

    Returns `{symbol: rows_loaded}`.
    """
    tickers_parquet = interim_root / "tickers" / "tickers.parquet"
    if not tickers_parquet.exists():
        raise FileNotFoundError(f"Run the transform step first; missing {tickers_parquet}")

    symbol_to_id = upsert_tickers_from_parquet(tickers_parquet)

    # Hive-partition glob pattern; filter by market/asset_type via path.
    ohlcv_root = interim_root / "ohlcv"
    pattern = "**/*.parquet"
    if market is not None:
        pattern = f"market={market.value}/**/*.parquet"

    counts: dict[str, int] = {}
    n_processed = 0

    for parquet_path in ohlcv_root.glob(pattern):
        if asset_types is not None:
            path_str = str(parquet_path)
            if not any(f"asset_type={at.value}" in path_str for at in asset_types):
                continue

        n = bulk_copy_ohlcv(parquet_path, symbol_to_id)
        if n > 0:
            counts[parquet_path.stem] = n
        n_processed += 1
        if n_processed % 200 == 0:
            logger.info("Loaded {} parquets so far", n_processed)

    logger.info(
        "Load complete | tickers={} | total_rows={}",
        len(counts),
        sum(counts.values()),
    )
    return counts
