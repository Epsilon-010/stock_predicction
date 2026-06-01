"""ETL package — medallion-style pipeline (bronze → silver → gold).

* `extract`   — read raw Stooq .txt files into Polars DataFrames
* `transform` — clean / validate / persist as Parquet (silver)
* `load`      — bulk-load Parquet into Postgres + TimescaleDB (gold)
"""

from src.etl.extract import (
    TickerMetadata,
    extract_market,
    iter_stooq_files,
    read_stooq_file,
)
from src.etl.load import (
    bulk_copy_ohlcv,
    load_silver_to_postgres,
    upsert_tickers_from_parquet,
)
from src.etl.transform import (
    clean_ohlcv,
    write_ohlcv_parquet,
    write_tickers_parquet,
)

__all__ = [  # noqa: RUF022 (grouped by section, not alphabetical)
    # ── extract ──
    "TickerMetadata",
    "extract_market",
    "iter_stooq_files",
    "read_stooq_file",
    # ── transform ──
    "clean_ohlcv",
    "write_ohlcv_parquet",
    "write_tickers_parquet",
    # ── load ──
    "bulk_copy_ohlcv",
    "load_silver_to_postgres",
    "upsert_tickers_from_parquet",
]
