"""Prefect 3 flows that orchestrate the project's pipelines.

CLI entry points (exposed via the Makefile):

    python -m src.orchestration.flows ingest      # raw .txt → Postgres
    python -m src.orchestration.flows features    # silver → processed features
    python -m src.orchestration.flows train       # placeholder
    python -m src.orchestration.flows predict     # placeholder

Each flow is a Prefect `@flow`; tasks are `@task`. Locally they execute
synchronously without needing a Prefect server. Pointing `PREFECT_API_URL`
at a server (cloud or self-hosted) is what later enables scheduling, retries
with backoff, and observability — without changing any code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from prefect import flow, task

from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.db import AssetType, Market
from src.etl.extract.from_stooq import extract_market
from src.etl.load.to_postgres import load_silver_to_postgres
from src.etl.transform.clean import clean_ohlcv
from src.etl.transform.to_parquet import write_ohlcv_parquet, write_tickers_parquet
from src.features.build_features import build_features
from src.models.predict import predict_latest, write_predictions
from src.models.train import train_all


# ─────────────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────────────
@task(name="extract_and_transform", log_prints=True)
def extract_and_transform_task(
    raw_root: Path,
    interim_root: Path,
    market: Market,
    asset_types: set[AssetType] | None = None,
    limit: int | None = None,
) -> int:
    """Walk raw .txt files for `market`, clean them, persist to silver."""
    tickers_seen = []
    n_files = 0

    for metadata, df in extract_market(raw_root, market, asset_types=asset_types, limit=limit):
        cleaned = clean_ohlcv(df, metadata.symbol)
        if cleaned.is_empty():
            continue
        write_ohlcv_parquet(cleaned, metadata, interim_root)
        tickers_seen.append(metadata)
        n_files += 1

    if tickers_seen:
        write_tickers_parquet(tickers_seen, interim_root)

    logger.info("Transform complete: {} tickers → silver", n_files)
    return n_files


@task(name="load_to_postgres", log_prints=True)
def load_to_postgres_task(
    interim_root: Path,
    market: Market,
    asset_types: set[AssetType] | None = None,
) -> int:
    counts = load_silver_to_postgres(interim_root, market=market, asset_types=asset_types)
    return sum(counts.values())


@task(name="build_features", log_prints=True)
def build_features_task(
    interim_root: Path,
    processed_root: Path,
    market: Market,
    asset_types: set[AssetType] | None = None,
    horizon_days: int = 5,
    direction_threshold: float = 0.0,
    min_history_days: int = 250,
    limit: int | None = None,
) -> Path:
    return build_features(
        silver_root=interim_root,
        processed_root=processed_root,
        market=market,
        asset_types=asset_types,
        horizon_days=horizon_days,
        direction_threshold=direction_threshold,
        min_history_days=min_history_days,
        limit=limit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Flows
# ─────────────────────────────────────────────────────────────────────────────
@flow(name="ingest", log_prints=True)
def ingest_flow(
    market: str = "jp",
    asset_types: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """End-to-end ingest: raw .txt → parquet → postgres.

    Args:
        market:      Stooq market key (e.g. `"jp"`).
        asset_types: Restrict to a subset, e.g. `["stock", "etf"]`. None = all.
        limit:       Stop after `limit` tickers (dev iteration).
    """
    settings = get_settings()
    market_enum = Market(market)
    types = {AssetType(t) for t in asset_types} if asset_types else None

    raw_root = settings.paths.raw_data_dir
    interim_root = settings.paths.interim_data_dir

    n = extract_and_transform_task(
        raw_root, interim_root, market_enum, asset_types=types, limit=limit
    )
    if n == 0:
        logger.warning("No tickers extracted — skipping load step")
        return

    rows = load_to_postgres_task(interim_root, market_enum, asset_types=types)
    logger.info("Ingest finished | tickers={} | rows_loaded={}", n, rows)


@flow(name="features", log_prints=True)
def features_flow(
    market: str = "jp",
    asset_types: list[str] | None = None,
    horizon_days: int = 5,
    direction_threshold: float = 0.0,
    min_history_days: int = 250,
    limit: int | None = None,
) -> None:
    """Silver → processed: compute the full feature matrix from silver parquets."""
    settings = get_settings()
    market_enum = Market(market)
    types = (
        {AssetType(t) for t in asset_types} if asset_types else {AssetType.STOCK, AssetType.INDEX}
    )

    out_path = build_features_task(
        interim_root=settings.paths.interim_data_dir,
        processed_root=settings.paths.processed_data_dir,
        market=market_enum,
        asset_types=types,
        horizon_days=horizon_days,
        direction_threshold=direction_threshold,
        min_history_days=min_history_days,
        limit=limit,
    )
    logger.info("Features flow finished → {}", out_path)


@task(name="train_all_models", log_prints=True)
def train_all_models_task(config_path: Path | None = None) -> dict[str, str | None]:
    return train_all(config_path=config_path)


@task(name="predict_and_persist", log_prints=True)
def predict_and_persist_task(
    symbols: list[str],
    model_name: str,
    persist: bool,
) -> int:
    rows = predict_latest(symbols=symbols, model_name=model_name)
    if not rows:
        logger.warning("No predictions produced — empty input or missing features")
        return 0
    if persist:
        return write_predictions(rows)
    for r in rows:
        logger.info(
            "{} {} | p={:.4f} class={}",
            r.symbol,
            r.date,
            r.predicted_probability,
            r.predicted_class,
        )
    return len(rows)


@flow(name="train", log_prints=True)
def train_flow(config_path: str | None = None) -> None:
    """Train every model enabled in `model_config.yaml`.

    Each model gets its own MLflow run with metrics, params and a logged
    artefact. If `register_best_model: true`, the final model is also pushed
    to the MLflow Model Registry.
    """
    results = train_all_models_task(config_path=Path(config_path) if config_path else None)
    successes = {k: v for k, v in results.items() if v is not None}
    failures = [k for k, v in results.items() if v is None]
    logger.info(
        "Train flow finished | trained={} | failed={}",
        list(successes.keys()),
        failures,
    )


@flow(name="predict", log_prints=True)
def predict_flow(
    symbols: list[str] | None = None,
    model_name: str = "xgboost",
    persist: bool = True,
) -> None:
    """Generate predictions for the requested symbols and write to Postgres.

    If `symbols` is None, defaults to "all known symbols in the processed
    feature matrix" — typically what you want for nightly scoring.
    """
    if not symbols:
        # Read just the symbol column to keep memory low.
        import polars as pl

        settings = get_settings()
        df = (
            pl.scan_parquet(settings.paths.processed_data_dir / "features.parquet")
            .select("symbol")
            .unique()
            .collect()
        )
        symbols = df["symbol"].to_list()

    logger.info("Predicting for {} symbols with model={}", len(symbols), model_name)
    n_written = predict_and_persist_task(symbols=symbols, model_name=model_name, persist=persist)
    logger.info("Predict flow finished | rows={}", n_written)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.orchestration.flows",
        description="Run a Prefect flow from the command line.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Raw .txt → parquet → postgres.")
    p_ingest.add_argument("--market", default="jp", help="Market key (default: jp).")
    p_ingest.add_argument(
        "--asset-types",
        nargs="*",
        default=None,
        help="Restrict to asset types (e.g. stock etf).",
    )
    p_ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tickers (dev iteration).",
    )

    p_feat = sub.add_parser("features", help="Silver → processed feature matrix.")
    p_feat.add_argument("--market", default="jp")
    p_feat.add_argument("--asset-types", nargs="*", default=None)
    p_feat.add_argument("--horizon-days", type=int, default=5)
    p_feat.add_argument("--direction-threshold", type=float, default=0.0)
    p_feat.add_argument("--min-history-days", type=int, default=250)
    p_feat.add_argument("--limit", type=int, default=None)

    p_train = sub.add_parser("train", help="Train all enabled models.")
    p_train.add_argument(
        "--config",
        default=None,
        help="Path to a model_config.yaml (default: src/config/model_config.yaml).",
    )

    p_pred = sub.add_parser("predict", help="Generate predictions and persist.")
    p_pred.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        default=None,
        help="Symbol to predict (repeat for multiple); default = all known.",
    )
    p_pred.add_argument("--model", default="xgboost", help="Model name (default: xgboost).")
    p_pred.add_argument(
        "--no-persist",
        action="store_true",
        help="Print predictions to stdout instead of writing to Postgres.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _build_parser().parse_args(argv)

    if args.command == "ingest":
        ingest_flow(
            market=args.market,
            asset_types=args.asset_types,
            limit=args.limit,
        )
    elif args.command == "features":
        features_flow(
            market=args.market,
            asset_types=args.asset_types,
            horizon_days=args.horizon_days,
            direction_threshold=args.direction_threshold,
            min_history_days=args.min_history_days,
            limit=args.limit,
        )
    elif args.command == "train":
        train_flow(config_path=args.config)
    elif args.command == "predict":
        predict_flow(
            symbols=args.symbols,
            model_name=args.model,
            persist=not args.no_persist,
        )
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
