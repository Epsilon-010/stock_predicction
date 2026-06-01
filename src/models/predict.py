"""Batch inference: load model from MLflow + processed features → predictions.

Two paths are exposed:

  * `predict_latest(symbols, model_name)` — fast path for the API. Pulls the
    most recent feature row per symbol, runs the model, returns rows.
  * `score_dataframe(df, model_name)` — generic; used by the orchestration
    `predict` flow to backfill predictions for arbitrary date ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import mlflow
import mlflow.pyfunc
import polars as pl
from loguru import logger
from sqlalchemy import text

from src.config.model_config import load_model_config
from src.config.settings import get_settings
from src.models.dataset import load_features, select_feature_columns
from src.utils.db import SyncSessionLocal


@dataclass(frozen=True)
class PredictionRow:
    symbol: str
    date: date
    horizon_days: int
    model_name: str
    predicted_probability: float
    predicted_class: int


def _load_pyfunc_model(model_name: str, stage: str = "None") -> Any:
    """Load the latest registered model. Falls back to last run if registry empty."""
    config = load_model_config()
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    registry_name = f"{config.mlflow.model_registry_name}_{model_name}"
    try:
        uri = (
            f"models:/{registry_name}/{stage}"
            if stage != "None"
            else f"models:/{registry_name}/latest"
        )
        return mlflow.pyfunc.load_model(uri)
    except Exception as exc:
        logger.warning(
            "Could not load from registry ({}); falling back to last run | error={}",
            registry_name,
            exc,
        )
        # Fallback: pick the most recent finished run in the experiment with
        # this model name.
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(settings.mlflow.experiment_name)
        if exp is None:
            raise FileNotFoundError(
                f"Experiment '{settings.mlflow.experiment_name}' missing"
            ) from exc
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{model_name}' and status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise FileNotFoundError(f"No finished run for model '{model_name}'") from exc
        return mlflow.pyfunc.load_model(f"runs:/{runs[0].info.run_id}/model")


def score_dataframe(
    df: pl.DataFrame,
    model_name: str = "xgboost",
    threshold: float = 0.5,
) -> list[PredictionRow]:
    """Score every row of `df` and return a list of typed prediction rows."""
    config = load_model_config()
    label_col = f"label_direction_{config.target.horizon_days}d"
    feature_cols = select_feature_columns(df, label_col)

    model = _load_pyfunc_model(model_name)
    X = df.select(feature_cols).to_pandas()
    proba = model.predict(X)
    # MLflow pyfunc wraps sklearn output — pull the positive-class column.
    if hasattr(proba, "values"):
        proba = proba.values
    if proba.ndim == 2:
        proba = proba[:, 1]

    out: list[PredictionRow] = []
    symbols = df["symbol"].to_list()
    dates = df["date"].to_list()
    for sym, d, p in zip(symbols, dates, proba, strict=True):
        out.append(
            PredictionRow(
                symbol=sym,
                date=d,
                horizon_days=config.target.horizon_days,
                model_name=model_name,
                predicted_probability=float(p),
                predicted_class=int(p >= threshold),
            )
        )
    return out


def predict_latest(
    symbols: list[str],
    model_name: str = "xgboost",
) -> list[PredictionRow]:
    """Score the most recent feature row for each symbol.

    Reads the processed parquet directly — for production this can be
    swapped for a DB query without changing the interface.
    """
    settings = get_settings()
    df = load_features(settings.paths.processed_data_dir / "features.parquet")
    latest = (
        df.filter(pl.col("symbol").is_in(symbols))
        .sort(["symbol", "date"])
        .group_by("symbol")
        .tail(1)
    )
    if latest.is_empty():
        return []
    return score_dataframe(latest, model_name=model_name)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
_UPSERT_PREDICTION_SQL = text(
    """
    INSERT INTO predictions (
        ticker_id, prediction_date, horizon_days, model_name,
        mlflow_run_id, predicted_probability, predicted_class
    )
    VALUES (
        :ticker_id, :prediction_date, :horizon_days, :model_name,
        :mlflow_run_id, :predicted_probability, :predicted_class
    )
    ON CONFLICT (ticker_id, prediction_date, horizon_days, model_name) DO UPDATE SET
        mlflow_run_id         = EXCLUDED.mlflow_run_id,
        predicted_probability = EXCLUDED.predicted_probability,
        predicted_class       = EXCLUDED.predicted_class
    """
)


def write_predictions(
    rows: list[PredictionRow],
    mlflow_run_id: str | None = None,
) -> int:
    """UPSERT a list of predictions into the `predictions` table.

    Idempotent — re-running on the same input updates rather than duplicates.
    Returns the number of rows written.
    """
    if not rows:
        return 0

    n_written = 0
    with SyncSessionLocal() as session:
        # Map every symbol → ticker_id in a single query.
        symbols = sorted({r.symbol for r in rows})
        ticker_lookup = session.execute(
            text("SELECT symbol, id FROM tickers WHERE symbol = ANY(:symbols)"),
            {"symbols": symbols},
        ).all()
        symbol_to_id = {row.symbol: row.id for row in ticker_lookup}

        for r in rows:
            ticker_id = symbol_to_id.get(r.symbol)
            if ticker_id is None:
                logger.warning("Skipping {}: not in tickers table", r.symbol)
                continue
            session.execute(
                _UPSERT_PREDICTION_SQL,
                {
                    "ticker_id": ticker_id,
                    "prediction_date": r.date,
                    "horizon_days": r.horizon_days,
                    "model_name": r.model_name,
                    "mlflow_run_id": mlflow_run_id,
                    "predicted_probability": r.predicted_probability,
                    "predicted_class": r.predicted_class,
                },
            )
            n_written += 1
        session.commit()

    logger.info("Wrote {} predictions to DB", n_written)
    return n_written
