"""Prediction service — orchestrates model inference + caching.

The service:
  1. Looks the request up in Redis cache (key = `pred:<model>:<symbol>`).
  2. On miss, calls `predict_latest` from the ML pipeline (which loads the
     MLflow model and scores the latest feature row).
  3. Writes the result back to Redis with the configured TTL.

Inference itself is CPU-bound and synchronous (sklearn/xgboost). We run it
in a thread pool via `asyncio.to_thread` so it doesn't block the event loop.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from redis.asyncio import Redis

from app.schemas.prediction import PredictionOut
from src.config.settings import get_settings
from src.models.predict import PredictionRow, predict_latest
from src.utils.redis_client import cache_get_json, cache_set_json


def _cache_key(model_name: str, symbol: str) -> str:
    return f"pred:{model_name}:{symbol.upper()}"


def _row_to_schema(row: PredictionRow) -> PredictionOut:
    return PredictionOut(
        symbol=row.symbol,
        prediction_date=row.date,
        horizon_days=row.horizon_days,
        model_name=row.model_name,
        predicted_probability=row.predicted_probability,
        predicted_class=row.predicted_class,
    )


class PredictionService:
    """Cached prediction lookups."""

    def __init__(self, redis: Redis, cache_ttl_seconds: int | None = None) -> None:
        self._redis = redis
        self._ttl = cache_ttl_seconds or get_settings().redis.cache_ttl_seconds

    async def predict(
        self,
        symbols: list[str],
        model_name: str = "xgboost",
        force_refresh: bool = False,
    ) -> tuple[list[PredictionOut], bool]:
        """Score the requested symbols, returning `(results, cache_hit)`.

        `cache_hit` is True iff *every* requested symbol was served from cache.
        Partial misses are recomputed; the recomputed values are cached too.
        """
        results: dict[str, PredictionOut] = {}
        misses: list[str] = []

        if not force_refresh:
            for sym in symbols:
                cached = await cache_get_json(_cache_key(model_name, sym))
                if cached is not None:
                    try:
                        results[sym] = PredictionOut.model_validate(cached)
                        continue
                    except Exception as exc:  # corrupt cache entry → drop & recompute
                        logger.warning("Discarding bad cache entry for {}: {}", sym, exc)
                misses.append(sym)
        else:
            misses = list(symbols)

        if misses:
            # CPU-bound inference → off-load to a worker thread.
            fresh_rows = await asyncio.to_thread(predict_latest, misses, model_name)
            for row in fresh_rows:
                schema = _row_to_schema(row)
                results[row.symbol] = schema
                await cache_set_json(
                    _cache_key(model_name, row.symbol),
                    schema.model_dump(mode="json"),
                    ttl_seconds=self._ttl,
                )

        ordered = [results[s] for s in symbols if s in results]
        cache_hit = len(misses) == 0 and len(ordered) == len(symbols)
        return ordered, cache_hit
