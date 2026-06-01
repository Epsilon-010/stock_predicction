"""Thin MLflow helpers — single context manager wraps every training run.

Why a wrapper instead of using `mlflow.start_run()` directly:

  * Centralises the `set_tracking_uri` / `set_experiment` calls so individual
    training scripts don't have to read settings.
  * Adds defensive logging: a failed training run still records *something*
    (status=FAILED + error message) instead of leaving a dangling open run.
  * Provides typed helpers for the patterns we actually use — params, metrics,
    artifacts, model registration.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger
from mlflow.entities import Run

from src.config.settings import get_settings


def _bootstrap() -> None:
    """Configure tracking URI + experiment from settings. Idempotent."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)


@contextmanager
def mlflow_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Iterator[Run]:
    """Start an MLflow run with proper error handling.

    Usage:
        with mlflow_run("baseline_logreg") as run:
            mlflow.log_params({...})
            mlflow.log_metrics({...})
            mlflow.sklearn.log_model(model, "model")
    """
    _bootstrap()
    run = mlflow.start_run(run_name=run_name, tags=tags or {}, nested=nested)
    try:
        logger.info("MLflow run started | name={} | run_id={}", run_name, run.info.run_id)
        yield run
        mlflow.end_run(status="FINISHED")
        logger.info("MLflow run finished | run_id={}", run.info.run_id)
    except Exception as exc:
        logger.exception("MLflow run failed | run_id={} | error={}", run.info.run_id, exc)
        mlflow.set_tag("error", str(exc)[:250])
        mlflow.end_run(status="FAILED")
        raise


def log_params_flat(params: dict[str, Any]) -> None:
    """Log a possibly-nested dict by flattening keys (`a.b.c`)."""
    flat: dict[str, Any] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            flat[prefix] = value

    _walk("", params)
    # MLflow truncates param values at ~6KB; stringify and clip defensively.
    mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})


def log_artifact_text(text: str, filename: str, tmp_dir: Path | None = None) -> None:
    """Persist a string as an MLflow artifact under `filename`."""
    tmp = tmp_dir or Path("/tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    file_path = tmp / filename
    file_path.write_text(text, encoding="utf-8")
    mlflow.log_artifact(str(file_path))


def register_model_if_enabled(
    model_uri: str,
    registry_name: str,
    enabled: bool = True,
) -> str | None:
    """Optionally register a logged model into the MLflow Model Registry.

    Returns the registered model version, or None if registration was skipped
    or failed (we never want registry errors to fail the training run itself).
    """
    if not enabled:
        return None
    try:
        result = mlflow.register_model(model_uri=model_uri, name=registry_name)
        logger.info("Registered model | name={} | version={}", registry_name, result.version)
        return str(result.version)
    except Exception as exc:
        logger.warning("Model registration skipped | name={} | error={}", registry_name, exc)
        return None
