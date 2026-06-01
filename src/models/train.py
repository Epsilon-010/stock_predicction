"""End-to-end training pipeline driven by `model_config.yaml`.

For each model enabled in the config:

  1. Load processed features.
  2. Generate walk-forward CV folds based on `split` config.
  3. For each fold: fit on train, predict on val, log per-fold metrics.
  4. Re-fit on the full train+val window, log final model + summary metrics.
  5. Optionally register the model in the MLflow Model Registry.

Every run lands in MLflow with full reproducibility info — config dump,
metrics, model artefact, and (when enabled) registered name+version.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from src.config.model_config import ModelConfig, ModelSpec, load_model_config
from src.config.settings import get_settings
from src.evaluation.metrics import compute_metrics, format_metrics, metrics_for_mlflow
from src.evaluation.walk_forward import Split, walk_forward_splits
from src.models.dataset import load_features, select_feature_columns, split_by_date, to_xy
from src.models.registry import SupportsProbaPredict, get_model_entry
from src.tracking.mlflow_utils import (
    log_params_flat,
    mlflow_run,
    register_model_if_enabled,
)


def _fit_model(
    model_name: str,
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
) -> SupportsProbaPredict:
    """Look the model up in the registry and forward the call.

    Only forwards the kwargs the underlying fit function actually accepts
    (so legacy fit functions don't need `**kwargs` boilerplate).
    """
    entry = get_model_entry(model_name)
    sig = inspect.signature(entry.fit_fn)
    kwargs: dict[str, Any] = {}
    accepted = sig.parameters
    if "X_val" in accepted:
        kwargs["X_val"] = X_val
    if "y_val" in accepted:
        kwargs["y_val"] = y_val
    if "params" in accepted:
        kwargs["params"] = spec.params
    if "early_stopping_rounds" in accepted and spec.early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = spec.early_stopping_rounds
    return entry.fit_fn(X_train, y_train, **kwargs)


def _log_model(model_name: str, model: SupportsProbaPredict) -> str:
    """Log the trained estimator via the registry-provided MLflow logger."""
    entry = get_model_entry(model_name)
    entry.mlflow_logger(model, "model")
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("No active MLflow run — call this inside `with mlflow_run(...)`.")
    return f"runs:/{active.info.run_id}/model"


def train_one_model(
    model_name: str,
    spec: ModelSpec,
    config: ModelConfig,
    df: pl.DataFrame,
) -> str | None:
    """Train one model end-to-end. Returns the MLflow run_id."""
    label_col = f"label_direction_{config.target.horizon_days}d"
    feature_cols = select_feature_columns(df, label_col)
    logger.info("Training {} | features={} | label={}", model_name, len(feature_cols), label_col)

    splits = list(
        walk_forward_splits(
            train_start=config.split.train_start,
            train_end=config.split.train_end,
            val_start=config.split.val_start,
            val_end=config.split.val_end,
            n_splits=1,
            embargo_days=config.split.embargo_days,
        )
    )

    with mlflow_run(
        run_name=model_name, tags={"horizon_days": str(config.target.horizon_days)}
    ) as run:
        # Reproducibility: log the entire config + the model's own params.
        log_params_flat(
            {
                "model": model_name,
                "horizon_days": config.target.horizon_days,
                "n_features": len(feature_cols),
                **spec.params,
            }
        )

        # ── Per-fold metrics ────────────────────────────────────────────────
        last_split: Split | None = None
        for split in splits:
            last_split = split
            train_df, val_df = split_by_date(
                df, split.train_start, split.train_end, split.val_start, split.val_end
            )
            X_train_pl, y_train_pl = to_xy(train_df, feature_cols, label_col)
            X_val_pl, y_val_pl = to_xy(val_df, feature_cols, label_col)

            if X_train_pl.is_empty() or X_val_pl.is_empty():
                logger.warning("Fold {} empty after filtering — skipping", split.fold)
                continue

            X_train = X_train_pl.to_pandas()
            X_val = X_val_pl.to_pandas()
            y_train = y_train_pl.to_numpy()
            y_val = y_val_pl.to_numpy()

            model = _fit_model(model_name, spec, X_train, y_train, X_val, y_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            metrics = compute_metrics(y_val, y_val_proba)

            logger.info("Fold {} val | {}", split.fold, format_metrics(metrics))
            mlflow.log_metrics(
                metrics_for_mlflow(metrics, prefix=f"fold{split.fold}_val_"),
                step=split.fold,
            )

        # ── Final fit on full train+val window, log model ───────────────────
        if last_split is None:
            logger.error("No splits produced — aborting")
            return None

        full_df, _ = split_by_date(
            df,
            last_split.train_start,
            last_split.val_end,
            last_split.val_end,
            last_split.val_end,
        )
        X_full_pl, y_full_pl = to_xy(full_df, feature_cols, label_col)
        X_full = X_full_pl.to_pandas()
        y_full = y_full_pl.to_numpy()

        final_model = _fit_model(model_name, spec, X_full, y_full, None, None)
        model_uri = _log_model(model_name, final_model)
        logger.info("Logged final model → {}", model_uri)

        # Optional registry registration.
        register_model_if_enabled(
            model_uri=model_uri,
            registry_name=f"{config.mlflow.model_registry_name}_{model_name}",
            enabled=config.mlflow.register_best_model,
        )

        return str(run.info.run_id)


def train_all(config_path: Path | None = None) -> dict[str, str | None]:
    """Train every enabled model from the config. Returns `{name: run_id}`."""
    config = load_model_config(config_path)
    settings = get_settings()
    features_path = settings.paths.processed_data_dir / "features.parquet"
    df = load_features(features_path)

    results: dict[str, str | None] = {}
    for name, spec in config.enabled_models().items():
        try:
            results[name] = train_one_model(name, spec, config, df)
        except Exception:
            logger.exception("Training failed for {}", name)
            results[name] = None
    return results
