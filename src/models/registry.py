"""Model factory + protocol — the Open/Closed seam for adding new models.

To add a new model:
  1. Implement a `fit_<name>(X_train, y_train, X_val, y_val, params, ...)` function.
  2. Register it here via `@register_model("<name>", logger=mlflow_log_fn)`.
  3. Add a section to `model_config.yaml`.

No changes needed in `train.py` or `predict.py` — they look the model up
through this registry. This decouples the training orchestration from the
specific algorithms, which keeps the code Open for extension but Closed for
modification (the O in SOLID).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Protocols — minimal interface every model must satisfy (Interface Segregation)
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class SupportsProbaPredict(Protocol):
    """sklearn-compatible classifier interface used by the rest of the pipeline."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


# Type aliases for the registered callables.
FitFn = Callable[..., SupportsProbaPredict]
MLflowLogger = Callable[[SupportsProbaPredict, str], None]


@dataclass(frozen=True)
class ModelEntry:
    """One registered model — the fit function + how to persist it to MLflow."""

    fit_fn: FitFn
    mlflow_logger: MLflowLogger


_REGISTRY: dict[str, ModelEntry] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Default MLflow loggers
# ─────────────────────────────────────────────────────────────────────────────
def _sklearn_logger(model: SupportsProbaPredict, artifact_path: str) -> None:
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)


def _xgboost_logger(model: SupportsProbaPredict, artifact_path: str) -> None:
    mlflow.xgboost.log_model(model, artifact_path=artifact_path)


# ─────────────────────────────────────────────────────────────────────────────
# Registration API
# ─────────────────────────────────────────────────────────────────────────────
def register_model(name: str, mlflow_logger: MLflowLogger) -> Callable[[FitFn], FitFn]:
    """Decorator that registers a fit function under `name`."""

    def decorator(fit_fn: FitFn) -> FitFn:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _REGISTRY[name] = ModelEntry(fit_fn=fit_fn, mlflow_logger=mlflow_logger)
        return fit_fn

    return decorator


def get_model_entry(name: str) -> ModelEntry:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def registered_models() -> list[str]:
    return sorted(_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Built-in registrations — import the concrete fit functions and bind them
# ─────────────────────────────────────────────────────────────────────────────
# We import inside the module body (not lazily) so that any model declared as
# enabled in model_config.yaml is guaranteed to exist before training starts.
from src.models.baseline import fit_logreg as _fit_logreg  # noqa: E402
from src.models.xgboost_model import fit_xgboost as _fit_xgboost  # noqa: E402

register_model("baseline_logreg", mlflow_logger=_sklearn_logger)(_fit_logreg)
register_model("xgboost", mlflow_logger=_xgboost_logger)(_fit_xgboost)
