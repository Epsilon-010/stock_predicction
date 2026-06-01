"""ML model layer: dataset loading, model definitions, training, inference."""

from src.models.baseline import build_logreg_pipeline, fit_logreg
from src.models.dataset import (
    load_features,
    select_feature_columns,
    split_by_date,
    to_xy,
)
from src.models.predict import (
    PredictionRow,
    predict_latest,
    score_dataframe,
    write_predictions,
)
from src.models.registry import (
    ModelEntry,
    SupportsProbaPredict,
    get_model_entry,
    register_model,
    registered_models,
)
from src.models.train import train_all, train_one_model
from src.models.xgboost_model import build_xgboost, fit_xgboost

__all__ = [  # noqa: RUF022 (grouped by section, not alphabetical)
    # ── dataset ──
    "load_features",
    "select_feature_columns",
    "split_by_date",
    "to_xy",
    # ── models ──
    "build_logreg_pipeline",
    "build_xgboost",
    "fit_logreg",
    "fit_xgboost",
    # ── train ──
    "train_all",
    "train_one_model",
    # ── predict ──
    "PredictionRow",
    "predict_latest",
    "score_dataframe",
    "write_predictions",
    # ── registry ──
    "ModelEntry",
    "SupportsProbaPredict",
    "get_model_entry",
    "register_model",
    "registered_models",
]
