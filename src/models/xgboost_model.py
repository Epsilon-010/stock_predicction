"""XGBoost classifier with automatic class-imbalance handling + early stopping.

XGBoost handles NaN values natively (no imputation needed), is robust to
unscaled features, and ships with `scale_pos_weight` for imbalanced labels —
which is exactly what we have when forecasting direction (markets are
positive-drifting, so the positive class is ~52-55% in most years).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


def _scale_pos_weight(y: np.ndarray) -> float:
    """`(N_negatives / N_positives)` — passed to XGBoost for class balancing."""
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def build_xgboost(params: dict[str, Any] | None = None) -> xgb.XGBClassifier:
    """Construct an XGBoost classifier with sane defaults for tabular finance."""
    params = dict(params or {})
    params.setdefault("n_estimators", 500)
    params.setdefault("max_depth", 6)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("subsample", 0.8)
    params.setdefault("colsample_bytree", 0.8)
    params.setdefault("min_child_weight", 5)
    params.setdefault("gamma", 0.1)
    params.setdefault("reg_alpha", 0.1)
    params.setdefault("reg_lambda", 1.0)
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "auc")
    params.setdefault("tree_method", "hist")
    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", -1)
    return xgb.XGBClassifier(**params)


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int | None = 50,
) -> xgb.XGBClassifier:
    """Fit XGBoost with optional early stopping on the validation set."""
    params = dict(params or {})
    params.setdefault("scale_pos_weight", _scale_pos_weight(y_train))

    if early_stopping_rounds is not None and X_val is not None and y_val is not None:
        params["early_stopping_rounds"] = early_stopping_rounds

    model = build_xgboost(params)
    eval_set = (
        [(X_train, y_train), (X_val, y_val)] if X_val is not None and y_val is not None else None
    )
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model
