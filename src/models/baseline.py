"""Baseline model: logistic regression on standardised features.

Why a baseline matters: a gradient-boosted ensemble that doesn't beat
logistic regression on AUC is not learning anything non-linear that
matters — it's just memorising noise. Every reported result for XGBoost
should be read alongside this baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logreg_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Imputer → scaler → logistic regression.

    The pipeline is fit on training data only (no leakage into validation),
    and scikit-learn handles the proper transform/inverse semantics
    automatically when called via `.predict_proba(X_val)`.
    """
    params = dict(params or {})
    params.setdefault("C", 1.0)
    params.setdefault("penalty", "l2")
    params.setdefault("solver", "liblinear")
    params.setdefault("class_weight", "balanced")
    params.setdefault("max_iter", 1000)
    params.setdefault("random_state", 42)

    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ]
    )


def fit_logreg(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    pipeline = build_logreg_pipeline(params)
    pipeline.fit(X_train, y_train)
    return pipeline
