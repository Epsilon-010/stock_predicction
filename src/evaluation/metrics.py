"""Classification metrics + baselines for the direction-prediction task.

The most informative number for stock-direction models is not raw accuracy
but how much better you do than `always predict positive` (markets tend
upward over long windows). Every `compute_metrics` call therefore returns
both the model's metrics *and* the trivial baselines, so it's impossible
to mistake a market-drift effect for genuine skill.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Suite of metrics for binary classification with calibrated probabilities."""

    roc_auc: float
    average_precision: float
    log_loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    positive_rate: float
    # Trivial baselines computed on the same y_true — for honest comparison.
    baseline_always_positive_accuracy: float
    baseline_majority_accuracy: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute the standard binary-classification suite plus baselines."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    pos_rate = float(y_true.mean())
    majority_acc = max(pos_rate, 1.0 - pos_rate)

    # Some folds may be single-class; AUC/AP are undefined there → emit NaN.
    if len(np.unique(y_true)) < 2:
        roc, ap, ll = float("nan"), float("nan"), float("nan")
    else:
        roc = float(roc_auc_score(y_true, y_proba))
        ap = float(average_precision_score(y_true, y_proba))
        ll = float(log_loss(y_true, np.clip(y_proba, 1e-7, 1 - 1e-7)))

    return ClassificationMetrics(
        roc_auc=roc,
        average_precision=ap,
        log_loss=ll,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0.0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0.0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0.0)),
        positive_rate=pos_rate,
        baseline_always_positive_accuracy=pos_rate,
        baseline_majority_accuracy=majority_acc,
    )


def format_metrics(metrics: ClassificationMetrics) -> str:
    """Pretty single-line summary for logging."""
    return (
        f"AUC={metrics.roc_auc:.4f} | "
        f"AP={metrics.average_precision:.4f} | "
        f"Acc={metrics.accuracy:.4f} (vs baseline {metrics.baseline_majority_accuracy:.4f}) | "
        f"P={metrics.precision:.4f} R={metrics.recall:.4f} F1={metrics.f1:.4f}"
    )


def metrics_for_mlflow(metrics: ClassificationMetrics, prefix: str = "") -> dict[str, Any]:
    """Convert a metrics dataclass to a flat dict suitable for `mlflow.log_metrics`."""
    return {f"{prefix}{k}": v for k, v in metrics.to_dict().items()}
