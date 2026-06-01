"""Unit tests for `src.evaluation.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, format_metrics, metrics_for_mlflow

pytestmark = pytest.mark.unit


def test_perfect_classifier_has_auc_one() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.95])
    m = compute_metrics(y, p)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.accuracy == pytest.approx(1.0)


def test_random_classifier_has_auc_around_half() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=1000)
    p = rng.uniform(size=1000)
    m = compute_metrics(y, p)
    assert 0.45 <= m.roc_auc <= 0.55


def test_baseline_metrics_match_class_distribution() -> None:
    y = np.array([1, 1, 1, 1, 0, 0])  # 67% positive
    p = np.full_like(y, 0.5, dtype=float)
    m = compute_metrics(y, p)
    assert m.positive_rate == pytest.approx(4 / 6)
    assert m.baseline_always_positive_accuracy == pytest.approx(4 / 6)
    assert m.baseline_majority_accuracy == pytest.approx(4 / 6)


def test_single_class_emits_nan_for_auc_safely() -> None:
    y = np.array([1, 1, 1, 1])
    p = np.array([0.6, 0.7, 0.8, 0.9])
    m = compute_metrics(y, p)
    assert np.isnan(m.roc_auc)
    assert np.isnan(m.average_precision)
    # Non-AUC metrics still defined.
    assert m.accuracy == pytest.approx(1.0)


def test_metrics_for_mlflow_flattens_with_prefix() -> None:
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    m = compute_metrics(y, p)
    flat = metrics_for_mlflow(m, prefix="val_")
    assert all(k.startswith("val_") for k in flat)
    assert "val_roc_auc" in flat


def test_format_metrics_is_single_line() -> None:
    y = np.array([0, 1, 1])
    p = np.array([0.3, 0.6, 0.8])
    m = compute_metrics(y, p)
    s = format_metrics(m)
    assert "\n" not in s
    assert "AUC=" in s
    assert "Acc=" in s
