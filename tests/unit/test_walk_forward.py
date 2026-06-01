"""Unit tests for walk-forward CV splits."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.evaluation.walk_forward import Split, walk_forward_splits

pytestmark = pytest.mark.unit


def test_single_split_returns_one_fold() -> None:
    splits = list(
        walk_forward_splits(
            train_start="2020-01-01",
            train_end="2022-12-31",
            val_start="2023-01-10",
            val_end="2023-12-31",
            n_splits=1,
            embargo_days=5,
        )
    )
    assert len(splits) == 1
    s = splits[0]
    assert isinstance(s, Split)
    assert s.train_start == date(2020, 1, 1)
    assert s.val_end == date(2023, 12, 31)


def test_embargo_is_enforced() -> None:
    with pytest.raises(ValueError, match="embargo"):
        list(
            walk_forward_splits(
                train_start="2020-01-01",
                train_end="2023-01-01",
                val_start="2023-01-02",  # only 1 day after train_end
                val_end="2023-12-31",
                embargo_days=5,
            )
        )


def test_val_must_be_after_train() -> None:
    with pytest.raises(ValueError, match="after train_end"):
        list(
            walk_forward_splits(
                train_start="2020-01-01",
                train_end="2023-06-01",
                val_start="2023-06-01",
                val_end="2023-12-31",
            )
        )


def test_n_splits_creates_contiguous_folds() -> None:
    splits = list(
        walk_forward_splits(
            train_start="2020-01-01",
            train_end="2022-12-31",
            val_start="2023-01-10",
            val_end="2023-12-31",
            n_splits=3,
            embargo_days=5,
        )
    )
    assert len(splits) == 3
    # All folds use the same training start (expanding window).
    assert all(s.train_start == date(2020, 1, 1) for s in splits)
    # Val periods cover the original range without gaps > fold_size.
    fold_size = (date(2023, 12, 31) - date(2023, 1, 10)).days // 3
    for i, s in enumerate(splits[:-1]):
        next_start = splits[i + 1].val_start
        assert (next_start - s.val_start).days == fold_size
    # Last fold extends to val_end.
    assert splits[-1].val_end == date(2023, 12, 31)


def test_rolling_window_drops_old_training_data() -> None:
    train_span_days = (date(2022, 12, 31) - date(2020, 1, 1)).days
    splits = list(
        walk_forward_splits(
            train_start="2020-01-01",
            train_end="2022-12-31",
            val_start="2023-01-10",
            val_end="2023-12-31",
            n_splits=3,
            embargo_days=5,
            expanding=False,
        )
    )
    # In rolling mode every fold has the *same* train span; expanding would
    # give larger and larger train spans.
    for s in splits:
        actual_span = (s.train_end - s.train_start).days
        assert actual_span == train_span_days


def test_split_gap_property() -> None:
    s = Split(0, date(2020, 1, 1), date(2022, 1, 1), date(2022, 1, 8), date(2023, 1, 1))
    assert s.gap_days == timedelta(days=7).days
