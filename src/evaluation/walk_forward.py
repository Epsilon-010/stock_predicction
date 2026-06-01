from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Split:
    """One walk-forward fold."""

    fold: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date

    @property
    def gap_days(self) -> int:
        return (self.val_start - self.train_end).days


def _to_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def walk_forward_splits(
    train_start: date | str,
    train_end: date | str,
    val_start: date | str,
    val_end: date | str,
    n_splits: int = 1,
    embargo_days: int = 5,
    expanding: bool = True,
) -> Iterator[Split]:
    """Yield walk-forward splits between `train_start` and `val_end`.

    If `n_splits == 1`, yields a single split with the exact bounds given.
    Otherwise the validation period is divided into `n_splits` contiguous
    folds; the training window either expands (default, includes everything
    before val_start) or rolls (drops the oldest equivalent of each fold).

    Args:
        train_start, train_end, val_start, val_end: ISO dates (`"YYYY-MM-DD"`)
            or `datetime.date` objects.
        n_splits:     Number of folds across the validation period.
        embargo_days: Forced gap between train_end and val_start (also between
            consecutive folds). Should be ≥ horizon_days of the labels.
        expanding:    Expanding-window vs rolling-window training set.
    """
    train_start_d = _to_date(train_start)
    train_end_d = _to_date(train_end)
    val_start_d = _to_date(val_start)
    val_end_d = _to_date(val_end)

    if val_start_d <= train_end_d:
        raise ValueError("val_start must be after train_end")
    if (val_start_d - train_end_d).days < embargo_days:
        raise ValueError(
            f"Gap between train_end and val_start ({(val_start_d - train_end_d).days}d) "
            f"is smaller than embargo_days ({embargo_days})"
        )

    if n_splits <= 1:
        yield Split(0, train_start_d, train_end_d, val_start_d, val_end_d)
        return

    total_val_days = (val_end_d - val_start_d).days
    fold_size = max(1, total_val_days // n_splits)

    for i in range(n_splits):
        fold_val_start = val_start_d + timedelta(days=i * fold_size)
        fold_val_end = (
            val_end_d if i == n_splits - 1 else fold_val_start + timedelta(days=fold_size)
        )
        fold_train_end = fold_val_start - timedelta(days=embargo_days)
        fold_train_start = (
            train_start_d
            if expanding
            else fold_train_end - timedelta(days=(train_end_d - train_start_d).days)
        )
        yield Split(i, fold_train_start, fold_train_end, fold_val_start, fold_val_end)
