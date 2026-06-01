"""Unit tests for the long-only top-K backtest."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.evaluation.backtest import backtest_top_k

pytestmark = pytest.mark.unit


def _make_preds(n_days: int = 10, n_per_day: int = 5) -> pl.DataFrame:
    """Synthetic prediction frame: top-ranked symbol has highest realised return."""
    start = date(2024, 1, 1)
    rows = []
    for d_off in range(n_days):
        d = start + timedelta(days=d_off)
        for rank in range(n_per_day):
            rows.append(
                {
                    "date": d,
                    "symbol": f"SYM{rank}",
                    "predicted_probability": (n_per_day - rank) / n_per_day,
                    "label_return_5d": (n_per_day - rank) * 0.01,  # 5%, 4%, …
                }
            )
    return pl.DataFrame(rows)


def test_empty_input_returns_zeros() -> None:
    empty = pl.DataFrame(
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "predicted_probability": pl.Float64,
            "label_return_5d": pl.Float64,
        }
    )
    r = backtest_top_k(empty, "label_return_5d")
    assert r.cumulative_return == 0.0
    assert r.n_trades == 0


def test_top_k_picks_highest_probabilities() -> None:
    df = _make_preds(n_days=20, n_per_day=10)
    result = backtest_top_k(df, "label_return_5d", top_k=3, transaction_cost_bps=0.0)
    # Picking top-3 (5%, 4%, 3% returns) averages to 4% per rebalance.
    # 20 rebalances → equity ~ 1.04^20.
    assert result.cumulative_return > 1.0
    assert result.win_rate == pytest.approx(1.0)
    assert result.n_trades == 20 * 3


def test_transaction_costs_reduce_return() -> None:
    df = _make_preds()
    r_free = backtest_top_k(df, "label_return_5d", top_k=2, transaction_cost_bps=0.0)
    r_costly = backtest_top_k(df, "label_return_5d", top_k=2, transaction_cost_bps=50.0)
    assert r_costly.cumulative_return < r_free.cumulative_return


def test_sharpe_is_finite_for_nonzero_vol() -> None:
    df = _make_preds(n_days=30)
    r = backtest_top_k(df, "label_return_5d", top_k=3)
    assert r.annualised_volatility >= 0.0
    # Constant returns → vol = 0 → sharpe defined as 0 by convention.
    assert r.sharpe_ratio == 0.0
