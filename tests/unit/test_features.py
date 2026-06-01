"""Unit tests for the feature engineering primitives.

These tests focus on:
  * shape preservation (no rows dropped, expected columns appended)
  * known numerical results on tiny hand-crafted inputs
  * absence of look-ahead leakage in labels
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.features.calendar import add_calendar_features
from src.features.cross_sectional import add_cross_sectional_ranks
from src.features.labels import add_direction_label, add_forward_returns
from src.features.lag import add_lagged_columns
from src.features.technical import (
    add_bollinger_bands,
    add_macd,
    add_obv,
    add_returns,
    add_rsi,
)

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_ohlcv(n: int = 60, start_price: float = 100.0) -> pl.DataFrame:
    """Build a deterministic synthetic price series for tests."""
    start = date(2024, 1, 1)
    prices = [start_price + (i * 0.5) for i in range(n)]  # straight uptrend
    return pl.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(n)],
            "open": prices,
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": [1_000_000 + i * 1000 for i in range(n)],
            "open_interest": [0] * n,
        },
        schema={
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "open_interest": pl.Int64,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators
# ─────────────────────────────────────────────────────────────────────────────
def test_add_returns_creates_expected_columns() -> None:
    df = _synthetic_ohlcv(30)
    out = add_returns(df, periods=(1, 5))
    assert "returns_1d" in out.columns
    assert "returns_5d" in out.columns
    assert "log_returns_1d" in out.columns
    # First N rows should be null (no prior data).
    assert out["returns_1d"][0] is None
    assert out["returns_5d"][4] is None
    # Straight uptrend → all subsequent returns are positive.
    assert all(r > 0 for r in out["returns_1d"][1:].to_list())


def test_add_rsi_is_bounded() -> None:
    df = _synthetic_ohlcv(50)
    out = add_rsi(df, window=14)
    assert "rsi_14" in out.columns
    valid = out["rsi_14"].drop_nulls()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_add_macd_creates_three_columns() -> None:
    df = _synthetic_ohlcv(60)
    out = add_macd(df)
    for col in ("macd", "macd_signal", "macd_hist"):
        assert col in out.columns


def test_add_bollinger_bands_relations() -> None:
    df = _synthetic_ohlcv(40)
    out = add_bollinger_bands(df, window=20)
    valid = out.drop_nulls(subset=["bb_upper_20", "bb_lower_20", "bb_mid_20"])
    # upper >= mid >= lower for every row
    assert (valid["bb_upper_20"] >= valid["bb_mid_20"]).all()
    assert (valid["bb_mid_20"] >= valid["bb_lower_20"]).all()


def test_add_obv_is_monotonic_on_uptrend() -> None:
    df = _synthetic_ohlcv(20)
    out = add_obv(df)
    obv = out["obv"].drop_nulls().to_list()
    # Pure uptrend → every close > previous → OBV strictly increasing.
    assert obv == sorted(obv)


# ─────────────────────────────────────────────────────────────────────────────
# Calendar
# ─────────────────────────────────────────────────────────────────────────────
def test_add_calendar_features() -> None:
    df = _synthetic_ohlcv(10)
    out = add_calendar_features(df)
    for col in ("day_of_week", "month", "quarter", "year", "is_month_end"):
        assert col in out.columns
    assert out["month"][0] == 1
    assert out["quarter"][0] == 1
    assert out["year"][0] == 2024


# ─────────────────────────────────────────────────────────────────────────────
# Lag
# ─────────────────────────────────────────────────────────────────────────────
def test_add_lagged_columns_shifts_correctly() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = add_lagged_columns(df, columns=["x"], lags=(1, 2))
    assert out["x_lag_1"].to_list() == [None, 1.0, 2.0, 3.0, 4.0]
    assert out["x_lag_2"].to_list() == [None, None, 1.0, 2.0, 3.0]


def test_add_lagged_columns_skips_missing_columns() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0]})
    out = add_lagged_columns(df, columns=["y"], lags=(1,))
    assert out.columns == ["x"]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional rank
# ─────────────────────────────────────────────────────────────────────────────
def test_cross_sectional_rank_per_date_in_unit_interval() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "symbol": ["A", "B", "C"],
            "returns_5d": [0.10, -0.05, 0.02],
        }
    )
    out = add_cross_sectional_ranks(df)
    assert "returns_5d_rank" in out.columns
    ranks = out["returns_5d_rank"].to_list()
    assert min(ranks) == pytest.approx(0.0)
    assert max(ranks) == pytest.approx(1.0)
    # Best return ("A", 0.10) should rank highest because descending=True.
    a_rank = out.filter(pl.col("symbol") == "A")["returns_5d_rank"][0]
    assert a_rank == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Labels — critical: no look-ahead leakage in features themselves
# ─────────────────────────────────────────────────────────────────────────────
def test_forward_returns_use_future_close() -> None:
    df = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]})
    out = add_forward_returns(df, horizon_days=2)
    # First row should look at close[2] = 102 → (102/100 - 1) = 0.02
    assert out["label_return_2d"][0] == pytest.approx(0.02)
    # Last `horizon_days` rows must be null — no future data.
    assert out["label_return_2d"][-1] is None
    assert out["label_return_2d"][-2] is None


def test_direction_label_above_threshold() -> None:
    df = pl.DataFrame({"close": [100.0, 100.0, 110.0, 95.0, 100.0]})
    out = add_direction_label(df, horizon_days=2, threshold=0.0)
    # row 0: close[2]/close[0] - 1 = 0.10 → 1
    # row 1: close[3]/close[1] - 1 = -0.05 → 0
    # row 2: close[4]/close[2] - 1 = -0.09 → 0
    # rows 3, 4: null
    assert out["label_direction_2d"][0] == 1
    assert out["label_direction_2d"][1] == 0
    assert out["label_direction_2d"][2] == 0
    assert out["label_direction_2d"][3] is None
    assert out["label_direction_2d"][4] is None
