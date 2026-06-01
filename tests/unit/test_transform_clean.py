"""Unit tests for `src.etl.transform.clean.clean_ohlcv`."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.etl.transform.clean import clean_ohlcv

pytestmark = pytest.mark.unit


def _make_df(rows: list[dict]) -> pl.DataFrame:
    """Build a DataFrame mirroring the columns the extractor produces."""
    return pl.DataFrame(
        rows,
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "open_interest": pl.Int64,
        },
    )


def test_clean_returns_empty_for_empty_input() -> None:
    df = _make_df([])
    assert clean_ohlcv(df, "X").is_empty()


def test_clean_drops_null_prices() -> None:
    df = _make_df(
        [
            {
                "ticker": "X",
                "date": date(2024, 1, 1),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "open_interest": 0,
            },
            {
                "ticker": "X",
                "date": date(2024, 1, 2),
                "open": None,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "open_interest": 0,
            },
        ]
    )
    out = clean_ohlcv(df, "X")
    assert out.height == 1
    assert out["date"][0] == date(2024, 1, 1)


def test_clean_drops_non_positive_prices() -> None:
    df = _make_df(
        [
            {
                "ticker": "X",
                "date": date(2024, 1, 1),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "open_interest": 0,
            },
            {
                "ticker": "X",
                "date": date(2024, 1, 2),
                "open": 0.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "open_interest": 0,
            },
            {
                "ticker": "X",
                "date": date(2024, 1, 3),
                "open": -1.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "open_interest": 0,
            },
        ]
    )
    assert clean_ohlcv(df, "X").height == 1


def test_clean_drops_inverted_high_low() -> None:
    df = _make_df(
        [
            {
                "ticker": "X",
                "date": date(2024, 1, 1),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "open_interest": 0,
            },
            {
                "ticker": "X",
                "date": date(2024, 1, 2),
                "open": 10.0,
                "high": 8.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "open_interest": 0,
            },
        ]
    )
    assert clean_ohlcv(df, "X").height == 1


def test_clean_dedupes_dates_keeping_first() -> None:
    df = _make_df(
        [
            {
                "ticker": "X",
                "date": date(2024, 1, 1),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "open_interest": 0,
            },
            {
                "ticker": "X",
                "date": date(2024, 1, 1),
                "open": 99.0,
                "high": 99.0,
                "low": 99.0,
                "close": 99.0,
                "volume": 999,
                "open_interest": 0,
            },
        ]
    )
    out = clean_ohlcv(df, "X")
    assert out.height == 1
    assert out["open"][0] == 10.0


def test_clean_sorts_by_date() -> None:
    df = _make_df(
        [
            {
                "ticker": "X",
                "date": date(2024, 1, 3),
                "open": 30.0,
                "high": 30.0,
                "low": 30.0,
                "close": 30.0,
                "volume": 0,
                "open_interest": 0,
            },
            {
                "ticker": "X",
                "date": date(2024, 1, 1),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 0,
                "open_interest": 0,
            },
            {
                "ticker": "X",
                "date": date(2024, 1, 2),
                "open": 20.0,
                "high": 20.0,
                "low": 20.0,
                "close": 20.0,
                "volume": 0,
                "open_interest": 0,
            },
        ]
    )
    out = clean_ohlcv(df, "X")
    dates = out["date"].to_list()
    assert dates == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


def test_clean_fills_null_volume_with_zero() -> None:
    df = _make_df(
        [
            {
                "ticker": "X",
                "date": date(2024, 1, 1),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": None,
                "open_interest": 0,
            },
        ]
    )
    out = clean_ohlcv(df, "X")
    assert out["volume"][0] == 0


def test_clean_preserves_null_open_interest() -> None:
    df = _make_df(
        [
            {
                "ticker": "X",
                "date": date(2024, 1, 1),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "open_interest": None,
            },
        ]
    )
    out = clean_ohlcv(df, "X")
    assert out["open_interest"][0] is None
