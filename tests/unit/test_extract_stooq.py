"""Unit tests for the pure helpers in `src.etl.extract.from_stooq`."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.db import AssetType
from src.etl.extract.from_stooq import (
    asset_type_from_path,
    read_stooq_file,
    sector_from_symbol,
    symbol_from_path,
)

pytestmark = pytest.mark.unit


# ── symbol_from_path ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "filename,expected",
    [
        ("7203.jp.txt", "7203.JP"),
        ("1305.jp.txt", "1305.JP"),
        ("^tpx.txt", "^TPX"),
        ("161050015.jp.txt", "161050015.JP"),
        ("Foo.TXT", "FOO"),  # uppercase + strip .txt regardless of case
    ],
)
def test_symbol_from_path(filename: str, expected: str) -> None:
    assert symbol_from_path(Path(filename)) == expected


# ── asset_type_from_path ─────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "rel_path,expected",
    [
        ("jp/tse stocks/1/7203.jp.txt", AssetType.STOCK),
        ("jp/tse stocks/2/9984.jp.txt", AssetType.STOCK),
        ("jp/tse etfs/1305.jp.txt", AssetType.ETF),
        ("jp/tse indices/^tpx.txt", AssetType.INDEX),
        ("jp/tse futures/161050015.jp.txt", AssetType.FUTURE),
        ("jp/tse options/1/abc.jp.txt", AssetType.OPTION),
        ("jp/tse treasury bonds/00010054.jp.txt", AssetType.BOND),
        ("jp/unknown folder/abc.txt", None),
    ],
)
def test_asset_type_from_path(rel_path: str, expected: AssetType | None, tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    full_path = raw_root / rel_path
    assert asset_type_from_path(full_path, raw_root) == expected


def test_asset_type_outside_raw_root_returns_none(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    outside = tmp_path / "elsewhere" / "tse stocks" / "7203.jp.txt"
    assert asset_type_from_path(outside, raw_root) is None


# ── sector_from_symbol ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "symbol,asset_type,expected",
    [
        ("7203.JP", AssetType.STOCK, "7000"),
        ("9984.JP", AssetType.STOCK, "9000"),
        ("1301.JP", AssetType.STOCK, "1000"),
        ("130a.JP", AssetType.STOCK, "1000"),  # 4-digit code with letter suffix
        ("^TPX", AssetType.INDEX, None),  # non-stocks don't get a sector
        ("7203.JP", AssetType.ETF, None),  # ETF with numeric code → no sector
        ("invalid", AssetType.STOCK, None),  # not a 4-digit code
    ],
)
def test_sector_from_symbol(symbol: str, asset_type: AssetType, expected: str | None) -> None:
    assert sector_from_symbol(symbol, asset_type) == expected


# ── read_stooq_file ──────────────────────────────────────────────────────────
def _write_sample_file(path: Path, rows: list[str]) -> None:
    header = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_read_stooq_file_parses_dates_and_columns(tmp_path: Path) -> None:
    f = tmp_path / "7203.jp.txt"
    _write_sample_file(
        f,
        [
            "7203.JP,D,20240102,000000,2500.0,2520.0,2495.0,2515.0,1000000,0",
            "7203.JP,D,20240103,000000,2515.0,2530.0,2510.0,2528.0,1200000,0",
        ],
    )

    df = read_stooq_file(f)
    assert df is not None
    assert df.height == 2
    assert set(df.columns) == {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    }
    # Dates parsed correctly
    dates = df["date"].to_list()
    assert dates[0].isoformat() == "2024-01-02"
    assert dates[1].isoformat() == "2024-01-03"
    # Numeric types
    assert df["open"].dtype == pl.Float64


def test_read_stooq_file_returns_none_on_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.jp.txt"
    f.write_text("", encoding="utf-8")
    assert read_stooq_file(f) is None
