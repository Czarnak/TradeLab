"""Tests for CSV / Parquet loaders and schema validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from market_lab.data.loaders import load_csv


@pytest.fixture
def tmp_bar_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "bars.csv"
    csv.write_text(textwrap.dedent("""\
        Date,Open,High,Low,Close,Volume
        2023-01-01,100,105,99,103,1000000
        2023-01-02,103,108,102,107,1200000
        2023-01-03,107,110,105,109,900000
    """))
    return csv


@pytest.fixture
def tmp_bar_csv_aliases(tmp_path: Path) -> Path:
    csv = tmp_path / "bars_alias.csv"
    csv.write_text(textwrap.dedent("""\
        Datetime,o,h,l,c,vol
        2023-06-01 09:30:00,150.0,152.0,149.0,151.0,500000
        2023-06-01 10:30:00,151.0,153.0,150.0,152.5,600000
    """))
    return csv


@pytest.fixture
def tmp_tick_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "ticks.csv"
    csv.write_text(textwrap.dedent("""\
        timestamp,bid,ask,volume
        2023-01-01 09:30:00.100,100.01,100.05,50
        2023-01-01 09:30:00.200,100.02,100.06,30
        2023-01-01 09:30:00.350,100.00,100.04,80
    """))
    return csv


@pytest.fixture
def tmp_tick_csv_no_volume(tmp_path: Path) -> Path:
    csv = tmp_path / "ticks_novol.csv"
    csv.write_text(textwrap.dedent("""\
        time,bid,offer
        2023-01-01 09:30:00,99.5,100.0
        2023-01-01 09:30:01,99.6,100.1
    """))
    return csv


class TestBarLoading:
    def test_basic_load(self, tmp_bar_csv: Path):
        df = load_csv(tmp_bar_csv, schema="bars")
        assert len(df) == 3
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"

    def test_alias_headers(self, tmp_bar_csv_aliases: Path):
        df = load_csv(tmp_bar_csv_aliases, schema="bars")
        assert len(df) == 2
        assert "open" in df.columns
        assert "close" in df.columns

    def test_sorted_index(self, tmp_bar_csv: Path):
        df = load_csv(tmp_bar_csv, schema="bars")
        assert df.index.is_monotonic_increasing

    def test_numeric_columns(self, tmp_bar_csv: Path):
        df = load_csv(tmp_bar_csv, schema="bars")
        for col in ("open", "high", "low", "close", "volume"):
            assert pd.api.types.is_numeric_dtype(df[col])


class TestTickLoading:
    def test_basic_load(self, tmp_tick_csv: Path):
        df = load_csv(tmp_tick_csv, schema="ticks")
        assert len(df) == 3
        assert "bid" in df.columns
        assert "ask" in df.columns
        assert "mid" in df.columns

    def test_mid_calculation(self, tmp_tick_csv: Path):
        df = load_csv(tmp_tick_csv, schema="ticks")
        expected_mid = (df["bid"] + df["ask"]) / 2
        pd.testing.assert_series_equal(df["mid"], expected_mid, check_names=False)

    def test_no_volume_column(self, tmp_tick_csv_no_volume: Path):
        df = load_csv(tmp_tick_csv_no_volume, schema="ticks")
        assert "volume" in df.columns
        assert (df["volume"] == 0.0).all()

    def test_offer_alias(self, tmp_tick_csv_no_volume: Path):
        df = load_csv(tmp_tick_csv_no_volume, schema="ticks")
        assert "ask" in df.columns


class TestEdgeCases:
    def test_empty_csv(self, tmp_path: Path):
        csv = tmp_path / "empty.csv"
        csv.write_text("Date,Open,High,Low,Close,Volume\n")
        with pytest.raises(ValueError, match="empty"):
            load_csv(csv, schema="bars")

    def test_missing_columns(self, tmp_path: Path):
        csv = tmp_path / "bad.csv"
        csv.write_text("Date,Open,High\n2023-01-01,100,105\n")
        with pytest.raises(ValueError, match="Missing required"):
            load_csv(csv, schema="bars")
