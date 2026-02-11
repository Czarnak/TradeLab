"""Tests for the dataset registry."""

from __future__ import annotations

import pandas as pd
import pytest

from market_lab.data.registry import DatasetRegistry


@pytest.fixture
def registry():
    return DatasetRegistry()


@pytest.fixture
def sample_df():
    idx = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": range(10), "high": range(10), "low": range(10),
         "close": range(10), "volume": range(10)},
        index=idx,
    )


class TestRegistry:
    def test_register_and_get(self, registry, sample_df):
        ds_id = registry.register(sample_df, "test_ds", "bars", "csv")
        assert ds_id in registry
        entry = registry.get(ds_id)
        assert entry.name == "test_ds"
        assert entry.rows == 10

    def test_remove(self, registry, sample_df):
        ds_id = registry.register(sample_df, "to_remove")
        registry.remove(ds_id)
        assert ds_id not in registry

    def test_list_all(self, registry, sample_df):
        registry.register(sample_df, "ds1")
        registry.register(sample_df, "ds2")
        summaries = registry.list_all()
        assert len(summaries) == 2
        names = {s["name"] for s in summaries}
        assert names == {"ds1", "ds2"}

    def test_get_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_len(self, registry, sample_df):
        assert len(registry) == 0
        registry.register(sample_df, "one")
        assert len(registry) == 1
