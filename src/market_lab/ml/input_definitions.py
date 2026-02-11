"""ML input feature definitions and registry.

Each feature builder takes an OHLCV DataFrame and a lag parameter,
returning a DataFrame of computed features. Builders are registered
by string name for GUI/config lookup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd

from market_lab.utils.logging import get_logger

log = get_logger("ml.input_definitions")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class FeatureBuilder(ABC):
    """Base class for all feature builders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier string (e.g. ``"close_lags"``)."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for GUI display."""
        ...

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    def build_features(self, bars: pd.DataFrame, n_lags: int) -> pd.DataFrame:
        """Build feature columns from OHLCV bars.

        Parameters
        ----------
        bars : pd.DataFrame
            Canonical OHLCV bars.
        n_lags : int
            Number of look-back bars.

        Returns
        -------
        pd.DataFrame
            Columns named ``<name>_lag_<i>`` for i in 1..n_lags.
            Index aligned with input bars. Rows with NaN from lagging
            are kept (caller is responsible for dropping).
        """
        ...

    def output_dim(self, n_lags: int) -> int:
        """Number of feature columns produced for a given n_lags."""
        return n_lags


# ---------------------------------------------------------------------------
# Built-in feature builders
# ---------------------------------------------------------------------------

class CloseLags(FeatureBuilder):
    @property
    def name(self) -> str:
        return "close_lags"

    @property
    def display_name(self) -> str:
        return "Close last N bars"

    @property
    def description(self) -> str:
        return "Raw close prices for the last N bars."

    def build_features(self, bars: pd.DataFrame, n_lags: int) -> pd.DataFrame:
        close = bars["close"]
        cols = {}
        for i in range(1, n_lags + 1):
            cols[f"close_lag_{i}"] = close.shift(i)
        return pd.DataFrame(cols, index=bars.index)


class OpenLags(FeatureBuilder):
    @property
    def name(self) -> str:
        return "open_lags"

    @property
    def display_name(self) -> str:
        return "Open last N bars"

    @property
    def description(self) -> str:
        return "Raw open prices for the last N bars."

    def build_features(self, bars: pd.DataFrame, n_lags: int) -> pd.DataFrame:
        series = bars["open"]
        cols = {}
        for i in range(1, n_lags + 1):
            cols[f"open_lag_{i}"] = series.shift(i)
        return pd.DataFrame(cols, index=bars.index)


class ReturnLags(FeatureBuilder):
    @property
    def name(self) -> str:
        return "return_lags"

    @property
    def display_name(self) -> str:
        return "Log return last N bars"

    @property
    def description(self) -> str:
        return "Log returns (ln(close/close_prev)) for the last N bars."

    def build_features(self, bars: pd.DataFrame, n_lags: int) -> pd.DataFrame:
        log_ret = np.log(bars["close"] / bars["close"].shift(1))
        cols = {}
        for i in range(1, n_lags + 1):
            cols[f"return_lag_{i}"] = log_ret.shift(i)
        return pd.DataFrame(cols, index=bars.index)


class HighLowRangeLags(FeatureBuilder):
    @property
    def name(self) -> str:
        return "hl_range_lags"

    @property
    def display_name(self) -> str:
        return "High-Low range last N bars"

    @property
    def description(self) -> str:
        return "High minus Low for the last N bars (volatility proxy)."

    def build_features(self, bars: pd.DataFrame, n_lags: int) -> pd.DataFrame:
        hl_range = bars["high"] - bars["low"]
        cols = {}
        for i in range(1, n_lags + 1):
            cols[f"hl_range_lag_{i}"] = hl_range.shift(i)
        return pd.DataFrame(cols, index=bars.index)


class VolumeLags(FeatureBuilder):
    @property
    def name(self) -> str:
        return "volume_lags"

    @property
    def display_name(self) -> str:
        return "Volume last N bars"

    @property
    def description(self) -> str:
        return "Trading volume for the last N bars."

    def build_features(self, bars: pd.DataFrame, n_lags: int) -> pd.DataFrame:
        vol = bars["volume"]
        cols = {}
        for i in range(1, n_lags + 1):
            cols[f"volume_lag_{i}"] = vol.shift(i)
        return pd.DataFrame(cols, index=bars.index)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FEATURE_REGISTRY: dict[str, FeatureBuilder] = {}


def register_feature(builder: FeatureBuilder) -> None:
    """Register a feature builder in the global registry."""
    _FEATURE_REGISTRY[builder.name] = builder


def get_feature_builder(name: str) -> FeatureBuilder:
    """Retrieve a feature builder by name."""
    if name not in _FEATURE_REGISTRY:
        raise KeyError(
            f"Feature builder '{name}' not found. "
            f"Available: {list(_FEATURE_REGISTRY.keys())}"
        )
    return _FEATURE_REGISTRY[name]


def list_feature_builders() -> list[str]:
    """Return names of all registered feature builders."""
    return list(_FEATURE_REGISTRY.keys())


def all_feature_builders() -> dict[str, FeatureBuilder]:
    """Return the full registry dict."""
    return dict(_FEATURE_REGISTRY)


# Auto-register built-ins
for _cls in (CloseLags, OpenLags, ReturnLags, HighLowRangeLags, VolumeLags):
    register_feature(_cls())
