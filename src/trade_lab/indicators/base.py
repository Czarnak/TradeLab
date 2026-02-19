"""Abstract indicator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

class Indicator(ABC):
    """Base class all indicators must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable indicator name."""
        ...

    @abstractmethod
    def parameters_schema(self) -> dict:
        """Return a JSON-schema-like dict describing tunable parameters.

        Each key is a parameter name. Value is a dict with:
        - ``type``: ``"int"`` | ``"float"`` | ``"bool"`` | ``"enum"``
        - ``default``: default value
        - ``min``, ``max``, ``step`` (optional, for numeric)
        - ``choices`` (for enum)
        - ``description`` (optional)
        """
        ...

    @abstractmethod
    def run(self, bars: pd.DataFrame, params: dict) -> pd.Series:
        """Calculate the indicator values on OHLCV bars with given parameters.

        Parameters
        ----------
        bars : pd.DataFrame
            Canonical OHLCV DataFrame.
        params : dict
            Parameter values (matching ``parameters_schema`` keys).

        Returns
        -------
        pd.Series
            Indicator values aligned to bars index.
        """
        ...

    @abstractmethod
    def plot(self, indicator_values: pd.Series, params: dict) -> Any:
        """Return a plotly figure for the indicator values."""
        ...


# ---------------------------------------------------------------------------
# Indicators registry (discover all built-in indicators)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Indicator] = {}


def register_indicator(indicator: Indicator) -> None:
    """Register a indicator instance in the global registry."""
    _REGISTRY[indicator.name] = indicator


def get_indicator(name: str) -> Indicator:
    """Retrieve a registered indicator by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Indicator '{name}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_indicators() -> list[str]:
    """Return names of all registered indicators."""
    return list(_REGISTRY.keys())


def all_indicators() -> dict[str, Indicator]:
    """Return the full registry dict."""
    return dict(_REGISTRY)