"""Abstract strategy interface and signal container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategySignals:
    """Output of a strategy run.

    Attributes
    ----------
    signal : pd.Series
        Integer series aligned to bars index:
        ``+1`` = long entry/hold, ``-1`` = short entry/hold, ``0`` = flat/exit.
    metadata : dict
        Any extra data the strategy wishes to expose (e.g. indicator values).
    """

    signal: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Base class all strategies must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
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
    def run(self, bars: pd.DataFrame, params: dict) -> StrategySignals:
        """Execute the strategy on OHLCV bars with given parameters.

        Parameters
        ----------
        bars : pd.DataFrame
            Canonical OHLCV DataFrame.
        params : dict
            Parameter values (matching ``parameters_schema`` keys).

        Returns
        -------
        StrategySignals
        """
        ...

    def default_params(self) -> dict:
        """Return default parameter values from the schema."""
        schema = self.parameters_schema()
        return {k: v.get("default") for k, v in schema.items()}


# ---------------------------------------------------------------------------
# Strategy registry (discover all built-in strategies)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Strategy] = {}


def register_strategy(strategy: Strategy) -> None:
    """Register a strategy instance in the global registry."""
    _REGISTRY[strategy.name] = strategy


def get_strategy(name: str) -> Strategy:
    """Retrieve a registered strategy by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Strategy '{name}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    return list(_REGISTRY.keys())


def all_strategies() -> dict[str, Strategy]:
    """Return the full registry dict."""
    return dict(_REGISTRY)
