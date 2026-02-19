"""Base signals definitions, used then to calculate Signal values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

class Signal(ABC):
    """Base class all signals must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable signal name."""
        ...

    @abstractmethod
    def run(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Calculate the signal values on OHLCV bars.

        Parameters
        ----------
        bars : pd.DataFrame
            Canonical OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            Signal values aligned to bars index.
        """
        ...

    @abstractmethod
    def plot(self, signal_values: pd.Series) -> Any:
        """Return a plotly figure for the signal."""
        ...


# ---------------------------------------------------------------------------
# Signals registry (discover all built-in Signals)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Signal] = {}


def register_Signal(signal: Signal) -> None:
    """Register a Signal instance in the global registry."""
    _REGISTRY[signal.name] = signal


def get_Signal(name: str) -> Signal:
    """Retrieve a registered Signal by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Signal '{name}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_Signals() -> list[str]:
    """Return names of all registered Signals."""
    return list(_REGISTRY.keys())


def all_Signals() -> dict[str, Signal]:
    """Return the full registry dict."""
    return dict(_REGISTRY)


