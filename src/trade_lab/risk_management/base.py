from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseTakeProfit(ABC):
    @abstractmethod
    def compute(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float | None:
        """Return the absolute take-profit price for the new position."""


class BaseStopLoss(ABC):
    @abstractmethod
    def compute(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float | None:
        """Return the absolute stop-loss price for the new position."""


class BaseTrailingStop(ABC):
    def __init__(self, step_points: float = 10.0):
        self.step_points = step_points

    @abstractmethod
    def compute_initial(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float | None:
        """Return the initial trailing-stop price when the trade is opened."""

    @abstractmethod
    def update(
        self,
        current_stop: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        """Return the updated trailing-stop price for the current bar."""

    def _apply_step_guard(
        self,
        current_stop: float,
        new_stop: float,
        direction: int,
    ) -> float:
        improvement = direction * (new_stop - current_stop)
        if improvement < self.step_points:
            return current_stop
        return new_stop
