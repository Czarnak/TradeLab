from __future__ import annotations

import pandas as pd

from trade_lab.risk_management.base import BaseTakeProfit


class FixedTP(BaseTakeProfit):
    def __init__(self, base_points: float):
        self.base_points = base_points

    def compute(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        return entry_price + direction * self.base_points


class SignalStrengthTP(BaseTakeProfit):
    def __init__(self, base_points: float):
        self.base_points = base_points

    def compute(
        self,
        entry_price: float,
        direction: int,
        signal_strength: float,
        bar: pd.Series,
    ) -> float:
        return entry_price + direction * self.base_points * abs(signal_strength)
