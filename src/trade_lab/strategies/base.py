from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from trade_lab.sizing.base import BasePositionSizer


class BaseStrategy(ABC):
    """
    Input:  Indicators (already computed), position sizing config.
    Output: DataFrame with 'signal_strength' column in [-1.0, 1.0].

    Backtester interprets signal_strength using:
        signal_strength > entry_threshold → open/maintain long
        signal_strength < -entry_threshold → open/maintain short
        |signal_strength| < exit_threshold → close position
    """

    def __init__(
        self,
        allow_long: bool = True,
        allow_short: bool = True,
        position_sizer: Optional[BasePositionSizer] = None,
        entry_threshold: float = 0.5,
        exit_threshold: float = 0.1,
    ):
        self.allow_long = allow_long
        self.allow_short = allow_short
        self.position_sizer = position_sizer
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full pipeline and return df with the 'signal_strength' column.
        Responsible for calling indicator.compute() internally.
        """
        ...
