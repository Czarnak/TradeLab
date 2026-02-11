"""Moving Average Crossover strategy.

Long when fast MA > slow MA, short when fast MA < slow MA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_lab.strategies.base import Strategy, StrategySignals, register_strategy


class MACrossover(Strategy):
    """Dual moving-average crossover strategy."""

    @property
    def name(self) -> str:
        return "MA Crossover"

    def parameters_schema(self) -> dict:
        return {
            "fast_period": {
                "type": "int",
                "default": 10,
                "min": 2,
                "max": 200,
                "step": 1,
                "description": "Fast moving average period",
            },
            "slow_period": {
                "type": "int",
                "default": 30,
                "min": 5,
                "max": 500,
                "step": 1,
                "description": "Slow moving average period",
            },
            "ma_type": {
                "type": "enum",
                "default": "SMA",
                "choices": ["SMA", "EMA"],
                "description": "Moving average type",
            },
            "allow_short": {
                "type": "bool",
                "default": True,
                "description": "Allow short positions",
            },
        }

    def run(self, bars: pd.DataFrame, params: dict) -> StrategySignals:
        fast_p = params.get("fast_period", 10)
        slow_p = params.get("slow_period", 30)
        ma_type = params.get("ma_type", "SMA")
        allow_short = params.get("allow_short", True)

        close = bars["close"]

        if ma_type == "EMA":
            fast_ma = close.ewm(span=fast_p, adjust=False).mean()
            slow_ma = close.ewm(span=slow_p, adjust=False).mean()
        else:
            fast_ma = close.rolling(fast_p).mean()
            slow_ma = close.rolling(slow_p).mean()

        signal = pd.Series(0, index=bars.index, dtype=int)
        signal[fast_ma > slow_ma] = 1
        if allow_short:
            signal[fast_ma < slow_ma] = -1
        else:
            signal[fast_ma < slow_ma] = 0

        # NaN region: no signal
        warmup = max(fast_p, slow_p)
        signal.iloc[:warmup] = 0

        metadata = {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
        }

        return StrategySignals(signal=signal, metadata=metadata)


# Auto-register
register_strategy(MACrossover())
