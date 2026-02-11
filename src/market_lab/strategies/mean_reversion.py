"""Mean Reversion strategy using z-score / Bollinger Bands.

Enter long when price is below lower band (oversold),
enter short when price is above upper band (overbought).
"""

from __future__ import annotations

import pandas as pd

from market_lab.strategies.base import Strategy, StrategySignals, register_strategy


class MeanReversion(Strategy):
    """Bollinger-Band / z-score mean-reversion strategy."""

    @property
    def name(self) -> str:
        return "Mean Reversion"

    def parameters_schema(self) -> dict:
        return {
            "lookback": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 200,
                "step": 1,
                "description": "Rolling window for mean and std",
            },
            "entry_z": {
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 4.0,
                "step": 0.1,
                "description": "Z-score threshold for entry",
            },
            "exit_z": {
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 2.0,
                "step": 0.1,
                "description": "Z-score threshold for exit (revert to 0)",
            },
            "allow_short": {
                "type": "bool",
                "default": True,
                "description": "Allow short positions",
            },
        }

    def run(self, bars: pd.DataFrame, params: dict) -> StrategySignals:
        lookback = params.get("lookback", 20)
        entry_z = params.get("entry_z", 2.0)
        exit_z = params.get("exit_z", 0.0)
        allow_short = params.get("allow_short", True)

        close = bars["close"]
        rolling_mean = close.rolling(lookback).mean()
        rolling_std = close.rolling(lookback).std()

        z_score = (close - rolling_mean) / (rolling_std + 1e-12)

        signal = pd.Series(0, index=bars.index, dtype=int)

        # Vectorised state: oversold → long, overbought → short
        position = 0
        signals = []
        for z in z_score:
            if pd.isna(z):
                signals.append(0)
                continue
            if position == 0:
                if z < -entry_z:
                    position = 1  # long — price below lower band
                elif z > entry_z and allow_short:
                    position = -1  # short — price above upper band
            elif position == 1:
                if z > -exit_z:
                    position = 0
            elif position == -1:
                if z < exit_z:
                    position = 0
            signals.append(position)

        signal = pd.Series(signals, index=bars.index, dtype=int)

        metadata = {
            "z_score": z_score,
            "rolling_mean": rolling_mean,
            "upper_band": rolling_mean + entry_z * rolling_std,
            "lower_band": rolling_mean - entry_z * rolling_std,
        }

        return StrategySignals(signal=signal, metadata=metadata)


# Auto-register
register_strategy(MeanReversion())
