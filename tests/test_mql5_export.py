from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trade_lab.indicators.base import BaseIndicator
from trade_lab.indicators.moving_averages import CMA, EMA, SMA
from trade_lab.indicators.oscillators import MACD, RSI
from trade_lab.mql5_export import export_to_mql5
from trade_lab.mql5_export.introspector import StrategyIntrospector
from trade_lab.mql5_export.validators import validate_strategy
from trade_lab.signals.base import BaseSignal
from trade_lab.signals.signals import OHLC
from trade_lab.signals.temporal import CyclicalTemporalSignal
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer
from trade_lab.strategies.standard import StandardStrategy


class _UnsupportedSignal(BaseSignal):
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.get_data(df.copy())
        out[self._raw_output_columns[0]] = 0.0
        return out

    def plot(self, df: pd.DataFrame):
        return None

    @property
    def _raw_output_columns(self) -> list[str]:
        return ["signal__unsupported"]


class _UnsupportedIndicator(BaseIndicator):
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.get_data(df.copy())
        out[self._raw_output_columns[0]] = 0.0
        return out

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.zeros(len(df)), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None):
        return None

    @property
    def _raw_output_columns(self) -> list[str]:
        return ["indicator__unsupported"]


class _UnsupportedSizer:
    pass


def test_validate_strategy_rejects_non_standard_strategy():
    result = validate_strategy(object())

    assert not result.is_valid
    assert result.errors
    assert "StandardStrategy" in result.errors[0]


def test_validate_strategy_reports_unsupported_parts():
    strategy = StandardStrategy(
        indicators=[(_UnsupportedIndicator(_UnsupportedSignal()), 1.0)],
        position_sizer=_UnsupportedSizer(),
    )

    result = validate_strategy(strategy)

    assert not result.is_valid
    assert any("Unsupported indicator" in err for err in result.errors)
    assert any("Unsupported signal" in err for err in result.errors)
    assert any("Unsupported position sizer" in err for err in result.errors)


def test_validate_strategy_emits_macd_and_cma_warning_once():
    strategy = StandardStrategy(
        indicators=[
            (MACD(fast_period=12, slow_period=26), 1.0),
            (MACD(fast_period=6, slow_period=13), 0.5),
            (CMA(period=10), 0.25),
            (CMA(period=20), -0.25),
        ]
    )

    result = validate_strategy(strategy)

    assert result.is_valid
    assert (
        sum("MACD signal strength is unnormalised" in w for w in result.warnings) == 1
    )
    assert sum("CMA (Cumulative Moving Average)" in w for w in result.warnings) == 1


def test_strategy_introspector_builds_names_and_deduplicates_signals():
    strategy = StandardStrategy(
        indicators=[
            (EMA(OHLC(), period=20), 1.0),
            (EMA(OHLC(), period=50), -0.5),
            (RSI(CyclicalTemporalSignal("hour", 24), period=14), 0.25),
        ],
        position_sizer=RiskBasedPositionSizer(max_fraction=0.03, risk_multiplier=2.5),
        entry_threshold=0.4,
        exit_threshold=0.15,
        allow_long=True,
        allow_short=False,
    )

    config = StrategyIntrospector().introspect(strategy)

    assert [ind.var_name for ind in config.indicators] == [
        "ema_fast",
        "ema_slow",
        "rsi",
    ]
    assert [ind.input_prefix for ind in config.indicators] == [
        "EMA_Fast",
        "EMA_Slow",
        "RSI",
    ]
    assert [ind.function_name for ind in config.indicators] == [
        "EMAFast",
        "EMASlow",
        "RSI",
    ]
    assert sorted(sig.signal_type for sig in config.signals) == [
        "cyclical_temporal",
        "ohlc",
    ]
    assert config.sizing.sizer_type == "risk_based"
    assert config.sizing.params["max_fraction"] == pytest.approx(0.03)
    assert config.sizing.params["risk_multiplier"] == pytest.approx(2.5)
    assert config.entry_threshold == pytest.approx(0.4)
    assert config.exit_threshold == pytest.approx(0.15)
    assert config.allow_long is True
    assert config.allow_short is False


def test_export_to_mql5_writes_file_and_returns_metadata(tmp_path: Path):
    strategy = StandardStrategy(
        indicators=[
            (SMA(OHLC(), period=10), 1.0),
            (RSI(period=14), 0.3),
        ],
        position_sizer=FixedPositionSizer(0.02),
        entry_threshold=0.35,
        exit_threshold=0.1,
    )

    out_file = tmp_path / "generated_ea.mq5"
    result = export_to_mql5(
        strategy,
        timeframe="PERIOD_M15",
        output_path=str(out_file),
        magic_number=42,
        ea_name="UnitTest EA",
    )

    assert out_file.exists()
    assert Path(result.filepath) == out_file.resolve()
    assert result.validation.is_valid
    assert len(result.indicators_exported) == 2
    assert "EntryThreshold = 0.3500;" in result.code
    assert "GetSMAStrength()" in result.code
    assert "GetRSIStrength()" in result.code
    assert "PERIOD_M15" in result.code
    assert "MagicNumber    = 42" in result.code
    assert out_file.read_bytes().startswith(b"\xef\xbb\xbf")


def test_export_to_mql5_raises_for_invalid_strategy(tmp_path: Path):
    out_file = tmp_path / "invalid.mq5"

    with pytest.raises(ValueError, match="Strategy validation failed"):
        export_to_mql5(object(), output_path=str(out_file))
