from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.indicators.base import BaseIndicator
from trade_lab.indicators.moving_averages import EMA, SMA
from trade_lab.indicators.oscillators import RSI
from trade_lab.signals.base import BaseSignal
from trade_lab.signals.signals import HeikinAshi, OHLC
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer


def _sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "Open": [10, 11, 12, 13, 14, 15, 16, 17],
            "High": [11, 12, 13, 14, 15, 16, 17, 18],
            "Low": [9, 10, 11, 12, 13, 14, 15, 16],
            "Close": [10, 11, 12, 13, 12, 11, 12, 13],
            "Volume": [100, 110, 120, 130, 140, 150, 160, 170],
        },
        index=idx,
    )


class _PassthroughSignal(BaseSignal):
    def __init__(self, source: BaseSignal | None = None):
        super().__init__(source)
        self.calls = 0

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls += 1
        data = self.get_data(df)
        out = data.copy()
        out["signal__dummy"] = 1.0
        return out

    def plot(self, df: pd.DataFrame):
        return None

    @property
    def output_columns(self) -> list[str]:
        return ["signal__dummy"]


class _PassthroughIndicator(BaseIndicator):
    def __init__(self, *signals: BaseSignal):
        super().__init__(*signals)
        self.calls = 0

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls += 1
        data = self.get_data(df)
        out = data.copy()
        out["indicator__dummy"] = 2.0
        return out

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.zeros(len(df)), index=df.index)

    def plot(self, df: pd.DataFrame, ax=None):
        return None

    @property
    def output_columns(self) -> list[str]:
        return ["indicator__dummy"]


def test_base_signal_and_indicator_chain_get_data():
    source = _PassthroughSignal()
    chained = _PassthroughSignal(source=source)
    indicator = _PassthroughIndicator(chained)

    out = indicator.compute(_sample_ohlcv())

    assert source.calls == 1
    assert chained.calls == 1
    assert indicator.calls == 1
    assert "signal__dummy" in out.columns
    assert "indicator__dummy" in out.columns


def test_ohlc_signal_compute_and_plot_and_columns():
    df = _sample_ohlcv()
    signal = OHLC()

    out = signal.compute(df.copy())
    signal.plot(out)

    for col in signal.output_columns:
        assert col in out.columns
    assert np.isnan(out["signal__log_return_close"].iloc[0])


def test_heikin_ashi_signal_compute_and_plot_and_columns():
    import plotly.graph_objects as go

    if not hasattr(go, "Candlestick"):
        class _DummyCandlestick:
            def __init__(self, *args, **kwargs):
                pass

        _figure_cls = go.Figure

        def _figure_factory(*args, **kwargs):
            return _figure_cls()

        go.Candlestick = _DummyCandlestick
        go.Figure = _figure_factory

    df = _sample_ohlcv()
    signal = HeikinAshi()

    out = signal.compute(df.copy())
    signal.plot(out)

    for col in signal.output_columns:
        assert col in out.columns
    assert np.isnan(out["signal__ha_close"].iloc[0])


def test_sma_ema_and_rsi_compute_signal_strength_plot_and_output_columns():
    df = _sample_ohlcv()
    sma = SMA(period=3)
    ema = EMA(period=3)
    rsi = RSI(period=3)

    df_sma = sma.compute(df.copy())
    df_ema = ema.compute(df.copy())
    df_rsi = rsi.compute(df.copy())
    sma_strength = sma.to_signal_strength(df_sma)
    ema_strength = ema.to_signal_strength(df_ema)
    rsi_strength = rsi.to_signal_strength(df_rsi)

    sma.plot(df_sma)
    ema.plot(df_ema)
    rsi.plot(df_rsi)

    assert sma.output_columns == ["indicator__sma_3"]
    assert ema.output_columns == ["indicator__ema_3"]
    assert rsi.output_columns == ["indicator__rsi_3"]
    assert len(sma_strength) == len(df)
    assert len(ema_strength) == len(df)
    assert len(rsi_strength) == len(df)
    assert np.isfinite(sma_strength.to_numpy()).all()
    assert np.isfinite(ema_strength.to_numpy()).all()
    assert np.isnan(rsi_strength.iloc[0])


def test_fixed_position_sizer_validation_and_compute_size():
    with pytest.raises(ValueError, match="fraction must be in"):
        FixedPositionSizer(0)

    sizer = FixedPositionSizer(0.25)
    assert sizer.compute_size(signal_strength=0.8, equity=2000, price=20) == 25


def test_risk_based_position_sizer_validation_and_compute_size():
    with pytest.raises(ValueError, match="max_fraction must be in"):
        RiskBasedPositionSizer(max_fraction=0)
    with pytest.raises(ValueError, match="risk_multiplier must be positive"):
        RiskBasedPositionSizer(risk_multiplier=0)

    sizer = RiskBasedPositionSizer(max_fraction=0.1, risk_multiplier=2.0)
    assert sizer.compute_size(0.5, equity=1000, price=10, volatility=None) == 0.0
    assert sizer.compute_size(0.5, equity=1000, price=10, volatility=0) == 0.0
    assert sizer.compute_size(-0.5, equity=1000, price=10, volatility=5.0) == 5.0

