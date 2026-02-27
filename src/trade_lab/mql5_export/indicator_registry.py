"""Registry mapping TradeLab indicator classes to MQL5 rendering metadata.

Each entry describes how the corresponding Python indicator maps to MQL5
built-in functions (or custom implementations), how many buffers are needed,
and how the signal strength formula is expressed in MQL5.
"""
from __future__ import annotations

from dataclasses import dataclass

from trade_lab.indicators.moving_averages import CMA, DEMA, EMA, SMA, TEMA, WMA
from trade_lab.indicators.oscillators import (
    MACD,
    BollingerBands,
    CCI,
    DeMarker,
    DPO,
    LarryWilliams,
    Momentum,
    ROC,
    RSI,
    RVI,
    Stochastic,
    TRIX,
)
from trade_lab.indicators.trend import ADX, ATR, MassIndex
from trade_lab.indicators.volume import CHO, OBV, ForceIndex

# ---------------------------------------------------------------------------
# Applied-price mapping (shared by all indicators that accept a column param)
# ---------------------------------------------------------------------------

APPLIED_PRICE_MAP: dict[str, str] = {
    "Close": "PRICE_CLOSE",
    "Open": "PRICE_OPEN",
    "High": "PRICE_HIGH",
    "Low": "PRICE_LOW",
}


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


@dataclass
class MQL5IndicatorDescriptor:
    """MQL5 rendering metadata for a TradeLab indicator.

    Parameters
    ----------
    uses_builtin_handle : bool
        True when MQL5 provides a built-in handle function (iMA, iRSI, iMACD).
        False for custom implementations (CMA, Momentum, LarryWilliams).
    builtin_function : str | None
        MQL5 function name, e.g. ``'iMA'``, ``'iRSI'``, ``'iMACD'``, or ``None``
        for custom indicators.
    ma_method : str | None
        MQL5 ``ENUM_MA_METHOD`` constant required by ``iMA``, e.g.
        ``'MODE_SMA'``, ``'MODE_EMA'``, ``'MODE_LWMA'``. ``None`` for
        non-MA indicators.
    n_buffers : int
        Number of CopyBuffer calls needed. 1 for most indicators, 2 for MACD
        (main line + signal line), 0 for fully custom indicators.
    signal_strength_formula : str
        Human-readable description of the MQL5 signal-strength formula,
        matching the Python ``to_signal_strength()`` implementation.
    applied_price_map : dict[str, str]
        Mapping from TradeLab ``column`` strings to MQL5 ``ENUM_APPLIED_PRICE``
        constants.
    """

    uses_builtin_handle: bool
    builtin_function: str | None
    ma_method: str | None
    n_buffers: int
    signal_strength_formula: str
    applied_price_map: dict[str, str]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

INDICATOR_REGISTRY: dict[type, MQL5IndicatorDescriptor] = {
    # ------------------------------------------------------------------
    # Moving averages — built-in iMA handle
    # ------------------------------------------------------------------
    SMA: MQL5IndicatorDescriptor(
        uses_builtin_handle=True,
        builtin_function="iMA",
        ma_method="MODE_SMA",
        n_buffers=1,
        signal_strength_formula=(
            "MathTanh((price - ma_value) / RollingStd(period, 0, applied_price))"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    EMA: MQL5IndicatorDescriptor(
        uses_builtin_handle=True,
        builtin_function="iMA",
        ma_method="MODE_EMA",
        n_buffers=1,
        signal_strength_formula=(
            "MathTanh((price - ma_value) / RollingStd(period, 0, applied_price))"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    WMA: MQL5IndicatorDescriptor(
        # MQL5's LWMA (Linearly Weighted MA) matches TradeLab's WMA exactly:
        # weights = [1, 2, ..., period] normalised.
        uses_builtin_handle=True,
        builtin_function="iMA",
        ma_method="MODE_LWMA",
        n_buffers=1,
        signal_strength_formula=(
            "MathTanh((price - ma_value) / RollingStd(period, 0, applied_price))"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    CMA: MQL5IndicatorDescriptor(
        # No built-in equivalent — requires a custom expanding-mean loop.
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "MathTanh((price - cma_value) / RollingStd(period, 0, applied_price))"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    DEMA: MQL5IndicatorDescriptor(
        # No MQL5 built-in; custom: 2*EMA(period) - EMA(EMA(period)).
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "MathTanh((price - dema_value) / RollingStd(period, 0, applied_price))"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    TEMA: MQL5IndicatorDescriptor(
        # No MQL5 built-in; custom: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)).
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "MathTanh((price - tema_value) / RollingStd(period, 0, applied_price))"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    # ------------------------------------------------------------------
    # Classic oscillators
    # ------------------------------------------------------------------
    RSI: MQL5IndicatorDescriptor(
        uses_builtin_handle=True,
        builtin_function="iRSI",
        ma_method=None,
        n_buffers=1,
        signal_strength_formula="MathTanh((50.0 - rsi_value) / 20.0)",
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    MACD: MQL5IndicatorDescriptor(
        # Two buffers: index 0 = MACD main line, index 1 = signal line.
        # Signal strength = raw signal line value (no tanh normalisation).
        uses_builtin_handle=True,
        builtin_function="iMACD",
        ma_method=None,
        n_buffers=2,
        signal_strength_formula=(
            "buf_signal[0]  // raw signal line, no tanh — may exceed [-1, 1]"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    Momentum: MQL5IndicatorDescriptor(
        # TradeLab: diff(period) then sign(). MQL5's iMomentum returns a ratio
        # (100 * close/close[period]), so we implement the difference directly.
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "diff = iClose(_Symbol, PERIOD_CURRENT, 0) - "
            "iClose(_Symbol, PERIOD_CURRENT, period); "
            "return diff > 0 ? 1.0 : diff < 0 ? -1.0 : 0.0;"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    LarryWilliams: MQL5IndicatorDescriptor(
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "highest = max(High[lag..lag+period-1]); "
            "lowest  = min(Low[lag..lag+period-1]); "
            "denom   = highest - lowest; "
            "lwr     = denom != 0 ? (highest - close) / denom * -100 : -50; "
            "return MathTanh((-50.0 - lwr) / 20.0);"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    # ------------------------------------------------------------------
    # New oscillators
    # ------------------------------------------------------------------
    BollingerBands: MQL5IndicatorDescriptor(
        # MT5 iBands: buffer 0 = middle, 1 = upper, 2 = lower.
        uses_builtin_handle=True,
        builtin_function="iBands",
        ma_method=None,
        n_buffers=3,
        signal_strength_formula=(
            "band_width = upper - lower; "
            "pct_b = band_width != 0 ? (price - lower) / band_width : 0.5; "
            "return MathTanh((0.5 - pct_b) * 4.0);"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    CCI: MQL5IndicatorDescriptor(
        uses_builtin_handle=True,
        builtin_function="iCCI",
        ma_method=None,
        n_buffers=1,
        signal_strength_formula="MathTanh(-cci_value / 100.0)",
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    Stochastic: MQL5IndicatorDescriptor(
        # MT5 iStochastic: buffer 0 = %K (main), buffer 1 = %D (signal).
        uses_builtin_handle=True,
        builtin_function="iStochastic",
        ma_method=None,
        n_buffers=2,
        signal_strength_formula="MathTanh((50.0 - k_value) / 20.0)",
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    ROC: MQL5IndicatorDescriptor(
        # No MT5 built-in; custom rate-of-change computation.
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "past = iClose(_Symbol, PERIOD_CURRENT, period); "
            "roc  = past != 0 ? (iClose(_Symbol, PERIOD_CURRENT, 0) - past) / past * 100 : 0; "
            "return MathTanh(roc / RollingStd(period, 0, PRICE_CLOSE));"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    TRIX: MQL5IndicatorDescriptor(
        # No MT5 built-in; custom triple-EMA + ROC.
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "// EMA3 computed manually; trix = (ema3 - ema3_prev) / ema3_prev * 100; "
            "return MathTanh(trix / RollingStd(period, 0, PRICE_CLOSE));"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    DPO: MQL5IndicatorDescriptor(
        # No MT5 built-in; custom detrended-price oscillator.
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "half = period / 2 + 1; "
            "sma  = iMA(_Symbol, PERIOD_CURRENT, half, 0, MODE_SMA, PRICE_CLOSE); "
            "dpo  = iClose(_Symbol, PERIOD_CURRENT, 0) - sma; "
            "return MathTanh(dpo / RollingStd(period, 0, PRICE_CLOSE));"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    RVI: MQL5IndicatorDescriptor(
        # MT5 iRVI: buffer 0 = RVI main, buffer 1 = signal line.
        uses_builtin_handle=True,
        builtin_function="iRVI",
        ma_method=None,
        n_buffers=2,
        signal_strength_formula="MathTanh(rvi_value * 5.0)",
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    DeMarker: MQL5IndicatorDescriptor(
        uses_builtin_handle=True,
        builtin_function="iDeMarker",
        ma_method=None,
        n_buffers=1,
        signal_strength_formula="MathTanh((0.5 - dem_value) * 6.0)",
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    # ------------------------------------------------------------------
    # Volume indicators  (require Volume column in the DataFrame)
    # ------------------------------------------------------------------
    OBV: MQL5IndicatorDescriptor(
        # No MT5 built-in matching TradeLab's ROC-of-OBV variant.
        # Raw iOBV would need a rate-of-change wrapper in MQL5.
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "obv_roc = (obv - obv[period]) / RollingStd(obv, period); "
            "return MathTanh(obv_roc);"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    ForceIndex: MQL5IndicatorDescriptor(
        # No MT5 built-in; custom: EMA(volume * diff(close), period).
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "force = volume * (close - close[1]); "
            "fi_ema = EMA(force, period); "
            "return MathTanh(fi_ema / RollingStd(fi_ema, period));"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    CHO: MQL5IndicatorDescriptor(
        # MT5 iChaikin: buffer 0 = oscillator value.
        uses_builtin_handle=True,
        builtin_function="iChaikin",
        ma_method=None,
        n_buffers=1,
        signal_strength_formula=(
            "return MathTanh(cho_value / RollingStd(cho, slow_period));"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    # ------------------------------------------------------------------
    # Trend / volatility indicators
    # ------------------------------------------------------------------
    ADX: MQL5IndicatorDescriptor(
        # MT5 iADX: buffer 0 = ADX, 1 = +DI, 2 = -DI.
        uses_builtin_handle=True,
        builtin_function="iADX",
        ma_method=None,
        n_buffers=3,
        signal_strength_formula=(
            "di_sum = plus_di + minus_di; "
            "ratio  = di_sum != 0 ? (plus_di - minus_di) / di_sum : 0.0; "
            "return ratio * MathTanh(adx_value / 25.0);"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    ATR: MQL5IndicatorDescriptor(
        # MT5 iATR: buffer 0 = ATR value.
        # to_signal_strength raises NotImplementedError — marked N/A here.
        uses_builtin_handle=True,
        builtin_function="iATR",
        ma_method=None,
        n_buffers=1,
        signal_strength_formula="N/A — ATR is a utility indicator with no directional signal",
        applied_price_map=APPLIED_PRICE_MAP,
    ),
    MassIndex: MQL5IndicatorDescriptor(
        # No MT5 built-in; custom double-EMA ratio summation.
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "return MathTanh((26.75 - mi_value) / 1.5);"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
}