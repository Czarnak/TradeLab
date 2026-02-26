"""Registry mapping TradeLab indicator classes to MQL5 rendering metadata.

Each entry describes how the corresponding Python indicator maps to MQL5
built-in functions (or custom implementations), how many buffers are needed,
and how the signal strength formula is expressed in MQL5.
"""
from __future__ import annotations

from dataclasses import dataclass

from trade_lab.indicators.moving_averages import CMA, EMA, SMA, WMA
from trade_lab.indicators.oscillators import MACD, LarryWilliams, Momentum, RSI

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
        # No built-in equivalent — requires a custom expanding-mean loop
        # that accumulates a running sum from bar 0 on each tick.
        uses_builtin_handle=False,
        builtin_function=None,
        ma_method=None,
        n_buffers=0,
        signal_strength_formula=(
            "MathTanh((price - cma_value) / RollingStd(period, 0, applied_price))"
        ),
        applied_price_map=APPLIED_PRICE_MAP,
    ),
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
        # No built-in equivalent that matches the sign convention.
        # Custom implementation: rolling max/min over period bars.
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
}
