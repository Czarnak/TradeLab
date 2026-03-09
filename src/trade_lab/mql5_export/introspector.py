"""Strategy introspector — walks a StandardStrategy object tree.

Produces a ``StrategyConfig`` dataclass that captures everything the Jinja2
templates need to render a complete MQL5 Expert Advisor without any further
Python-side logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trade_lab.indicators.base import BaseIndicator
from trade_lab.signals.base import BaseSignal
from trade_lab.signals.temporal import CyclicalTemporalSignal
from trade_lab.risk_management import (
    BaseStopLoss,
    BaseTakeProfit,
    BaseTrailingStop,
    FixedSL,
    FixedTP,
    FixedTS,
    MovingAverageSL,
    MovingAverageTS,
    ParabolicSARSL,
    ParabolicSARTS,
    SignalStrengthSL,
    SignalStrengthTP,
    SignalStrengthTS,
)
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer
from trade_lab.strategies.standard import StandardStrategy

# ---------------------------------------------------------------------------
# Type → string maps
# ---------------------------------------------------------------------------

from trade_lab.indicators.moving_averages import CMA, EMA, SMA, WMA
from trade_lab.indicators.oscillators import MACD, LarryWilliams, Momentum, RSI
from trade_lab.indicators.statistical import BaseKernel
from trade_lab.signals.signals import HeikinAshi, OHLC

_INDICATOR_TYPE_MAP: dict[type, str] = {
    SMA: "sma",
    EMA: "ema",
    WMA: "wma",
    CMA: "cma",
    RSI: "rsi",
    MACD: "macd",
    Momentum: "momentum",
    LarryWilliams: "larry_williams",
}

_SIGNAL_TYPE_MAP: dict[type, str] = {
    OHLC: "ohlc",
    HeikinAshi: "heikin_ashi",
    CyclicalTemporalSignal: "cyclical_temporal",
}

# Display names for input prefix and function name generation
# (base_input_prefix, base_function_name)
_TYPE_DISPLAY_MAP: dict[str, tuple[str, str]] = {
    "sma": ("SMA", "SMA"),
    "ema": ("EMA", "EMA"),
    "wma": ("WMA", "WMA"),
    "cma": ("CMA", "CMA"),
    "rsi": ("RSI", "RSI"),
    "macd": ("MACD", "MACD"),
    "momentum": ("Momentum", "Momentum"),
    "larry_williams": ("Larry_Williams", "LarryWilliams"),
}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SignalConfig:
    """Structured representation of an upstream signal.

    Parameters
    ----------
    signal_type : str
        One of ``'ohlc'``, ``'heikin_ashi'``, ``'cyclical_temporal'``.
    class_name : str
        Python class name, e.g. ``'OHLC'``.
    params : dict
        Signal constructor parameters (empty for OHLC/HeikinAshi).
        For ``CyclicalTemporalSignal``: ``{'component': str, 'period': float}``.
    output_columns : list[str]
        Column names this signal appends to the DataFrame.
    """

    signal_type: str
    class_name: str
    params: dict
    output_columns: list[str]


@dataclass
class IndicatorConfig:
    """Structured representation of one indicator slot in a strategy.

    Parameters
    ----------
    indicator_type : str
        Snake-case type identifier, e.g. ``'ema'``, ``'rsi'``, ``'macd'``.
    class_name : str
        Python class name, e.g. ``'EMA'``, ``'RSI'``.
    params : dict
        Indicator constructor parameters extracted by introspection.
        Keys: ``'column'``, ``'period'``, ``'fast_period'``, ``'slow_period'``
        as applicable.
    weight : float
        Combination weight from the ``(indicator, weight)`` pair.
    signals : list[SignalConfig]
        Upstream signals attached to this indicator (may be empty).
    output_columns : list[str]
        Column names this indicator appends to the DataFrame.
    var_name : str
        Unique snake_case MQL5 variable prefix, e.g. ``'ema_fast'``,
        ``'ema_slow'``, ``'rsi'``. Derived from indicator type and ordering.
    input_prefix : str
        Title-case prefix for MQL5 ``input`` variable names,
        e.g. ``'EMA_Fast'`` → ``EMA_Fast_Period``, ``EMA_Fast_Price``.
    function_name : str
        PascalCase suffix for the signal-strength function name,
        e.g. ``'EMAFast'`` → ``GetEMAFastStrength()``.
    lag : int
        Bars by which this indicator's output is shifted backward.
        0 means no lag (the default).  Forwarded to ``CopyBuffer`` offset
        in MQL5 templates.
    """

    indicator_type: str
    class_name: str
    params: dict
    weight: float
    signals: list[SignalConfig]
    output_columns: list[str]
    var_name: str
    input_prefix: str
    function_name: str
    lag: int = 0


@dataclass
class SizingConfig:
    """Structured representation of a position sizer.

    Parameters
    ----------
    sizer_type : str
        One of ``'none'``, ``'fixed'``, ``'risk_based'``.
    params : dict
        Sizer constructor parameters.
    """

    sizer_type: str
    params: dict


@dataclass
class RiskConfig:
    has_tp: bool = False
    tp_type: str = "none"
    tp_base_points: float | None = None

    has_sl: bool = False
    sl_type: str = "none"
    sl_base_points: float | None = None

    has_ts: bool = False
    ts_type: str = "none"
    ts_base_points: float | None = None
    ts_step_points: float | None = None


@dataclass
class StrategyConfig:
    """Full structured configuration of a ``StandardStrategy`` for MQL5 export.

    Parameters
    ----------
    indicators : list[IndicatorConfig]
        All indicator slots in the order they appear in the strategy.
    signals : list[SignalConfig]
        Deduplicated union of all upstream signals across all indicators.
    sizing : SizingConfig
        Position sizer configuration.
    entry_threshold : float
        From ``BaseStrategy.entry_threshold``.
    exit_threshold : float
        From ``BaseStrategy.exit_threshold``.
    allow_long : bool
        From ``BaseStrategy.allow_long``.
    allow_short : bool
        From ``BaseStrategy.allow_short``.
    """

    indicators: list[IndicatorConfig]
    signals: list[SignalConfig]
    sizing: SizingConfig
    entry_threshold: float
    exit_threshold: float
    allow_long: bool
    allow_short: bool
    risk: RiskConfig = field(default_factory=RiskConfig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_signal_config(signal: BaseSignal) -> SignalConfig:
    """Extract a ``SignalConfig`` from a signal object."""
    cls = type(signal)
    signal_type = _SIGNAL_TYPE_MAP.get(cls, cls.__name__.lower())
    params: dict = {}
    if isinstance(signal, CyclicalTemporalSignal):
        params = {"component": signal.component, "period": signal.period}
    return SignalConfig(
        signal_type=signal_type,
        class_name=cls.__name__,
        params=params,
        output_columns=list(signal.output_columns),
    )


def _extract_indicator_params(indicator: BaseIndicator) -> dict:
    """Extract relevant constructor params from an indicator by attribute lookup."""
    params: dict = {}
    for attr in (
        "column",
        "period",
        "fast_period",
        "slow_period",
        "bandwidth",
        "deviations",
    ):
        if hasattr(indicator, attr):
            params[attr] = getattr(indicator, attr)
    if hasattr(indicator, "price_source"):
        params["price_source"] = getattr(indicator, "price_source").value
    if hasattr(indicator, "kernel_type"):
        params["kernel"] = getattr(indicator, "kernel_type").value
    return params


def _extract_indicator_type(indicator: BaseIndicator) -> str:
    """Return the stable type identifier used by the exporter."""
    cls = type(indicator)
    if cls in _INDICATOR_TYPE_MAP:
        return _INDICATOR_TYPE_MAP[cls]
    if isinstance(indicator, BaseKernel):
        if cls.__name__ == "SquareKernel":
            return "square_kernel"
        return f"{indicator.kernel_type.name.lower()}_kernel"
    return cls.__name__.lower()


def _default_display_names(indicator_type: str) -> tuple[str, str]:
    parts = [part for part in indicator_type.split("_") if part]
    title = "_".join(
        part.upper() if len(part) <= 4 else part.capitalize() for part in parts
    )
    pascal = "".join(
        part.upper() if len(part) <= 4 else part.capitalize() for part in parts
    )
    return title, pascal


def _signals_equal(a: SignalConfig, b: SignalConfig) -> bool:
    """Return True if two signal configs represent the same signal instance."""
    return a.class_name == b.class_name and a.params == b.params


def _sort_key(params: dict) -> int | float:
    """Return an ordering key for an indicator based on its primary period."""
    return params.get("period", params.get("fast_period", 0))


def _generate_var_names(
    indicator_types: list[str],
    indicator_params: list[dict],
) -> list[str]:
    """Generate unique snake_case MQL5 variable name prefixes for each indicator.

    Rules
    -----
    * Single indicator of a type → just the type name (e.g. ``'rsi'``).
    * Exactly two of the same type → ``'<type>_fast'`` / ``'<type>_slow'``
      ordered by primary period ascending.
    * Three or more → ``'<type>_1'``, ``'<type>_2'``, ... ordered ascending.
    """
    from collections import defaultdict

    # Group indices by type
    type_indices: dict[str, list[int]] = defaultdict(list)
    for i, itype in enumerate(indicator_types):
        type_indices[itype].append(i)

    var_names: list[str] = [""] * len(indicator_types)

    for itype, indices in type_indices.items():
        if len(indices) == 1:
            var_names[indices[0]] = itype
        else:
            # Sort by primary period ascending
            sorted_indices = sorted(
                indices,
                key=lambda i: _sort_key(indicator_params[i]),
            )
            if len(sorted_indices) == 2:
                var_names[sorted_indices[0]] = f"{itype}_fast"
                var_names[sorted_indices[1]] = f"{itype}_slow"
            else:
                for rank, idx in enumerate(sorted_indices, start=1):
                    var_names[idx] = f"{itype}_{rank}"

    return var_names


def _make_input_prefix(var_name: str, indicator_type: str) -> str:
    """Convert var_name to a Title_Case MQL5 input prefix.

    Uses ``_TYPE_DISPLAY_MAP`` to preserve abbreviations (EMA, RSI, MACD)
    while capitalising any trailing suffix (fast, slow, 1, 2, ...).

    Examples
    --------
    >>> _make_input_prefix('ema_fast', 'ema')
    'EMA_Fast'
    >>> _make_input_prefix('rsi', 'rsi')
    'RSI'
    >>> _make_input_prefix('larry_williams_fast', 'larry_williams')
    'Larry_Williams_Fast'
    """
    base_input, _ = _TYPE_DISPLAY_MAP.get(
        indicator_type, _default_display_names(indicator_type)
    )
    suffix = var_name[len(indicator_type) :]  # '' or '_fast', '_1', ...
    if suffix:
        suffix_parts = suffix.lstrip("_").split("_")
        suffix_str = "_".join(p.capitalize() for p in suffix_parts if p)
        return f"{base_input}_{suffix_str}"
    return base_input


def _make_function_name(var_name: str, indicator_type: str) -> str:
    """Convert var_name + indicator_type to a PascalCase MQL5 function name.

    Examples
    --------
    >>> _make_function_name('ema_fast', 'ema')
    'EMAFast'
    >>> _make_function_name('rsi', 'rsi')
    'RSI'
    >>> _make_function_name('larry_williams', 'larry_williams')
    'LarryWilliams'
    """
    _, base_fn = _TYPE_DISPLAY_MAP.get(
        indicator_type, _default_display_names(indicator_type)
    )
    suffix = var_name[len(indicator_type) :]
    if suffix:
        suffix_parts = suffix.lstrip("_").split("_")
        suffix_str = "".join(p.capitalize() for p in suffix_parts if p)
        return f"{base_fn}{suffix_str}"
    return base_fn


def _extract_take_profit_config(take_profit: BaseTakeProfit | None) -> dict:
    if take_profit is None:
        return {"has_tp": False, "tp_type": "none", "tp_base_points": None}
    if isinstance(take_profit, FixedTP):
        return {
            "has_tp": True,
            "tp_type": "fixed",
            "tp_base_points": take_profit.base_points,
        }
    if isinstance(take_profit, SignalStrengthTP):
        return {
            "has_tp": True,
            "tp_type": "signal_strength",
            "tp_base_points": take_profit.base_points,
        }
    return {"has_tp": True, "tp_type": "none", "tp_base_points": None}


def _extract_stop_loss_config(stop_loss: BaseStopLoss | None) -> dict:
    if stop_loss is None:
        return {"has_sl": False, "sl_type": "none", "sl_base_points": None}
    if isinstance(stop_loss, FixedSL):
        return {
            "has_sl": True,
            "sl_type": "fixed",
            "sl_base_points": stop_loss.base_points,
        }
    if isinstance(stop_loss, SignalStrengthSL):
        return {
            "has_sl": True,
            "sl_type": "signal_strength",
            "sl_base_points": stop_loss.base_points,
        }
    if isinstance(stop_loss, MovingAverageSL):
        return {"has_sl": True, "sl_type": "ma", "sl_base_points": None}
    if isinstance(stop_loss, ParabolicSARSL):
        return {"has_sl": True, "sl_type": "sar", "sl_base_points": None}
    return {"has_sl": True, "sl_type": "none", "sl_base_points": None}


def _extract_trailing_stop_config(trailing_stop: BaseTrailingStop | None) -> dict:
    if trailing_stop is None:
        return {
            "has_ts": False,
            "ts_type": "none",
            "ts_base_points": None,
            "ts_step_points": None,
        }
    if isinstance(trailing_stop, FixedTS):
        return {
            "has_ts": True,
            "ts_type": "fixed",
            "ts_base_points": trailing_stop.base_points,
            "ts_step_points": trailing_stop.step_points,
        }
    if isinstance(trailing_stop, SignalStrengthTS):
        return {
            "has_ts": True,
            "ts_type": "signal_strength",
            "ts_base_points": trailing_stop.base_points,
            "ts_step_points": trailing_stop.step_points,
        }
    if isinstance(trailing_stop, MovingAverageTS):
        return {
            "has_ts": True,
            "ts_type": "ma",
            "ts_base_points": None,
            "ts_step_points": trailing_stop.step_points,
        }
    if isinstance(trailing_stop, ParabolicSARTS):
        return {
            "has_ts": True,
            "ts_type": "sar",
            "ts_base_points": None,
            "ts_step_points": trailing_stop.step_points,
        }
    return {
        "has_ts": True,
        "ts_type": "none",
        "ts_base_points": None,
        "ts_step_points": getattr(trailing_stop, "step_points", None),
    }


def extract_risk_config(strategy) -> RiskConfig:
    return RiskConfig(
        **_extract_take_profit_config(strategy.take_profit),
        **_extract_stop_loss_config(strategy.stop_loss),
        **_extract_trailing_stop_config(strategy.trailing_stop),
    )


# ---------------------------------------------------------------------------
# Public introspector
# ---------------------------------------------------------------------------


class StrategyIntrospector:
    """Walks a ``StandardStrategy`` object tree and returns a ``StrategyConfig``."""

    def introspect(self, strategy: StandardStrategy) -> StrategyConfig:
        """Introspect a ``StandardStrategy`` and return a ``StrategyConfig``."""

        # First pass: collect types and params for var-name generation
        indicator_types: list[str] = []
        indicator_params_list: list[dict] = []
        raw: list[tuple] = []  # (indicator, weight, itype, params)

        for indicator, weight in strategy.indicators:
            itype = _extract_indicator_type(indicator)
            params = _extract_indicator_params(indicator)
            indicator_types.append(itype)
            indicator_params_list.append(params)
            raw.append((indicator, weight, itype, params))

        var_names = _generate_var_names(indicator_types, indicator_params_list)

        # Second pass: build IndicatorConfig and deduplicate signals
        all_signals: list[SignalConfig] = []
        indicator_configs: list[IndicatorConfig] = []

        for (indicator, weight, itype, params), var_name in zip(raw, var_names):
            signal_configs: list[SignalConfig] = []
            for signal in indicator.signals:
                sc = _extract_signal_config(signal)
                signal_configs.append(sc)
                if not any(_signals_equal(sc, existing) for existing in all_signals):
                    all_signals.append(sc)

            input_prefix = _make_input_prefix(var_name, itype)
            function_name = _make_function_name(var_name, itype)

            indicator_configs.append(
                IndicatorConfig(
                    indicator_type=itype,
                    class_name=type(indicator).__name__,
                    params=params,
                    weight=weight,
                    signals=signal_configs,
                    output_columns=list(indicator.output_columns),
                    var_name=var_name,
                    input_prefix=input_prefix,
                    function_name=function_name,
                    lag=indicator.lag,  # <-- new: read from indicator
                )
            )

        # Position sizing
        sizer = strategy.position_sizer
        if sizer is None:
            sizing = SizingConfig(sizer_type="none", params={})
        elif isinstance(sizer, FixedPositionSizer):
            sizing = SizingConfig(
                sizer_type="fixed", params={"fraction": sizer.fraction}
            )
        elif isinstance(sizer, RiskBasedPositionSizer):
            sizing = SizingConfig(
                sizer_type="risk_based",
                params={
                    "max_fraction": sizer.max_fraction,
                    "risk_multiplier": sizer.risk_multiplier,
                },
            )
        else:
            sizing = SizingConfig(
                sizer_type="unknown",
                params={"class": type(sizer).__name__},
            )

        return StrategyConfig(
            indicators=indicator_configs,
            signals=all_signals,
            sizing=sizing,
            entry_threshold=strategy.entry_threshold,
            exit_threshold=strategy.exit_threshold,
            allow_long=strategy.allow_long,
            allow_short=strategy.allow_short,
            risk=extract_risk_config(strategy),
        )
