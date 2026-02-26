"""Strategy introspection for MQL5 code generation.

Walks a ``StandardStrategy`` object tree and extracts a fully structured
``StrategyConfig`` that the code generator can consume without further
knowledge of the TradeLab class hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trade_lab.indicators.moving_averages import CMA, EMA, SMA, WMA
from trade_lab.indicators.oscillators import MACD, LarryWilliams, Momentum, RSI
from trade_lab.signals.signals import OHLC, HeikinAshi
from trade_lab.signals.temporal import CyclicalTemporalSignal
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer

if TYPE_CHECKING:
    from trade_lab.indicators.base import BaseIndicator
    from trade_lab.signals.base import BaseSignal
    from trade_lab.strategies.standard import StandardStrategy

# ---------------------------------------------------------------------------
# Internal type maps
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SignalConfig:
    """Structured representation of an upstream signal.

    Parameters
    ----------
    signal_type : str
        Snake-case type identifier, e.g. ``'ohlc'``, ``'heikin_ashi'``,
        ``'cyclical_temporal'``.
    class_name : str
        Python class name, e.g. ``'OHLC'``, ``'HeikinAshi'``.
    params : dict
        Signal constructor parameters extracted by introspection.
        For ``CyclicalTemporalSignal``: ``{'component': str, 'period': float}``.
        For ``OHLC`` and ``HeikinAshi``: ``{}``.
    output_columns : list[str]
        Column names appended to the DataFrame, e.g.
        ``['signal__log_return_open', ...]``.
    """

    signal_type: str
    class_name: str
    params: dict
    output_columns: list[str]


@dataclass
class IndicatorConfig:
    """Structured representation of one indicator slot in a ``StandardStrategy``.

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


@dataclass
class SizingConfig:
    """Structured representation of a position sizer.

    Parameters
    ----------
    sizer_type : str
        One of ``'none'``, ``'fixed'``, ``'risk_based'``.
    params : dict
        Sizer constructor parameters.
        For ``'fixed'``: ``{'fraction': float}``.
        For ``'risk_based'``: ``{'max_fraction': float, 'risk_multiplier': float}``.
        For ``'none'``: ``{}``.
    """

    sizer_type: str
    params: dict


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
    for attr in ("column", "period", "fast_period", "slow_period"):
        if hasattr(indicator, attr):
            params[attr] = getattr(indicator, attr)
    return params


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
    * Exactly two indicators of the same type → ``'<type>_fast'`` and
      ``'<type>_slow'`` ordered by primary period ascending.
    * Three or more → ``'<type>_1'``, ``'<type>_2'``, ``'<type>_3'`` ...
      ordered by primary period ascending.

    Parameters
    ----------
    indicator_types : list[str]
        Snake-case type string for each indicator (same order as strategy).
    indicator_params : list[dict]
        Params dict for each indicator (used for period-based ordering).

    Returns
    -------
    list[str]
        Unique var_name for each indicator in the original order.
    """
    # Group indices by type
    type_to_indices: dict[str, list[int]] = {}
    for i, itype in enumerate(indicator_types):
        type_to_indices.setdefault(itype, []).append(i)

    var_names: list[str] = [""] * len(indicator_types)

    for itype, indices in type_to_indices.items():
        n = len(indices)
        if n == 1:
            var_names[indices[0]] = itype
        elif n == 2:
            ordered = sorted(indices, key=lambda i: _sort_key(indicator_params[i]))
            var_names[ordered[0]] = f"{itype}_fast"
            var_names[ordered[1]] = f"{itype}_slow"
        else:
            ordered = sorted(indices, key=lambda i: _sort_key(indicator_params[i]))
            for rank, idx in enumerate(ordered, start=1):
                var_names[idx] = f"{itype}_{rank}"

    return var_names


# Proper display case for each indicator type.
# Tuple: (input_prefix_base, function_name_base)
# input_prefix_base uses underscores (e.g. 'Larry_Williams')
# function_name_base is PascalCase without underscores (e.g. 'LarryWilliams')
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


def _make_input_prefix(var_name: str, indicator_type: str) -> str:
    """Convert var_name + indicator_type to a Title_Case MQL5 input prefix.

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
    base_input, _ = _TYPE_DISPLAY_MAP.get(indicator_type, (indicator_type.capitalize(), ""))
    suffix = var_name[len(indicator_type):]   # '' or '_fast', '_1', ...
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
    _, base_fn = _TYPE_DISPLAY_MAP.get(indicator_type, ("", indicator_type.capitalize()))
    suffix = var_name[len(indicator_type):]
    if suffix:
        suffix_parts = suffix.lstrip("_").split("_")
        suffix_str = "".join(p.capitalize() for p in suffix_parts if p)
        return f"{base_fn}{suffix_str}"
    return base_fn


# ---------------------------------------------------------------------------
# Public introspector
# ---------------------------------------------------------------------------


class StrategyIntrospector:
    """Walks a ``StandardStrategy`` object tree and returns a ``StrategyConfig``.

    No knowledge of Jinja2 or MQL5 syntax is required here — this class
    purely extracts and restructures Python-side information.

    Examples
    --------
    >>> from trade_lab.indicators import EMA, RSI
    >>> from trade_lab.strategies import StandardStrategy
    >>> from trade_lab.mql5_export.introspector import StrategyIntrospector
    >>>
    >>> strategy = StandardStrategy(
    ...     indicators=[(EMA(period=20), 1.0), (RSI(period=14), 0.5)],
    ... )
    >>> config = StrategyIntrospector().introspect(strategy)
    >>> print(config.indicators[0].var_name)   # 'ema'
    >>> print(config.indicators[0].function_name)  # 'EMA'
    """

    def introspect(self, strategy: StandardStrategy) -> StrategyConfig:
        """Introspect a ``StandardStrategy`` and return a ``StrategyConfig``.

        Parameters
        ----------
        strategy : StandardStrategy
            The strategy to introspect.

        Returns
        -------
        StrategyConfig
            Fully structured configuration ready for MQL5 template rendering.
        """
        # First pass: collect types and params for var-name generation
        indicator_types: list[str] = []
        indicator_params_list: list[dict] = []
        raw: list[tuple] = []  # (indicator, weight, itype, params)

        for indicator, weight in strategy.indicators:
            cls = type(indicator)
            itype = _INDICATOR_TYPE_MAP.get(cls, cls.__name__.lower())
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
        )
