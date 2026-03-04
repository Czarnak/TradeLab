"""Pre-export validation for MQL5 code generation.

Checks a strategy object for compatibility with the MQL5 exporter before
any code is generated. Errors are fatal; warnings are informational.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trade_lab.indicators.moving_averages import CMA, EMA, SMA, WMA
from trade_lab.indicators.oscillators import MACD, LarryWilliams, Momentum, RSI
from trade_lab.signals.signals import OHLC, HeikinAshi
from trade_lab.signals.temporal import CyclicalTemporalSignal
from trade_lab.sizing.fixed import FixedPositionSizer
from trade_lab.sizing.risk_based import RiskBasedPositionSizer
from trade_lab.strategies.standard import StandardStrategy

_SUPPORTED_INDICATORS = (SMA, EMA, WMA, CMA, RSI, MACD, Momentum, LarryWilliams)
_SUPPORTED_SIGNALS = (OHLC, HeikinAshi, CyclicalTemporalSignal)
_SUPPORTED_SIZERS = (type(None), FixedPositionSizer, RiskBasedPositionSizer)


@dataclass
class ValidationResult:
    """Result of pre-export strategy validation.

    Parameters
    ----------
    is_valid : bool
        True if the strategy can be exported (no fatal errors).
    errors : list[str]
        Fatal issues that prevent code generation.
    warnings : list[str]
        Non-fatal issues the user should be aware of.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_strategy(strategy: object) -> ValidationResult:
    """Validate a strategy for MQL5 export compatibility.

    Parameters
    ----------
    strategy : object
        The strategy to validate. Must be a ``StandardStrategy`` instance.

    Returns
    -------
    ValidationResult
        Contains ``is_valid``, ``errors`` (fatal), and ``warnings`` (non-fatal).

    Notes
    -----
    Validation is performed in this order:

    1. Strategy class check — must be ``StandardStrategy``.
    2. Indicator class check — all must be from the supported set.
    3. Signal class check — all upstream signals must be from the supported set.
    4. Position sizer class check — must be ``None``, ``FixedPositionSizer``,
       or ``RiskBasedPositionSizer``.
    5. Warnings for MACD (unnormalised signal strength) and CMA (unusual for
       live trading due to expanding-window state).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Strategy type
    if not isinstance(strategy, StandardStrategy):
        errors.append(
            f"Strategy must be StandardStrategy, got {type(strategy).__name__}. "
            "MLStrategy and other subclasses are not yet supported for MQL5 export."
        )
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    _supported_indicator_names = [c.__name__ for c in _SUPPORTED_INDICATORS]
    _supported_signal_names = [c.__name__ for c in _SUPPORTED_SIGNALS]

    macd_warned = False
    cma_warned = False

    for indicator, _ in strategy.indicators:
        # 2. Indicator class
        if not isinstance(indicator, _SUPPORTED_INDICATORS):
            errors.append(
                f"Unsupported indicator: {type(indicator).__name__}. "
                f"Supported: {_supported_indicator_names}"
            )

        # 3. Signal classes
        for signal in indicator.signals:
            if not isinstance(signal, _SUPPORTED_SIGNALS):
                errors.append(
                    f"Unsupported signal {type(signal).__name__} on indicator "
                    f"{type(indicator).__name__}. "
                    f"Supported: {_supported_signal_names}"
                )

        # Warnings (emit each type at most once)
        if not macd_warned and isinstance(indicator, MACD):
            warnings.append(
                "MACD signal strength is unnormalised (raw signal line values, not in "
                "[-1, 1]). This may cause the combined signal_strength to exceed its "
                "expected range and behave unexpectedly relative to the entry threshold."
            )
            macd_warned = True

        if not cma_warned and isinstance(indicator, CMA):
            warnings.append(
                "CMA (Cumulative Moving Average) uses an expanding window from bar 0. "
                "The generated EA implements this with a custom loop buffer that resets "
                "when the EA is reloaded — results may differ from a continuous backtest."
            )
            cma_warned = True

    # 4. Position sizer
    if strategy.position_sizer is not None and not isinstance(
        strategy.position_sizer, (FixedPositionSizer, RiskBasedPositionSizer)
    ):
        errors.append(
            f"Unsupported position sizer: {type(strategy.position_sizer).__name__}. "
            "Supported: None, FixedPositionSizer, RiskBasedPositionSizer."
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
