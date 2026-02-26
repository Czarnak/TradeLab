"""MQL5 code generator — orchestrates introspection, rendering and file output.

Ties together the validator, introspector, registries, and Jinja2 templates
to produce a complete MQL5 Expert Advisor source file from a StandardStrategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import jinja2
except ImportError as _exc:
    raise ImportError(
        "The mql5_export module requires Jinja2. "
        "Install it with:  pip install 'TradeLab[mql5]'"
    ) from _exc

from trade_lab.indicators.moving_averages import CMA, EMA, SMA, WMA
from trade_lab.indicators.oscillators import MACD, LarryWilliams, Momentum, RSI
from trade_lab.mql5_export.indicator_registry import APPLIED_PRICE_MAP, INDICATOR_REGISTRY
from trade_lab.mql5_export.introspector import StrategyIntrospector
from trade_lab.mql5_export.validators import ValidationResult, validate_strategy

# ---------------------------------------------------------------------------
# Mapping from indicator_type string → Python class (for registry lookup)
# ---------------------------------------------------------------------------

_INDICATOR_CLASS_MAP: dict[str, type] = {
    "sma": SMA,
    "ema": EMA,
    "wma": WMA,
    "cma": CMA,
    "rsi": RSI,
    "macd": MACD,
    "momentum": Momentum,
    "larry_williams": LarryWilliams,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MQL5ExportResult:
    """Result returned by ``export_to_mql5()``.

    Parameters
    ----------
    filepath : str
        Absolute path of the written ``.mq5`` file.
    code : str
        Full rendered MQL5 source code.
    validation : ValidationResult
        Validation result (may contain warnings even on success).
    indicators_exported : list[str]
        Human-readable summary of each exported indicator.
    """

    filepath: str
    code: str
    validation: ValidationResult
    indicators_exported: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_jinja_env() -> jinja2.Environment:
    """Create a Jinja2 environment pointing at the templates directory."""
    templates_dir = Path(__file__).parent / "templates"
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _enrich_indicator(ind) -> dict:
    """Return a flat dict merging IndicatorConfig fields with registry metadata.

    The template receives this dict rather than the dataclass so it does not
    need to navigate Python-specific objects or know the registry structure.
    """
    from trade_lab.indicators.moving_averages import CMA, EMA, SMA, WMA
    from trade_lab.indicators.oscillators import MACD, LarryWilliams, Momentum, RSI

    _INDICATOR_CLASS_MAP = {
        "sma": SMA, "ema": EMA, "wma": WMA, "cma": CMA,
        "rsi": RSI, "macd": MACD, "momentum": Momentum,
        "larry_williams": LarryWilliams,
    }

    cls = _INDICATOR_CLASS_MAP[ind.indicator_type]
    desc = INDICATOR_REGISTRY[cls]
    column = ind.params.get("column", "Close")
    return {
        # --- IndicatorConfig fields ---
        "indicator_type":  ind.indicator_type,
        "class_name":      ind.class_name,
        "params":          ind.params,
        "weight":          ind.weight,
        "signals":         ind.signals,
        "output_columns":  ind.output_columns,
        "var_name":        ind.var_name,
        "input_prefix":    ind.input_prefix,
        "function_name":   ind.function_name,
        "lag":             ind.lag,             # <-- new
        # --- Registry descriptor fields ---
        "uses_builtin_handle": desc.uses_builtin_handle,
        "builtin_function":    desc.builtin_function,
        "ma_method":           desc.ma_method,
        "n_buffers":           desc.n_buffers,
        # --- Computed convenience fields ---
        "applied_price":  APPLIED_PRICE_MAP.get(column, "PRICE_CLOSE"),
        "is_ma_type":     ind.indicator_type in ("sma", "ema", "wma"),
    }





def _format_indicator_summary(ind: dict) -> str:
    """Format a one-line indicator description for MQL5ExportResult."""
    params_str = ", ".join(f"{k}={v}" for k, v in ind["params"].items())
    return f"{ind['class_name']}({params_str}) weight={ind['weight']}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_to_mql5(
    strategy: object,
    symbol: str = "EURUSD",
    timeframe: str = "PERIOD_H1",
    output_path: str = "outputs\\TradeLab_EA.mq5",
    magic_number: int = 123456,
    max_spread: int = 20,
    ea_name: str = "TradeLab Expert Advisor",
    ea_description: str = "Auto-generated from TradeLab StandardStrategy",
) -> MQL5ExportResult:
    """Export a ``StandardStrategy`` to a MetaTrader 5 Expert Advisor ``.mq5`` file.

    Parameters
    ----------
    strategy : StandardStrategy
        The strategy to export.
    symbol : str
        Default trading symbol used in indicator handle creation, e.g.
        ``'EURUSD'``.  In the EA this is always overridden by ``_Symbol``
        at runtime, but some handle calls benefit from a compile-time default.
    timeframe : str
        MQL5 ``ENUM_TIMEFRAMES`` constant, e.g. ``'PERIOD_H1'``,
        ``'PERIOD_D1'``.
    output_path : str
        File path for the generated ``.mq5`` file.
    magic_number : int
        EA magic number.  Trades opened by the EA carry this number so
        position management can filter its own trades from others.
    max_spread : int
        Maximum allowed spread in points.  Ticks with a wider spread are
        skipped.
    ea_name : str
        EA name shown in the MetaTrader Experts list.
    ea_description : str
        One-line description in the ``#property description`` block.

    Returns
    -------
    MQL5ExportResult
        File path, rendered code, validation result, and indicator summaries.

    Raises
    ------
    ValueError
        If the strategy fails validation (fatal errors).
    ImportError
        If Jinja2 is not installed.
    """
    # 1. Validate
    validation = validate_strategy(strategy)
    if not validation.is_valid:
        lines = "\n".join(f"  • {e}" for e in validation.errors)
        raise ValueError(f"Strategy validation failed:\n{lines}")

    # Print warnings so caller is aware even without inspecting the result
    for warning in validation.warnings:
        print(f"[mql5_export WARNING] {warning}")

    # 2. Introspect
    config = StrategyIntrospector().introspect(strategy)

    # 3. Enrich indicators with descriptor metadata (flat dicts for templates)
    indicators = [_enrich_indicator(ind) for ind in config.indicators]

    # 4. Build Jinja2 context
    context: dict = {
        "ea_name": ea_name,
        "ea_description": ea_description,
        "symbol": symbol,
        "timeframe": timeframe,
        "magic_number": magic_number,
        "max_spread": max_spread,
        "config": config,         # StrategyConfig dataclass (for sizing, thresholds, flags)
        "indicators": indicators,  # list of enriched dicts
    }

    # 5. Render
    env = _make_jinja_env()
    template = env.get_template("ea_main.mq5.j2")
    code = template.render(**context)

    # 6. Write file (UTF-8 with BOM — MetaEditor expects this)
    output_file = Path(output_path)
    output_file.write_text(code, encoding="utf-8-sig")

    # 7. Return result
    indicators_exported = [_format_indicator_summary(ind) for ind in indicators]
    return MQL5ExportResult(
        filepath=str(output_file.resolve()),
        code=code,
        validation=validation,
        indicators_exported=indicators_exported,
    )
