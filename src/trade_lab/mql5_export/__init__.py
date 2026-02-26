"""MQL5 Expert Advisor export module for TradeLab.

Translates a ``StandardStrategy`` Python object into a complete, readable,
well-commented MQL5 Expert Advisor (``.mq5`` file) using introspection of the
strategy object tree and Jinja2 templates.

Typical workflow
----------------
1. Build a ``StandardStrategy`` (with indicators, weights and thresholds).
2. Call ``export_to_mql5()`` with the desired output path and EA metadata.
3. Open the generated ``.mq5`` file in MetaEditor, compile, and attach to a chart.

Supported indicators
--------------------
SMA, EMA, WMA, CMA, RSI, MACD, Momentum, LarryWilliams

Supported upstream signals
--------------------------
OHLC, HeikinAshi, CyclicalTemporalSignal

Supported position sizers
-------------------------
None (fixed lot input), FixedPositionSizer, RiskBasedPositionSizer

Optional dependency
-------------------
Requires ``Jinja2>=3.1`` (included in the ``[mql5]`` extra):

.. code-block:: bash

    pip install "TradeLab[mql5]"

Example
-------
>>> from trade_lab.indicators import EMA, RSI
>>> from trade_lab.strategies import StandardStrategy
>>> from trade_lab.mql5_export import export_to_mql5
>>>
>>> strategy = StandardStrategy(
...     indicators=[
...         (EMA(period=20), 1.0),
...         (EMA(period=50), -1.0),
...         (RSI(period=14), 0.5),
...     ],
...     entry_threshold=0.3,
...     exit_threshold=0.1,
...     allow_long=True,
...     allow_short=True,
... )
>>>
>>> result = export_to_mql5(
...     strategy,
...     symbol="EURUSD",
...     timeframe="PERIOD_H1",
...     output_path="MyEA.mq5",
...     magic_number=123456,
... )
>>> print(result.filepath)
>>> print(f"Exported {len(result.indicators_exported)} indicators")
>>> for warning in result.validation.warnings:
...     print(f"[WARNING] {warning}")
"""
from __future__ import annotations

try:
    import jinja2  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "The mql5_export module requires Jinja2. "
        "Install it with:  pip install 'TradeLab[mql5]'"
    ) from _exc

from trade_lab.mql5_export.code_generator import MQL5ExportResult, export_to_mql5
from trade_lab.mql5_export.introspector import StrategyIntrospector
from trade_lab.mql5_export.validators import validate_strategy

__all__ = [
    "export_to_mql5",
    "MQL5ExportResult",
    "StrategyIntrospector",
    "validate_strategy",
]
