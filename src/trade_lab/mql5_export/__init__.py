"""MQL5 Expert Advisor export module for TradeLab.

Translates a TradeLab strategy Python object into a complete, readable,
well-commented MQL5 Expert Advisor (``.mq5`` file) using introspection of the
strategy object tree and Jinja2 templates.

Two export paths are available:

``StandardStrategy`` path
--------------------------
Uses weighted fuzzy-logic aggregation. Indicators are translated to MQL5
built-in handles (``iMA``, ``iRSI``, etc.) and signal-strength functions.

``MLStrategy`` path
-------------------
Hardcodes the trained Keras Dense network weights as static MQL5 arrays and
renders a pure-MQL5 forward pass function. No Python runtime or ONNX
dependency is required at execution time.

Typical workflow — StandardStrategy
-------------------------------------
1. Build a ``StandardStrategy`` (with indicators, weights, and thresholds).
2. Call ``export_to_mql5()`` with the desired output path and EA metadata.
3. Open the generated ``.mq5`` file in MetaEditor, compile, attach to a chart.

Typical workflow — MLStrategy
-------------------------------------
1. Train an ``MLStrategy`` with a ``KerasModelWrapper`` model.
2. Call ``export_ml_to_mql5()`` with the desired output path and EA metadata.
3. Open the generated ``.mq5`` file in MetaEditor, compile, attach to a chart.
4. Populate the ``Feature_N_*`` input variables with live indicator values.

Supported indicators (StandardStrategy)
-----------------------------------------
SMA, EMA, WMA, CMA, RSI, MACD, Momentum, LarryWilliams

Supported upstream signals (StandardStrategy)
----------------------------------------------
OHLC, HeikinAshi, CyclicalTemporalSignal

Supported position sizers (both paths)
-----------------------------------------
None (fixed lot input), FixedPositionSizer, RiskBasedPositionSizer

Supported Keras layer types (MLStrategy)
-----------------------------------------
Dense (translated), Dropout (inference no-op), InputLayer, Concatenate

Optional dependency
-------------------
Requires ``Jinja2>=3.1`` (included in the ``[mql5]`` extra):

.. code-block:: bash

    pip install "TradeLab[mql5]"

Example — StandardStrategy
----------------------------
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

Example — MLStrategy
----------------------
>>> from trade_lab.strategies import MLStrategy
>>> from trade_lab.mql5_export import export_ml_to_mql5
>>>
>>> result = export_ml_to_mql5(
...     ml_strategy,
...     output_path="MyML_EA.mq5",
...     magic_number=654321,
... )
>>> print(result.filepath)
>>> for line in result.indicators_exported:   # layer architecture summary
...     print(line)
"""
from __future__ import annotations

try:
    import jinja2  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "The mql5_export module requires Jinja2. "
        "Install it with:  pip install 'TradeLab[mql5]'"
    ) from _exc

from trade_lab.mql5_export.code_generator import MQL5ExportResult, export_ml_to_mql5, export_to_mql5
from trade_lab.mql5_export.introspector import StrategyIntrospector
from trade_lab.mql5_export.ml_introspector import MLStrategyIntrospector
from trade_lab.mql5_export.ml_validator import validate_ml_strategy
from trade_lab.mql5_export.validators import validate_strategy

__all__ = [
    "export_to_mql5",
    "export_ml_to_mql5",
    "MQL5ExportResult",
    "StrategyIntrospector",
    "MLStrategyIntrospector",
    "validate_strategy",
    "validate_ml_strategy",
]