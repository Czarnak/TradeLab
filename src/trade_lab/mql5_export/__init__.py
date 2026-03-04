"""MQL5 Expert Advisor export module for TradeLab.

Translates a TradeLab strategy Python object into a complete, readable,
well-commented MQL5 Expert Advisor (``.mq5`` file) using introspection of the
strategy object tree and Jinja2 templates.

Three export paths are available:

``StandardStrategy`` path
--------------------------
Uses weighted fuzzy-logic aggregation. Indicators are translated to MQL5
built-in handles (``iMA``, ``iRSI``, etc.) and signal-strength functions.

``MLStrategy`` path — hardcoded weights
----------------------------------------
Hardcodes the trained Keras Dense network weights as static MQL5 arrays and
renders a pure-MQL5 forward pass function. No Python runtime or ONNX
dependency is required at execution time.

``MLStrategy`` path — ONNX file reference
------------------------------------------
Converts the Keras model to a ``.onnx`` binary and generates a lightweight
``.mq5`` EA that loads it at runtime via MT5's built-in ``OnnxCreate`` API.
This is the recommended path for non-trivial models. Requires
``pip install 'TradeLab[onnx]'``.

Typical workflow — StandardStrategy
-------------------------------------
1. Build a ``StandardStrategy`` (with indicators, weights, and thresholds).
2. Call ``export_to_mql5()`` with the desired output path and EA metadata.
3. Open the generated ``.mq5`` file in MetaEditor, compile, attach to a chart.

Typical workflow — MLStrategy (hardcoded weights)
--------------------------------------------------
1. Train an ``MLStrategy`` with a ``KerasModelWrapper`` model.
2. Call ``export_ml_to_mql5()`` with the desired output path and EA metadata.
3. Open the generated ``.mq5`` file in MetaEditor, compile, attach to a chart.
4. Populate the ``Feature_N_*`` input variables with live indicator values.

Typical workflow — MLStrategy (ONNX)
--------------------------------------
1. Train an ``MLStrategy`` with a ``KerasModelWrapper`` model.
2. Call ``export_ml_to_mql5_onnx()`` — produces both a ``.mq5`` and a ``.onnx`` file.
3. Copy the ``.onnx`` file to ``<MT5 data folder>\\MQL5\\Files\\``.
4. Open the ``.mq5`` file in MetaEditor, compile, attach to a chart.
5. Populate the ``Feature_*`` input parameters with live indicator values.

Supported indicators (StandardStrategy)
-----------------------------------------
SMA, EMA, WMA, CMA, RSI, MACD, Momentum, LarryWilliams

Supported upstream signals (StandardStrategy)
----------------------------------------------
OHLC, HeikinAshi, CyclicalTemporalSignal

Supported position sizers (all paths)
-----------------------------------------
None (fixed lot input), FixedPositionSizer, RiskBasedPositionSizer

Supported Keras layer types (MLStrategy hardcoded weights)
-----------------------------------------------------------
Dense (translated), Dropout (inference no-op), InputLayer, Concatenate

Supported Keras layer types (MLStrategy ONNX)
----------------------------------------------
Any architecture supported by tf2onnx (Sequential and Functional models).

Optional dependencies
---------------------
MQL5 export requires ``Jinja2>=3.1``:

.. code-block:: bash

    pip install "TradeLab[mql5]"

ONNX export additionally requires ``tf2onnx`` and ``onnx`` (Python 3.10-3.12):

.. code-block:: bash

    pip install "TradeLab[onnx]"

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

Example — MLStrategy (hardcoded weights)
------------------------------------------
>>> from trade_lab.strategies import MLStrategy
>>> from trade_lab.mql5_export import export_ml_to_mql5
>>>
>>> result = export_ml_to_mql5(
...     ml_strategy,
...     output_path="MyML_EA.mq5",
...     magic_number=654321,
... )
>>> print(result.filepath)
>>> for line in result.indicators_exported:
...     print(line)

Example — MLStrategy (ONNX)
------------------------------
>>> from trade_lab.mql5_export import export_ml_to_mql5_onnx
>>>
>>> result = export_ml_to_mql5_onnx(
...     ml_strategy,
...     output_path="MyML_ONNX_EA.mq5",
...     magic_number=654321,
... )
>>> print(result.filepath)       # .mq5 path
>>> print(result.onnx_filepath)  # .onnx path — copy to MQL5\\Files\\
"""

from __future__ import annotations

try:
    import jinja2  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "The mql5_export module requires Jinja2. "
        "Install it with:  pip install 'TradeLab[mql5]'"
    ) from _exc

from trade_lab.mql5_export.code_generator import (
    MQL5ExportResult,
    export_ml_to_mql5,
    export_ml_to_mql5_onnx,
    export_to_mql5,
)
from trade_lab.mql5_export.introspector import StrategyIntrospector
from trade_lab.mql5_export.ml_introspector import MLStrategyIntrospector
from trade_lab.mql5_export.ml_validator import (
    validate_ml_strategy,
    validate_ml_strategy_onnx,
)
from trade_lab.mql5_export.validators import validate_strategy

__all__ = [
    "export_to_mql5",
    "export_ml_to_mql5",
    "export_ml_to_mql5_onnx",
    "MQL5ExportResult",
    "StrategyIntrospector",
    "MLStrategyIntrospector",
    "validate_strategy",
    "validate_ml_strategy",
    "validate_ml_strategy_onnx",
]
