from __future__ import annotations

import pytest

from trade_lab.indicators.moving_averages import DEMA, TEMA
from trade_lab.indicators.oscillators import (
    BollingerBands,
    CCI,
    DeMarker,
    DPO,
    ROC,
    RVI,
    Stochastic,
    TRIX,
)
from trade_lab.indicators.trend import ADX, ATR, MassIndex
from trade_lab.indicators.volume import CHO, OBV, ForceIndex
from trade_lab.mql5_export.indicator_registry import INDICATOR_REGISTRY


@pytest.mark.parametrize(
    ("indicator_cls", "uses_builtin", "builtin_function", "n_buffers"),
    [
        (DEMA, False, None, 0),
        (TEMA, False, None, 0),
        (BollingerBands, True, "iBands", 3),
        (CCI, True, "iCCI", 1),
        (Stochastic, True, "iStochastic", 2),
        (ROC, False, None, 0),
        (TRIX, False, None, 0),
        (DPO, False, None, 0),
        (RVI, True, "iRVI", 2),
        (DeMarker, True, "iDeMarker", 1),
        (OBV, False, None, 0),
        (ForceIndex, False, None, 0),
        (CHO, True, "iChaikin", 1),
        (ADX, True, "iADX", 3),
        (ATR, True, "iATR", 1),
        (MassIndex, False, None, 0),
    ],
)
def test_indicator_registry_contains_new_indicators(
    indicator_cls,
    uses_builtin,
    builtin_function,
    n_buffers,
):
    assert indicator_cls in INDICATOR_REGISTRY

    descriptor = INDICATOR_REGISTRY[indicator_cls]
    assert descriptor.uses_builtin_handle is uses_builtin
    assert descriptor.builtin_function == builtin_function
    assert descriptor.n_buffers == n_buffers
    assert isinstance(descriptor.signal_strength_formula, str)
    assert descriptor.signal_strength_formula
    assert descriptor.applied_price_map["Close"] == "PRICE_CLOSE"


def test_atr_registry_marks_non_directional_signal():
    descriptor = INDICATOR_REGISTRY[ATR]
    assert "no directional signal" in descriptor.signal_strength_formula.lower()

