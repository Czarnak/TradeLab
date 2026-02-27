from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.indicators import (
    ADX,
    ATR,
    CHO,
    CCI,
    DEMA,
    DPO,
    OBV,
    ROC,
    RVI,
    TEMA,
    TRIX,
    BollingerBands,
    DeMarker,
    ForceIndex,
    MassIndex,
    Stochastic,
)


def _sample_ohlcv(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    base = np.linspace(100.0, 120.0, n)
    wave = np.sin(np.arange(n) / 4.0) * 2.0
    close = base + wave
    open_ = close + np.sin(np.arange(n) / 6.0) * 0.5
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1000 + (np.arange(n) % 11) * 40
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


def test_dema_compute_matches_formula():
    df = _sample_ohlcv()
    period = 10
    indicator = DEMA(period=period)

    out = indicator.compute(df.copy())
    col = indicator.output_columns[0]

    ema = df["Close"].ewm(span=period, adjust=False).mean()
    expected = 2.0 * ema - ema.ewm(span=period, adjust=False).mean()

    pd.testing.assert_series_equal(
        out[col],
        expected.rename(col),
        check_names=True,
    )


def test_tema_compute_matches_formula():
    df = _sample_ohlcv()
    period = 10
    indicator = TEMA(period=period)

    out = indicator.compute(df.copy())
    col = indicator.output_columns[0]

    ema1 = df["Close"].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    expected = 3.0 * ema1 - 3.0 * ema2 + ema3

    pd.testing.assert_series_equal(
        out[col],
        expected.rename(col),
        check_names=True,
    )


@pytest.mark.parametrize(
    ("indicator", "expected_columns"),
    [
        (BollingerBands(period=20, num_deviations=2.0), [
            "indicator__bb_upper_20",
            "indicator__bb_middle_20",
            "indicator__bb_lower_20",
        ]),
        (CCI(period=14), ["indicator__cci_14"]),
        (Stochastic(k_period=14, d_period=3, slowing=3), [
            "indicator__stoch_k_14",
            "indicator__stoch_d_14",
        ]),
        (ROC(period=12), ["indicator__roc_12"]),
        (TRIX(period=14), ["indicator__trix_14"]),
        (DPO(period=20), ["indicator__dpo_20"]),
        (RVI(period=10), ["indicator__rvi_10", "indicator__rvi_signal_10"]),
        (DeMarker(period=14), ["indicator__demarker_14"]),
        (OBV(period=14), ["indicator__obv_roc_14"]),
        (ForceIndex(period=13), ["indicator__force_13"]),
        (CHO(fast_period=3, slow_period=10), ["indicator__cho_3_10"]),
        (ADX(period=14), [
            "indicator__adx_14",
            "indicator__adx_plus_di_14",
            "indicator__adx_minus_di_14",
        ]),
        (MassIndex(ema_period=9, sum_period=25), ["indicator__mi_9_25"]),
    ],
)
def test_new_indicator_compute_and_signal_strength(indicator, expected_columns):
    df = _sample_ohlcv()

    out = indicator.compute(df.copy())
    assert indicator.output_columns == expected_columns
    for col in expected_columns:
        assert col in out.columns

    strength = indicator.to_signal_strength(out)
    assert len(strength) == len(df)
    assert strength.notna().sum() > 0
    assert np.isfinite(strength.dropna().to_numpy()).all()


def test_atr_compute_and_signal_strength_not_implemented():
    df = _sample_ohlcv()
    indicator = ATR(period=14)

    out = indicator.compute(df.copy())
    assert indicator.output_columns == ["indicator__atr_14"]
    assert "indicator__atr_14" in out.columns

    with pytest.raises(NotImplementedError, match="volatility utility indicator"):
        indicator.to_signal_strength(out)


def test_indicators_package_exports_new_classes():
    import trade_lab.indicators as indicators

    for name in [
        "DEMA",
        "TEMA",
        "BollingerBands",
        "CCI",
        "Stochastic",
        "ROC",
        "TRIX",
        "DPO",
        "RVI",
        "DeMarker",
        "OBV",
        "ForceIndex",
        "CHO",
        "ADX",
        "ATR",
        "MassIndex",
    ]:
        assert hasattr(indicators, name)

