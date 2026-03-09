from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trade_lab.indicators import (
    BaseKernel,
    CauchyKernel,
    CosineKernel,
    EpanechnikovKernel,
    ExponentialKernel,
    GaussianKernel,
    KernelType,
    LaplaceKernel,
    LogLogisticKernel,
    LogisticKernel,
    MortersKernel,
    ParabolicKernel,
    PowerKernel,
    QuarticKernel,
    SilvermanKernel,
    SincKernel,
    SquareKernel,
    TentKernel,
    TriangularKernel,
    WaveKernel,
)
from trade_lab.signals.base import PriceSource


def _sample_ohlcv(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = 100.0 + np.linspace(0.0, 9.0, n) + np.sin(np.arange(n) / 3.5) * 1.8
    open_ = close + np.cos(np.arange(n) / 5.0) * 0.35
    high = np.maximum(open_, close) + 1.1
    low = np.minimum(open_, close) - 1.1
    volume = 1_000 + (np.arange(n) % 9) * 25
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


def _kernel_value(kernel_type: KernelType, source: float, bandwidth: float) -> float:
    scaled = source / bandwidth

    if kernel_type in {KernelType.TRIANGULAR, KernelType.TENT}:
        return 1.0 - abs(scaled) if abs(scaled) <= 1.0 else 0.0
    if kernel_type is KernelType.GAUSSIAN:
        return math.exp(-(scaled**2) / 2.0) / math.sqrt(2.0 * math.pi)
    if kernel_type is KernelType.EPANECHNIKOV:
        return 0.75 * (1.0 - scaled**2) if abs(scaled) <= 1.0 else 0.0
    if kernel_type is KernelType.LOGISTIC:
        return 1.0 / (math.exp(scaled) + 2.0 + math.exp(-scaled))
    if kernel_type is KernelType.LOG_LOGISTIC:
        return 1.0 / (1.0 + abs(scaled)) ** 2
    if kernel_type is KernelType.COSINE:
        return (
            (math.pi / 4.0) * math.cos((math.pi / 2.0) * scaled)
            if abs(scaled) <= 1.0
            else 0.0
        )
    if kernel_type is KernelType.SINC:
        if source == 0.0:
            return 1.0
        angle = math.pi * source / bandwidth
        return math.sin(angle) / angle
    if kernel_type is KernelType.LAPLACE:
        return (1.0 / (2.0 * bandwidth)) * math.exp(-abs(scaled))
    if kernel_type is KernelType.QUARTIC:
        return (15.0 / 16.0) * (1.0 - scaled**2) ** 2 if abs(scaled) <= 1.0 else 0.0
    if kernel_type is KernelType.PARABOLIC:
        return 1.0 - scaled**2 if abs(scaled) <= 1.0 else 0.0
    if kernel_type is KernelType.EXPONENTIAL:
        return (1.0 / bandwidth) * math.exp(-abs(scaled))
    if kernel_type is KernelType.SILVERMAN:
        if abs(scaled) <= 0.5:
            return (
                0.5 * math.exp(-(scaled) / 2.0) * math.sin(scaled / 2.0 + math.pi / 4.0)
            )
        return 0.0
    if kernel_type is KernelType.CAUCHY:
        return 1.0 / (math.pi * bandwidth * (1.0 + scaled**2))
    if kernel_type is KernelType.WAVE:
        return (
            (1.0 - abs(scaled)) * math.cos(math.pi * scaled)
            if abs(scaled) <= 1.0
            else 0.0
        )
    if kernel_type is KernelType.POWER:
        return (1.0 - abs(scaled) ** 3) ** 3 if abs(scaled) <= 1.0 else 0.0
    if kernel_type is KernelType.MORTERS:
        return (
            (1.0 + math.cos(scaled)) / (2.0 * math.pi * bandwidth)
            if abs(scaled) <= math.pi
            else 0.0
        )

    raise AssertionError(f"Unhandled kernel type: {kernel_type}")


def _expected_weights(kernel_type: KernelType, bandwidth: int) -> np.ndarray:
    denominator = float(bandwidth * bandwidth)
    return np.array(
        [
            _kernel_value(kernel_type, (index * index) / denominator, 1.0)
            for index in range(bandwidth)
        ],
        dtype=float,
    )


def _expected_estimate(
    series: pd.Series, kernel_type: KernelType, bandwidth: int
) -> pd.Series:
    weights = _expected_weights(kernel_type, bandwidth)
    normalized_weights = weights / weights.sum()
    values = np.convolve(series.to_numpy(dtype=float), normalized_weights, mode="full")[
        : len(series)
    ]
    if bandwidth > 1:
        values[: bandwidth - 1] = np.nan
    return pd.Series(values, index=series.index)


def _expected_stdev(
    series: pd.Series, estimate: pd.Series, bandwidth: int, deviations: float
) -> pd.Series:
    if bandwidth == 1:
        return pd.Series(0.0, index=series.index, dtype=float)
    residual_sq = (series - estimate).pow(2)
    return (
        np.sqrt(residual_sq.rolling(bandwidth).sum() / float(bandwidth - 1))
        * deviations
    )


KERNEL_CASES = [
    (TriangularKernel, KernelType.TRIANGULAR),
    (GaussianKernel, KernelType.GAUSSIAN),
    (EpanechnikovKernel, KernelType.EPANECHNIKOV),
    (LogisticKernel, KernelType.LOGISTIC),
    (LogLogisticKernel, KernelType.LOG_LOGISTIC),
    (CosineKernel, KernelType.COSINE),
    (SincKernel, KernelType.SINC),
    (LaplaceKernel, KernelType.LAPLACE),
    (QuarticKernel, KernelType.QUARTIC),
    (ParabolicKernel, KernelType.PARABOLIC),
    (ExponentialKernel, KernelType.EXPONENTIAL),
    (SilvermanKernel, KernelType.SILVERMAN),
    (CauchyKernel, KernelType.CAUCHY),
    (TentKernel, KernelType.TENT),
    (WaveKernel, KernelType.WAVE),
    (PowerKernel, KernelType.POWER),
    (MortersKernel, KernelType.MORTERS),
]


@pytest.mark.parametrize(("indicator_cls", "kernel_type"), KERNEL_CASES)
def test_kernel_indicator_compute_matches_expected_formula(indicator_cls, kernel_type):
    df = _sample_ohlcv()
    bandwidth = 5
    deviations = 1.75
    indicator = indicator_cls(bandwidth=bandwidth, deviations=deviations)

    out = indicator.compute(df.copy())
    estimate_col, upper_col, lower_col, stdev_col = indicator.output_columns

    expected_estimate = _expected_estimate(df["Close"], kernel_type, bandwidth)
    expected_stdev = _expected_stdev(
        df["Close"], expected_estimate, bandwidth, deviations
    )

    pd.testing.assert_series_equal(
        out[estimate_col],
        expected_estimate.rename(estimate_col),
        atol=1e-12,
        rtol=1e-12,
        check_names=True,
    )
    pd.testing.assert_series_equal(
        out[stdev_col],
        expected_stdev.rename(stdev_col),
        atol=1e-12,
        rtol=1e-12,
        check_names=True,
    )
    pd.testing.assert_series_equal(
        out[upper_col],
        (expected_estimate + expected_stdev).rename(upper_col),
        atol=1e-12,
        rtol=1e-12,
        check_names=True,
    )
    pd.testing.assert_series_equal(
        out[lower_col],
        (expected_estimate - expected_stdev).rename(lower_col),
        atol=1e-12,
        rtol=1e-12,
        check_names=True,
    )


def test_base_kernel_uses_price_source_with_repo_title_case_columns():
    df = _sample_ohlcv()
    indicator = BaseKernel(
        bandwidth=4,
        deviations=1.5,
        price_source=PriceSource.HIGH,
        kernel="gaussian",
    )

    out = indicator.compute(df.copy())
    estimate_col = indicator.output_columns[0]
    expected = _expected_estimate(df["High"], KernelType.GAUSSIAN, 4)

    pd.testing.assert_series_equal(out[estimate_col], expected.rename(estimate_col))


def test_base_kernel_lag_suffixes_and_shifts_all_outputs():
    df = _sample_ohlcv()
    reference = GaussianKernel(bandwidth=4, deviations=1.5)
    lagged = GaussianKernel(bandwidth=4, deviations=1.5, lag=2)

    reference_out = reference.compute(df.copy())
    lagged_out = lagged.compute(df.copy())

    for raw_col, final_col in zip(reference.output_columns, lagged.output_columns):
        assert raw_col not in lagged_out.columns
        pd.testing.assert_series_equal(
            lagged_out[final_col],
            reference_out[raw_col].shift(2).rename(final_col),
            check_names=True,
        )


def test_base_kernel_signal_strength_and_crossover_flags_are_usable():
    df = _sample_ohlcv()
    indicator = LaplaceKernel(bandwidth=6, deviations=2.0)

    out = indicator.compute(df.copy())
    strength = indicator.to_signal_strength(out)
    flags = indicator.crossover_flags(out)

    assert len(strength) == len(df)
    assert np.isfinite(strength.to_numpy()).all()
    assert list(flags.columns) == ["bullish", "bearish"]
    assert flags.dtypes.to_list() == [bool, bool]
    assert flags.any(axis=1).sum() > 0


def test_square_kernel_alias_matches_quartic_kernel():
    df = _sample_ohlcv()
    square = SquareKernel(bandwidth=5, deviations=1.25)
    quartic = QuarticKernel(bandwidth=5, deviations=1.25)

    square_out = square.compute(df.copy())
    quartic_out = quartic.compute(df.copy())

    pd.testing.assert_frame_equal(
        square_out[square.output_columns],
        quartic_out[quartic.output_columns].set_axis(square.output_columns, axis=1),
    )


def test_base_kernel_rejects_unknown_kernel_name():
    with pytest.raises(ValueError, match="Unsupported kernel"):
        BaseKernel(kernel="unknown")


def test_indicators_package_exports_statistical_classes():
    import trade_lab.indicators as indicators

    for name in [
        "KernelType",
        "BaseKernel",
        "TriangularKernel",
        "GaussianKernel",
        "EpanechnikovKernel",
        "LogisticKernel",
        "LogLogisticKernel",
        "CosineKernel",
        "SincKernel",
        "LaplaceKernel",
        "QuarticKernel",
        "ParabolicKernel",
        "ExponentialKernel",
        "SilvermanKernel",
        "CauchyKernel",
        "TentKernel",
        "WaveKernel",
        "PowerKernel",
        "MortersKernel",
        "SquareKernel",
    ]:
        assert hasattr(indicators, name)
