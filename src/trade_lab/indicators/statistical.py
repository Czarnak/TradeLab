from __future__ import annotations

import math
from enum import Enum

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trade_lab.indicators.base import BaseIndicator
from trade_lab.signals.base import BaseSignal, PriceSource


class KernelType(str, Enum):
    TRIANGULAR = "Triangular"
    GAUSSIAN = "Gaussian"
    EPANECHNIKOV = "Epanechnikov"
    LOGISTIC = "Logistic"
    LOG_LOGISTIC = "Log Logistic"
    COSINE = "Cosine"
    SINC = "Sinc"
    LAPLACE = "Laplace"
    QUARTIC = "Quartic"
    PARABOLIC = "Parabolic"
    EXPONENTIAL = "Exponential"
    SILVERMAN = "Silverman"
    CAUCHY = "Cauchy"
    TENT = "Tent"
    WAVE = "Wave"
    POWER = "Power"
    MORTERS = "Morters"


def _coerce_kernel_type(kernel: KernelType | str) -> KernelType:
    if isinstance(kernel, KernelType):
        return kernel

    normalized = kernel.strip().lower().replace("-", " ").replace("_", " ")
    for candidate in KernelType:
        if normalized in {
            candidate.value.lower(),
            candidate.name.lower(),
            candidate.name.lower().replace("_", " "),
        }:
            return candidate

    supported = ", ".join(member.value for member in KernelType)
    raise ValueError(f"Unsupported kernel {kernel!r}. Choose one of: {supported}")


def _gaussian(source: float, bandwidth: float) -> float:
    return math.exp(-((source / bandwidth) ** 2) / 2.0) / math.sqrt(2.0 * math.pi)


def _triangular(source: float, bandwidth: float) -> float:
    scaled = abs(source / bandwidth)
    return 1.0 - scaled if scaled <= 1.0 else 0.0


def _epanechnikov(source: float, bandwidth: float) -> float:
    scaled = source / bandwidth
    return 0.75 * (1.0 - scaled**2) if abs(scaled) <= 1.0 else 0.0


def _quartic(source: float, bandwidth: float) -> float:
    scaled = source / bandwidth
    return (15.0 / 16.0) * (1.0 - scaled**2) ** 2 if abs(scaled) <= 1.0 else 0.0


def _logistic(source: float, bandwidth: float) -> float:
    scaled = source / bandwidth
    return 1.0 / (math.exp(scaled) + 2.0 + math.exp(-scaled))


def _cosine(source: float, bandwidth: float) -> float:
    scaled = source / bandwidth
    return (
        (math.pi / 4.0) * math.cos((math.pi / 2.0) * scaled)
        if abs(scaled) <= 1.0
        else 0.0
    )


def _laplace(source: float, bandwidth: float) -> float:
    scaled = abs(source / bandwidth)
    return (1.0 / (2.0 * bandwidth)) * math.exp(-scaled)


def _exponential(source: float, bandwidth: float) -> float:
    scaled = abs(source / bandwidth)
    return (1.0 / bandwidth) * math.exp(-scaled)


def _silverman(source: float, bandwidth: float) -> float:
    scaled = source / bandwidth
    if abs(scaled) <= 0.5:
        return 0.5 * math.exp(-(scaled) / 2.0) * math.sin(scaled / 2.0 + math.pi / 4.0)
    return 0.0


def _cauchy(source: float, bandwidth: float) -> float:
    scaled = source / bandwidth
    return 1.0 / (math.pi * bandwidth * (1.0 + scaled**2))


def _sinc(source: float, bandwidth: float) -> float:
    if source == 0.0:
        return 1.0
    scaled = math.pi * source / bandwidth
    return math.sin(scaled) / scaled


def _wave(source: float, bandwidth: float) -> float:
    scaled = source / bandwidth
    if abs(scaled) <= 1.0:
        return (1.0 - abs(scaled)) * math.cos((math.pi * source) / bandwidth)
    return 0.0


def _parabolic(source: float, bandwidth: float) -> float:
    scaled = source / bandwidth
    return 1.0 - scaled**2 if abs(scaled) <= 1.0 else 0.0


def _power(source: float, bandwidth: float) -> float:
    scaled = abs(source / bandwidth)
    return (1.0 - scaled**3) ** 3 if scaled <= 1.0 else 0.0


def _log_logistic(source: float, bandwidth: float) -> float:
    return 1.0 / (1.0 + abs(source / bandwidth)) ** 2


def _morters(source: float, bandwidth: float) -> float:
    scaled = source / bandwidth
    if abs(scaled) <= math.pi:
        return (1.0 + math.cos(scaled)) / (2.0 * math.pi * bandwidth)
    return 0.0


KERNEL_FUNCTIONS = {
    KernelType.TRIANGULAR: _triangular,
    KernelType.GAUSSIAN: _gaussian,
    KernelType.EPANECHNIKOV: _epanechnikov,
    KernelType.LOGISTIC: _logistic,
    KernelType.LOG_LOGISTIC: _log_logistic,
    KernelType.COSINE: _cosine,
    KernelType.SINC: _sinc,
    KernelType.LAPLACE: _laplace,
    KernelType.QUARTIC: _quartic,
    KernelType.PARABOLIC: _parabolic,
    KernelType.EXPONENTIAL: _exponential,
    KernelType.SILVERMAN: _silverman,
    KernelType.CAUCHY: _cauchy,
    KernelType.TENT: _triangular,
    KernelType.WAVE: _wave,
    KernelType.POWER: _power,
    KernelType.MORTERS: _morters,
}


class BaseKernel(BaseIndicator):
    """Kernel regression indicator backed by the non-repainting Pine formula."""

    def __init__(
        self,
        *signals: BaseSignal,
        lag: int = 0,
        bandwidth: int = 14,
        deviations: float = 2.0,
        price_source: PriceSource = PriceSource.CLOSE,
        kernel: KernelType | str = KernelType.LAPLACE,
    ) -> None:
        super().__init__(*signals, lag=lag)
        if bandwidth < 1:
            raise ValueError(f"bandwidth must be >= 1, got {bandwidth}")
        if deviations < 0:
            raise ValueError(f"deviations must be >= 0, got {deviations}")

        self.bandwidth = bandwidth
        self.deviations = deviations
        self.price_source = price_source
        self.kernel_type = _coerce_kernel_type(kernel)
        self.plot_title = f"{self.kernel_type.value} Kernel Regression"
        self._weights = self._build_weights()
        self._weight_sum = float(self._weights.sum())
        if self._weight_sum == 0.0:
            raise ValueError("Kernel weights sum to zero; choose different parameters")

    def _price_column(self, df: pd.DataFrame) -> str:
        raw_name = self.price_source.value
        if raw_name in df.columns:
            return raw_name
        title_name = raw_name.capitalize()
        if title_name in df.columns:
            return title_name
        raise KeyError(
            f"Price source {self.price_source.value!r} not found in DataFrame columns"
        )

    def _format_value(self, value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    def _kernel_slug(self) -> str:
        return self.kernel_type.name.lower()

    def _build_weights(self) -> np.ndarray:
        kernel_function = KERNEL_FUNCTIONS[self.kernel_type]
        denominator = float(self.bandwidth * self.bandwidth)
        return np.fromiter(
            (
                kernel_function((index * index) / denominator, 1.0)
                for index in range(self.bandwidth)
            ),
            dtype=float,
            count=self.bandwidth,
        )

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_data(df)
        price = df[self._price_column(df)].astype(float)
        normalized_weights = self._weights / self._weight_sum

        estimate_values = np.convolve(
            price.to_numpy(dtype=float),
            normalized_weights,
            mode="full",
        )[: len(price)]
        if self.bandwidth > 1:
            estimate_values[: self.bandwidth - 1] = np.nan

        estimate = pd.Series(estimate_values, index=df.index)
        residual_sq = (price - estimate).pow(2)

        if self.bandwidth == 1:
            stdev = pd.Series(0.0, index=df.index, dtype=float)
        else:
            stdev = np.sqrt(
                residual_sq.rolling(self.bandwidth).sum() / float(self.bandwidth - 1)
            )
        stdev = stdev * self.deviations

        estimate_col, upper_col, lower_col, stdev_col = self._raw_output_columns
        df[estimate_col] = estimate
        df[upper_col] = estimate + stdev
        df[lower_col] = estimate - stdev
        df[stdev_col] = stdev
        return df

    def to_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        estimate_col, _, _, stdev_col = self.output_columns
        price = df[self._price_column(df)].astype(float)
        estimate = df[estimate_col]

        scale = df[stdev_col].replace(0, np.nan)
        fallback = price.rolling(self.bandwidth).std().replace(0, np.nan)
        normalized = ((price - estimate) / scale.fillna(fallback)).replace(
            [np.inf, -np.inf], np.nan
        )
        return pd.Series(np.tanh(normalized.fillna(0.0)), index=df.index)

    def crossover_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        estimate = df[self.output_columns[0]]
        delta = estimate.diff()
        return pd.DataFrame(
            {
                "bullish": (delta > 0) & (delta.shift(1) < 0),
                "bearish": (delta < 0) & (delta.shift(1) > 0),
            },
            index=df.index,
        )

    def plot(self, df: pd.DataFrame, ax=None) -> None:
        estimate_col, upper_col, lower_col, _ = self.output_columns
        price_column = self._price_column(df)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[price_column],
                mode="lines",
                name=price_column,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[upper_col],
                mode="lines",
                line=dict(dash="dot"),
                name="Upper Band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[lower_col],
                mode="lines",
                line=dict(dash="dot"),
                fill="tonexty",
                fillcolor="rgba(100,149,237,0.12)",
                name="Lower Band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_numpy(),
                y=df[estimate_col],
                mode="lines",
                line=dict(width=3),
                name=self.kernel_type.value,
            )
        )
        fig.update_layout(
            title=self.plot_title,
            xaxis_title="Date",
            yaxis_title="Price",
        )
        fig.show()

    @property
    def _raw_output_columns(self) -> list[str]:
        suffix = (
            f"{self._kernel_slug()}_{self.price_source.value}_{self.bandwidth}_"
            f"{self._format_value(self.deviations)}"
        )
        return [
            f"indicator__kernel_{suffix}",
            f"indicator__kernel_upper_{suffix}",
            f"indicator__kernel_lower_{suffix}",
            f"indicator__kernel_stdev_{suffix}",
        ]


class _FixedKernel(BaseKernel):
    kernel_type: KernelType

    def __init__(
        self,
        *signals: BaseSignal,
        lag: int = 0,
        bandwidth: int = 14,
        deviations: float = 2.0,
        price_source: PriceSource = PriceSource.CLOSE,
    ) -> None:
        super().__init__(
            *signals,
            lag=lag,
            bandwidth=bandwidth,
            deviations=deviations,
            price_source=price_source,
            kernel=self.kernel_type,
        )


class TriangularKernel(_FixedKernel):
    kernel_type = KernelType.TRIANGULAR


class GaussianKernel(_FixedKernel):
    kernel_type = KernelType.GAUSSIAN


class EpanechnikovKernel(_FixedKernel):
    kernel_type = KernelType.EPANECHNIKOV


class LogisticKernel(_FixedKernel):
    kernel_type = KernelType.LOGISTIC


class LogLogisticKernel(_FixedKernel):
    kernel_type = KernelType.LOG_LOGISTIC


class CosineKernel(_FixedKernel):
    kernel_type = KernelType.COSINE


class SincKernel(_FixedKernel):
    kernel_type = KernelType.SINC


class LaplaceKernel(_FixedKernel):
    kernel_type = KernelType.LAPLACE


class QuarticKernel(_FixedKernel):
    kernel_type = KernelType.QUARTIC


class ParabolicKernel(_FixedKernel):
    kernel_type = KernelType.PARABOLIC


class ExponentialKernel(_FixedKernel):
    kernel_type = KernelType.EXPONENTIAL


class SilvermanKernel(_FixedKernel):
    kernel_type = KernelType.SILVERMAN


class CauchyKernel(_FixedKernel):
    kernel_type = KernelType.CAUCHY


class TentKernel(_FixedKernel):
    kernel_type = KernelType.TENT


class WaveKernel(_FixedKernel):
    kernel_type = KernelType.WAVE


class PowerKernel(_FixedKernel):
    kernel_type = KernelType.POWER


class MortersKernel(_FixedKernel):
    kernel_type = KernelType.MORTERS


class SquareKernel(QuarticKernel):
    """Compatibility alias for the squared parabolic kernel variant."""


__all__ = [
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
]
