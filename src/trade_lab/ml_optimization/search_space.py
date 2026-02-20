"""Search space descriptors for ML-based indicator optimisation.

Each ``IndicatorSpec`` defines a candidate indicator and its hyperparameter
ranges. ``MLObjective`` samples concrete configurations from these specs
during each Optuna trial.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndicatorSpec:
    """Descriptor for one candidate indicator in the search space.

    Parameters
    ----------
    name : str
        Unique identifier used as Optuna parameter prefix (e.g. ``'ema'``).
        Must be a valid Python identifier.
    indicator_class : type
        Indicator class (e.g. ``EMA``, ``SMA``, ``RSI``). Instantiated per
        trial with the sampled ``period``.
    period_low : int
        Lower bound (inclusive) of the period search range.
    period_high : int
        Upper bound (inclusive) of the period search range.
    lag_low : int
        Minimum lag value (0 = unshifted column).
    lag_high : int
        Maximum lag value.
    max_lags : int
        Maximum number of distinct lag values to sample per trial.
    optional : bool
        If ``True``, Optuna may exclude this indicator from a trial.
    """

    name: str
    indicator_class: type
    period_low: int
    period_high: int
    lag_low: int
    lag_high: int
    max_lags: int = 5
    optional: bool = True

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(
                f"name must be a valid Python identifier, got {self.name!r}"
            )
        if self.period_low >= self.period_high:
            raise ValueError(
                f"period_low ({self.period_low}) must be < "
                f"period_high ({self.period_high})"
            )
        if self.lag_low < 0:
            raise ValueError(f"lag_low must be >= 0, got {self.lag_low}")
        if self.lag_low >= self.lag_high:
            raise ValueError(
                f"lag_low ({self.lag_low}) must be < lag_high ({self.lag_high})"
            )
        if self.max_lags < 1:
            raise ValueError(f"max_lags must be >= 1, got {self.max_lags}")
