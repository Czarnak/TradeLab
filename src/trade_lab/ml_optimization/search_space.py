"""Search space descriptors for ML-based indicator optimisation.

Each ``IndicatorSpec`` defines a candidate indicator and its hyperparameter
ranges. ``MLObjective`` samples concrete configurations from these specs
during each Optuna trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IndicatorSpec:
    """Descriptor for one candidate indicator in the search space.

    Each trial samples one ``period`` value and one ``lag`` value per
    included indicator. The sampled lag is passed directly to the
    indicator constructor (e.g. ``EMA(period=20, lag=3)``).

    Parameters
    ----------
    name : str
        Unique identifier used as Optuna parameter prefix (e.g. ``'ema'``).
        Must be a valid Python identifier.
    indicator_class : type
        Indicator class (e.g. ``EMA``, ``SMA``, ``RSI``). Instantiated per
        trial with the sampled ``period`` and ``lag``.
    period_low : int
        Lower bound (inclusive) of the period search range.
    period_high : int
        Upper bound (inclusive) of the period search range.
    lag_values : list[int]
        Candidate lag values Optuna may choose from.  Must include at least
        one value; ``[0]`` means no lag is applied.  Optuna samples one
        value per trial using ``suggest_categorical``.
    optional : bool
        If ``True``, Optuna may exclude this indicator from a trial entirely.

    Examples
    --------
    >>> IndicatorSpec('ema', EMA, period_low=5, period_high=50,
    ...               lag_values=[0, 1, 2, 5], optional=False)
    >>> IndicatorSpec('rsi', RSI, period_low=7, period_high=28,
    ...               lag_values=[0], optional=True)
    """

    name: str
    indicator_class: type
    period_low: int
    period_high: int
    lag_values: list[int] = field(default_factory=lambda: [0])
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
        if not self.lag_values:
            raise ValueError("lag_values must contain at least one value.")
        if any(v < 0 for v in self.lag_values):
            raise ValueError("All lag_values must be >= 0.")