"""Parameter space descriptors for strategy optimisation.

Each descriptor defines a single tuneable parameter: its name, type, and
the range or set of values Optuna is allowed to sample from.

These objects are intentionally data-only (no logic). The translation from
descriptor → Optuna trial suggestion lives in ``objective.py``, keeping this
module dependency-free and easily unit-testable.

Usage example
-------------
>>> from trade_lab.optimization.param_space import IntParam, FloatParam, CategoricalParam
>>>
>>> param_space = [
...     IntParam('fast_period', low=5,   high=50),
...     IntParam('slow_period', low=20,  high=200, step=5),
...     FloatParam('weight_ema', low=0.1, high=2.0),
...     FloatParam('weight_rsi', low=0.1, high=2.0),
...     CategoricalParam('allow_short', choices=[True, False]),
... ]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntParam:
    """Integer parameter sampled uniformly over [low, high].

    Parameters
    ----------
    name : str
        Parameter name. Used as the key in the params dict passed to the
        strategy factory. Must be unique within a param space list.
    low : int
        Inclusive lower bound.
    high : int
        Inclusive upper bound.
    step : int
        Step size between candidate values. Default 1 (every integer).
        Useful for parameters with natural granularity, e.g. periods in
        multiples of 5.

    Examples
    --------
    >>> IntParam('rsi_period', low=7, high=28)          # 7, 8, ..., 28
    >>> IntParam('slow_period', low=20, high=200, step=5)  # 20, 25, ..., 200
    """

    name: str
    low: int
    high: int
    step: int = 1


@dataclass
class FloatParam:
    """Floating-point parameter sampled over [low, high].

    Parameters
    ----------
    name : str
        Parameter name.
    low : float
        Inclusive lower bound.
    high : float
        Inclusive upper bound.
    log : bool
        If True, values are sampled on a log scale. Useful for parameters
        that span several orders of magnitude where small values matter
        proportionally more (e.g. learning rates, regularisation strengths).
        Both ``low`` and ``high`` must be strictly positive when log=True.

    Examples
    --------
    >>> FloatParam('weight_ema', low=0.1, high=3.0)
    >>> FloatParam('learning_rate', low=1e-5, high=1e-1, log=True)
    """

    name: str
    low: float
    high: float
    log: bool = False


@dataclass
class CategoricalParam:
    """Parameter sampled from a fixed set of discrete choices.

    The choices list may contain any type that is hashable and can be stored
    in an Optuna trial: str, int, float, bool. Mixed types in a single
    CategoricalParam are supported by Optuna but discouraged — use separate
    params instead for clarity.

    Parameters
    ----------
    name : str
        Parameter name.
    choices : list[Any]
        Ordered list of candidate values. Must contain at least two elements.

    Examples
    --------
    >>> CategoricalParam('allow_short', choices=[True, False])
    >>> CategoricalParam('signal_type', choices=['ohlc', 'heikin_ashi'])
    """

    name: str
    choices: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(
                f"CategoricalParam '{self.name}' must have at least 2 choices, "
                f"got {len(self.choices)}."
            )


# Type alias for a mixed param space list — used in type hints throughout
# the optimisation module.
ParamSpace = list[IntParam | FloatParam | CategoricalParam]