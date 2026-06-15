"""Statistical-significance tools for honest strategy evaluation.

A backtest Sharpe is a point estimate from one noisy sample, usually the *best*
of many threshold/parameter trials. These helpers quantify how much of it could
be luck:

* :func:`probabilistic_sharpe_ratio` — P(true Sharpe > benchmark) given sample
  length, skew and kurtosis.
* :func:`expected_max_sharpe` / :func:`deflated_sharpe_ratio` — discount the
  observed Sharpe for the number of trials run (multiple-testing / selection bias).
* :func:`block_bootstrap_sharpe_ci` — a confidence interval for the Sharpe that
  preserves serial dependence via circular block resampling.
* :func:`regime_breakdown` — metrics split by market regime.
* :func:`promotion_gates` / :func:`min_trade_gate` — explicit pass/fail checks
  against baselines and a minimum-evidence floor.

References
----------
Bailey, D. & Lopez de Prado, M. (2012/2014): "The Sharpe Ratio Efficient
Frontier" and "The Deflated Sharpe Ratio".
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import NamedTuple

import numpy as np
import pandas as pd

_NORM = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329
_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Return values
# ---------------------------------------------------------------------------


class SharpeCI(NamedTuple):
    sharpe: float
    ci_low: float
    ci_high: float
    confidence: float


class GateResult(NamedTuple):
    n_trades: int
    min_trades: int
    passed: bool


# ---------------------------------------------------------------------------
# Probabilistic / deflated Sharpe
# ---------------------------------------------------------------------------


def probabilistic_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability the true Sharpe exceeds ``benchmark_sr``.

    All Sharpe inputs must share the **same frequency** as ``n_obs`` (e.g. per-bar
    Sharpe with a per-bar sample count) — do not pass an annualised Sharpe here.

    Parameters
    ----------
    observed_sr : float
        Estimated (non-annualised) Sharpe ratio.
    n_obs : int
        Number of return observations (``>= 2``).
    benchmark_sr : float
        Sharpe to beat (default ``0``).
    skew : float
        Skewness of the returns (``0`` for normal).
    kurtosis : float
        Non-excess kurtosis of the returns (``3`` for normal).
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")

    denom = math.sqrt(
        1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2
    )
    if denom == 0:
        raise ValueError("degenerate moment combination (zero denominator)")

    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / denom
    return _NORM.cdf(z)


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent trials under the null.

    This is the multiple-testing hurdle: with many trials the *best* Sharpe is
    positive by chance alone. Returns ``0`` when ``n_trials <= 1`` (no selection).
    """
    if sr_variance < 0:
        raise ValueError("sr_variance must be >= 0")
    if n_trials <= 1:
        return 0.0

    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * (
        (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    )


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    sr_variance: float,
    n_trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe deflated by the multiple-testing hurdle.

    Equivalent to :func:`probabilistic_sharpe_ratio` with the benchmark set to the
    expected maximum Sharpe across ``n_trials`` (see :func:`expected_max_sharpe`),
    so it answers "is the *selected* Sharpe better than the best we'd expect from
    luck across all trials?".
    """
    hurdle = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(
        observed_sr, n_obs, benchmark_sr=hurdle, skew=skew, kurtosis=kurtosis
    )


# ---------------------------------------------------------------------------
# Block-bootstrap Sharpe CI
# ---------------------------------------------------------------------------


def _annualised_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    sd = returns.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(returns.mean() / sd * math.sqrt(periods_per_year))


def block_bootstrap_sharpe_ci(
    returns,
    block_size: int,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    periods_per_year: int = _TRADING_DAYS,
) -> SharpeCI:
    """Circular block-bootstrap confidence interval for the annualised Sharpe.

    Resampling in blocks preserves the short-range serial dependence that an
    i.i.d. bootstrap would destroy. ``sharpe`` is the point estimate on the
    original series; ``ci_low``/``ci_high`` are bootstrap percentiles.

    Parameters
    ----------
    returns : array-like
        Per-period returns.
    block_size : int
        Length of each resampled block (``>= 1``).
    n_boot : int
        Number of bootstrap resamples.
    confidence : float
        Two-sided coverage, in ``(0, 1)`` (e.g. ``0.95``).
    seed : int | None
        Seed for reproducibility.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        raise ValueError("need at least 2 finite returns")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    point = _annualised_sharpe(r, periods_per_year)

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_size)
    sharpes = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        # Circular blocks: wrap indices so every start is valid.
        idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel() % n
        sample = r[idx[:n]]
        sharpes[b] = _annualised_sharpe(sample, periods_per_year)

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(sharpes, alpha))
    ci_high = float(np.quantile(sharpes, 1.0 - alpha))
    return SharpeCI(sharpe=point, ci_low=ci_low, ci_high=ci_high, confidence=confidence)


# ---------------------------------------------------------------------------
# Regime breakdown
# ---------------------------------------------------------------------------


def regime_breakdown(
    returns: pd.Series,
    regimes: pd.Series,
    periods_per_year: int = _TRADING_DAYS,
) -> pd.DataFrame:
    """Per-regime metrics: count, mean return, and annualised Sharpe.

    ``returns`` and ``regimes`` are aligned on their shared index. A strategy that
    only works in one regime is exposed here rather than hidden in the blended
    aggregate.
    """
    aligned = pd.DataFrame({"ret": returns, "regime": regimes}).dropna()
    rows = []
    for label, group in aligned.groupby("regime"):
        vals = group["ret"].to_numpy()
        rows.append(
            {
                "regime": label,
                "n": len(vals),
                "mean_return": float(np.mean(vals)),
                "sharpe": _annualised_sharpe(vals, periods_per_year),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------


def promotion_gates(
    strategy_value: float,
    baselines: dict[str, float],
    min_margin: float = 0.0,
) -> pd.DataFrame:
    """Pass/fail table comparing the strategy metric against each baseline.

    A row passes when ``strategy_value > baseline_value + min_margin``. Use a
    risk-adjusted metric (e.g. Sharpe) and baselines such as buy-and-hold,
    always-long, momentum and volatility-scaled.
    """
    rows = [
        {
            "baseline": name,
            "baseline_value": value,
            "strategy_value": strategy_value,
            "passed": bool(strategy_value > value + min_margin),
        }
        for name, value in baselines.items()
    ]
    return pd.DataFrame(
        rows, columns=["baseline", "baseline_value", "strategy_value", "passed"]
    )


def min_trade_gate(n_trades: int, min_trades: int) -> GateResult:
    """Flag results with too few trades to support an inference."""
    return GateResult(
        n_trades=n_trades,
        min_trades=min_trades,
        passed=bool(n_trades >= min_trades),
    )
