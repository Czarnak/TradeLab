"""Statistical analysis of Monte Carlo simulation results.

Provides raw distributional statistics over simulated metric values.
No verdict or threshold logic is applied — interpretation is left to the user,
who may be evaluating robustness in very different contexts (live trading,
research, regime-specific analysis, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_lab.monte_carlo.runner import MonteCarloResult


class MonteCarloAnalysis:
    """Statistical analysis of Monte Carlo simulation results.

    All methods return raw numbers. The ``percentile_of`` method is the most
    useful single-number anti-overfitting check: it answers "at what percentile
    of the simulated distribution does my real backtest result fall?" A very
    high percentile (e.g. above 95) suggests the strategy's edge may be
    partially fitted to the specific historical path rather than structural.

    Parameters
    ----------
    result : MonteCarloResult
        Output of ``MonteCarloRunner.run()``.
    """

    def __init__(self, result: MonteCarloResult):
        self.result = result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(
        self,
        percentiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Descriptive statistics for every metric across all simulations.

        Parameters
        ----------
        percentiles : list[float] | None
            Percentile points to include (values in 0–100).
            Defaults to [5, 25, 50, 75, 95].

        Returns
        -------
        pd.DataFrame
            Index = metric names.
            Columns = mean, std, min, p{N} for each percentile, max,
            n_valid (number of finite values), n_nan (number of NaNs).
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        rows: dict[str, dict] = {}
        for metric, values in self.result.metric_series.items():
            rows[metric] = self._describe(values, percentiles)

        return pd.DataFrame(rows).T

    def distributions(self) -> dict[str, np.ndarray]:
        """Raw metric value arrays for each metric.

        Returns
        -------
        dict[str, np.ndarray]
            metric_name → float array of length n_simulations.
            May contain NaN for simulations that produced no trades or
            otherwise returned None for that metric.
        """
        return {
            metric: np.array(values, dtype=float)
            for metric, values in self.result.metric_series.items()
        }

    def confidence_interval(
        self,
        metric: str,
        lower: float = 5.0,
        upper: float = 95.0,
    ) -> tuple[float, float]:
        """Percentile-based confidence interval for a single metric.

        Parameters
        ----------
        metric : str
            Metric name (must exist in result.metric_series).
        lower : float
            Lower percentile bound, in 0–100. Default 5 (i.e. 5th percentile).
        upper : float
            Upper percentile bound, in 0–100. Default 95.

        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound). Both are NaN if no finite values exist.

        Raises
        ------
        KeyError
            If the metric name is not found in the results.
        """
        arr = self._get_finite(metric)
        if len(arr) == 0:
            return float('nan'), float('nan')
        return float(np.percentile(arr, lower)), float(np.percentile(arr, upper))

    def percentile_of(self, metric: str, value: float) -> float:
        """Percentile rank of a specific value within the simulated distribution.

        Answers: "If my real backtest achieved this metric value, where does
        it sit relative to all simulated (counterfactual) market paths?"

        Interpretation guide (for metrics where higher = better, e.g. Sharpe):
            < 50th percentile  → strategy underperforms the average synthetic path.
            50–75th percentile → moderate edge, may be partially path-dependent.
            75–95th percentile → strategy is robust across most synthetic paths.
            > 95th percentile  → very strong or possibly over-fitted; investigate.

        For metrics where lower = better (e.g. max_drawdown), invert the
        interpretation: a very low percentile indicates robustness.

        Parameters
        ----------
        metric : str
            Metric name.
        value : float
            The value to locate in the distribution (e.g., real backtest Sharpe).

        Returns
        -------
        float
            Percentile rank in [0, 100]. NaN if no finite simulation values exist.

        Raises
        ------
        KeyError
            If the metric name is not found in the results.
        """
        arr = self._get_finite(metric)
        if len(arr) == 0:
            return float('nan')
        return float(np.mean(arr <= value) * 100)

    def available_metrics(self) -> list[str]:
        """List of metric names present in the results."""
        return list(self.result.metric_series.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_finite(self, metric: str) -> np.ndarray:
        """Return finite values for a metric, raising KeyError if not found."""
        if metric not in self.result.metric_series:
            available = list(self.result.metric_series.keys())
            raise KeyError(
                f"Metric '{metric}' not found in results. "
                f"Available metrics: {available}"
            )
        arr = np.array(self.result.metric_series[metric], dtype=float)
        return arr[np.isfinite(arr)]

    def _describe(
        self,
        values: list[float],
        percentiles: list[float],
    ) -> dict:
        """Compute descriptive statistics for a list of metric values."""
        arr = np.array(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        n_nan = int(np.sum(~np.isfinite(arr)))

        if len(finite) == 0:
            row: dict = {'mean': float('nan'), 'std': float('nan'), 'min': float('nan')}
            for p in percentiles:
                row[f'p{int(p)}'] = float('nan')
            row['max'] = float('nan')
            row['n_valid'] = 0
            row['n_nan'] = n_nan
            return row

        row = {
            'mean': float(np.mean(finite)),
            'std': float(np.std(finite)),
            'min': float(np.min(finite)),
        }
        for p in percentiles:
            row[f'p{int(p)}'] = float(np.percentile(finite, p))
        row['max'] = float(np.max(finite))
        row['n_valid'] = len(finite)
        row['n_nan'] = n_nan

        return row