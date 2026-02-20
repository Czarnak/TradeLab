"""Monte Carlo simulation runner.

Coordinates N simulation runs: for each run a synthetic OHLCV DataFrame is
generated, the strategy produces signals on it, and the BacktestEngine
simulates trading. Raw metric distributions are collected for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trade_lab.backtesting.engine import BacktestEngine
from trade_lab.monte_carlo.generators import BaseGenerator

# tqdm is an optional dependency — progress display degrades gracefully if absent.
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


@dataclass
class MonteCarloResult:
    """Raw output of a Monte Carlo simulation run.

    Attributes
    ----------
    n_simulations : int
        Number of simulations completed.
    metric_series : dict[str, list[float]]
        Mapping of metric name to a list of values across all simulations.
        Length of each list equals n_simulations (NaN for failed runs).
    generator_name : str
        Class name of the generator used, for traceability.
    """

    n_simulations: int
    metric_series: dict[str, list[float]] = field(default_factory=dict)
    generator_name: str = ''


class MonteCarloRunner:
    """Runs N Monte Carlo backtests on synthetic OHLCV data.

    For each simulation the runner:
      1. Creates a per-simulation generator with a deterministic but varied seed.
      2. Calls ``generator.generate(df)`` to produce a synthetic OHLCV DataFrame.
      3. Calls ``engine.run_on(synthetic_df)`` to run the full backtest pipeline.
      4. Collects the resulting metrics.

    The base ``engine`` provides the strategy and simulation parameters
    (commission, slippage, initial_capital). The data source is replaced by
    the synthetic DataFrame for every simulation.

    Parameters
    ----------
    engine : BacktestEngine
        A fully configured backtest engine. Its ``strategy`` and financial
        parameters are reused for every simulation.
    generator : BaseGenerator
        The Monte Carlo data generator. A fresh seed-varied copy is created
        per simulation so results are independent yet reproducible.
    n_simulations : int
        Number of synthetic scenarios to generate and evaluate.
    metrics : list[str] | None
        Subset of metric keys to collect from each backtest result.
        If None, all metrics produced by ``compute_metrics`` are collected.
    verbose : bool
        If True, display a tqdm progress bar during the simulation run.
        Requires ``tqdm`` to be installed (``pip install tqdm``). If tqdm
        is not available and ``verbose=True``, a plain-text fallback is used
        that prints progress to stdout every 10% of simulations completed.
        Defaults to True.

    Examples
    --------
    >>> runner = MonteCarloRunner(engine, BlockBootstrap(seed=42), n_simulations=500)
    >>> mc_result = runner.run(original_df)
    >>> analysis = MonteCarloAnalysis(mc_result)
    >>> print(analysis.summary())
    """

    def __init__(
        self,
        engine: BacktestEngine,
        generator: BaseGenerator,
        n_simulations: int = 1000,
        metrics: list[str] | None = None,
        verbose: bool = True,
    ):
        self.engine = engine
        self.generator = generator
        self.n_simulations = n_simulations
        self.metrics = metrics
        self.verbose = verbose

    def run(self, df: pd.DataFrame) -> MonteCarloResult:
        """Run all simulations against the provided original DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Original OHLCV DataFrame. Must have columns: Open, High, Low,
            Close, Volume. Typically the output of ``BacktestEngine._fetch_data()``
            or any compatible source (e.g. a CSV loaded into a DataFrame).

        Returns
        -------
        MonteCarloResult
            Container holding raw metric distributions across all simulations.
        """
        metric_series: dict[str, list[float]] = {}
        seen_keys: set[str] = set()

        indices = range(self.n_simulations)
        iterator = self._make_iterator(indices)

        for i in iterator:
            sim_result = self._run_single_simulation(df, i, seen_keys)

            # Filter to requested metrics if specified
            selected = (
                {k: v for k, v in sim_result.items() if k in self.metrics}
                if self.metrics is not None
                else sim_result
            )

            for key, value in selected.items():
                seen_keys.add(key)
                metric_series.setdefault(key, []).append(
                    float(value) if value is not None else float('nan')
                )

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            metric_series=metric_series,
            generator_name=type(self.generator).__name__,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_iterator(self, indices: range):
        """Wrap an index range with a progress indicator if verbose=True.

        Two behaviours depending on whether tqdm is installed:

        - tqdm available  → returns a tqdm progress bar with a descriptive label,
          ETA, and per-simulation throughput (simulations/s).
        - tqdm unavailable → returns a _FallbackProgress wrapper that prints a
          one-line update to stdout every 10% of total simulations. This keeps
          long runs from feeling frozen without adding a hard dependency.

        If verbose=False, the raw range is returned unchanged — zero overhead.
        """
        if not self.verbose:
            return indices

        generator_name = type(self.generator).__name__

        if _TQDM_AVAILABLE:
            return _tqdm(
                indices,
                total=self.n_simulations,
                desc=f"Monte Carlo ({generator_name})",
                unit="sim",
                dynamic_ncols=True,
            )

        return _FallbackProgress(indices, self.n_simulations, generator_name)

    def _run_single_simulation(
        self,
        df: pd.DataFrame,
        simulation_index: int,
        seen_keys: set[str],
    ) -> dict:
        """Run one simulation and return its metrics dict.

        Seeds the generator deterministically: base_seed + simulation_index.
        This guarantees that:
        - The same (runner, df) pair always produces the same sequence of
          synthetic scenarios (reproducible full runs).
        - Each simulation within a run gets a different scenario (no repeats).
        - If the user sets no seed, each simulation is independently random.

        Parameters
        ----------
        df : pd.DataFrame
            Original OHLCV data.
        simulation_index : int
            Index of the current simulation (0-based).

        Returns
        -------
        dict
            Metric name → float value from this simulation's backtest.
        """
        base_seed = self.generator.seed
        sim_seed = None if base_seed is None else base_seed + simulation_index
        sim_generator = self.generator.with_seed(sim_seed)

        try:
            synthetic_df = sim_generator.generate(df)
            backtest_result = self.engine.run_on(synthetic_df)
            return backtest_result.metrics
        except Exception as exc:
            import sys
            print(
                f"[MonteCarloRunner] simulation {simulation_index} failed "
                f"({type(exc).__name__}: {exc}) — recording NaN for all metrics.",
                file=sys.stderr,
            )
            return {key: float('nan') for key in seen_keys}


# ---------------------------------------------------------------------------
# Fallback progress reporter (no tqdm dependency)
# ---------------------------------------------------------------------------

class _FallbackProgress:
    """Minimal progress reporter used when tqdm is not installed.

    Wraps an iterable and prints a one-line update to stdout at every 10%
    milestone of the total count. This is intentionally simple — install tqdm
    for a richer experience.

    Design note: implemented as an iterator class rather than a generator
    function so it carries its own state and can be used exactly like tqdm
    (i.e. ``for i in _FallbackProgress(...)``).
    """

    def __init__(self, indices: range, total: int, label: str):
        self._indices = indices
        self._total = total
        self._label = label
        # Report at every 10% step, always including the final bar.
        step = max(1, total // 10)
        self._milestones = set(range(step, total + 1, step))
        self._milestones.add(total)

    def __iter__(self):
        for i in self._indices:
            yield i
            completed = i + 1
            if completed in self._milestones:
                pct = completed / self._total * 100
                print(
                    f"Monte Carlo ({self._label}): "
                    f"{completed}/{self._total} simulations complete "
                    f"({pct:.0f}%)"
                )
