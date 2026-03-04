"""Monte Carlo simulation module for trade_lab.

Provides tools for generating synthetic market scenarios and evaluating
strategy robustness across those scenarios to guard against curve fitting.

Generators
----------
ReturnShuffler
    Naive baseline: permutes daily log returns. Destroys all temporal structure.
BlockBootstrap
    Stationary Block Bootstrap (Politis & Romano, 1994). Resamples blocks with
    geometric-distributed lengths to preserve short-range autocorrelation.
    Recommended as the primary robustness test.
CircularBlockBootstrap
    Like BlockBootstrap but wraps around the series end. Eliminates boundary
    bias. Good complement to BlockBootstrap.
GBMSimulator
    Generates synthetic paths under Geometric Brownian Motion assumptions
    (log-normal returns, constant drift and volatility). Analytical baseline.

Runner
------
MonteCarloRunner
    Orchestrates N simulations: generates synthetic data, runs backtests,
    collects metric distributions.

Analysis
--------
MonteCarloAnalysis
    Statistical summaries (percentiles, confidence intervals, percentile rank
    of a real backtest result within the MC distribution).

Typical usage
-------------
>>> from trade_lab.backtesting import BacktestEngine
>>> from trade_lab.monte_carlo import BlockBootstrap, MonteCarloRunner, MonteCarloAnalysis
>>>
>>> engine = BacktestEngine(strategy=my_strategy, ticker='SPY', ...)
>>> original_df = engine.fetch_data()
>>>
>>> runner = MonteCarloRunner(engine, BlockBootstrap(seed=42), n_simulations=500)
>>> mc_result = runner.run(original_df)
>>>
>>> analysis = MonteCarloAnalysis(mc_result)
>>> print(analysis.summary())
>>> print(analysis.percentile_of('sharpe_ratio', real_sharpe))
"""

from trade_lab.monte_carlo.generators import (
    BaseGenerator,
    ReturnShuffler,
    BlockBootstrap,
    CircularBlockBootstrap,
    GBMSimulator,
)
from trade_lab.monte_carlo.runner import MonteCarloRunner, MonteCarloResult
from trade_lab.monte_carlo.analysis import MonteCarloAnalysis

__all__ = [
    "BaseGenerator",
    "ReturnShuffler",
    "BlockBootstrap",
    "CircularBlockBootstrap",
    "GBMSimulator",
    "MonteCarloRunner",
    "MonteCarloResult",
    "MonteCarloAnalysis",
]
