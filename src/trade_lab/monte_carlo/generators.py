"""Monte Carlo data generators for financial time series.

All generators share a common interface: accept an original OHLCV DataFrame
and return a synthetic OHLCV DataFrame with identical index and column
structure but statistically varied price series.

OHLCV reconstruction strategy
------------------------------
Generators operate on log-returns of Close (log-returns are addable,
approximately normally distributed, and guarantee positive prices).
After generating a synthetic Close series, intrabar structure (Open, High,
Low) is reconstructed by sampling historical ratios (e.g. High/Close) with
replacement and enforcing standard OHLC consistency constraints:
    High ≥ max(Open, Close)
    Low  ≤ min(Open, Close)
Volume is independently resampled from the historical distribution.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_block_size(n: int) -> int:
    """Politis & Romano (1994) default: block_size ≈ n^(1/3).

    The cube-root rule balances bias (too-small blocks break autocorrelation)
    and variance (too-large blocks reduce effective resampling diversity).
    Minimum of 2 to avoid degenerate single-bar blocks.
    """
    return max(2, int(round(n ** (1 / 3))))


def _log_returns(close: np.ndarray) -> np.ndarray:
    """Compute log returns from a Close price array.

    Returns an array of length n-1 (first observation is lost).
    """
    return np.diff(np.log(close))


def _close_from_log_returns(log_returns: np.ndarray, initial_close: float) -> np.ndarray:
    """Reconstruct a Close price series from log returns.

    Parameters
    ----------
    log_returns : np.ndarray
        Array of length n-1 log returns.
    initial_close : float
        Starting price (first bar's Close from the original series).

    Returns
    -------
    np.ndarray
        Price array of length n (same as original).
    """
    log_prices = np.concatenate([
        [np.log(initial_close)],
        np.log(initial_close) + np.cumsum(log_returns),
    ])
    return np.exp(log_prices)


def _reconstruct_ohlcv(
    synthetic_close: np.ndarray,
    original_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a full OHLCV DataFrame from a synthetic Close series.

    Intrabar ratios (High/Close, Low/Close, Open/Close) are sampled with
    replacement from the historical data, then OHLC constraints are enforced:
        High ≥ max(Open, Close)
        Low  ≤ min(Open, Close)

    Volume is independently resampled from historical Volume values.

    Parameters
    ----------
    synthetic_close : np.ndarray
        Synthetic Close price array.
    original_df : pd.DataFrame
        Original OHLCV DataFrame (used as ratio source).
    rng : np.random.Generator
        Random number generator (already seeded by the caller).

    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV DataFrame sharing the original index.
    """
    n = len(synthetic_close)
    orig_close = original_df['Close'].to_numpy(dtype=float)

    # Compute ratios; guard against zero Close prices
    safe_close = np.where(orig_close == 0, np.nan, orig_close)
    high_ratio = original_df['High'].to_numpy(dtype=float) / safe_close
    low_ratio = original_df['Low'].to_numpy(dtype=float) / safe_close
    open_ratio = original_df['Open'].to_numpy(dtype=float) / safe_close

    # Keep only rows where all ratios are finite
    mask = np.isfinite(high_ratio) & np.isfinite(low_ratio) & np.isfinite(open_ratio)
    high_ratio = high_ratio[mask]
    low_ratio = low_ratio[mask]
    open_ratio = open_ratio[mask]

    # Sample ratios independently for each synthetic bar
    idx = rng.integers(0, len(high_ratio), size=n)
    synth_open = synthetic_close * open_ratio[idx]
    synth_high = synthetic_close * high_ratio[idx]
    synth_low = synthetic_close * low_ratio[idx]

    # Enforce OHLC consistency
    synth_high = np.maximum(synth_high, np.maximum(synth_open, synthetic_close))
    synth_low = np.minimum(synth_low, np.minimum(synth_open, synthetic_close))

    # Resample Volume from historical distribution
    hist_volume = original_df['Volume'].to_numpy(dtype=float)
    synth_volume = hist_volume[rng.integers(0, len(hist_volume), size=n)]

    return pd.DataFrame(
        {
            'Open': synth_open,
            'High': synth_high,
            'Low': synth_low,
            'Close': synthetic_close,
            'Volume': synth_volume,
        },
        index=original_df.index,
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseGenerator(ABC):
    """Abstract base class for Monte Carlo OHLCV generators.

    Subclasses implement ``generate(df)`` to produce one synthetic scenario.
    The ``seed`` mechanism ensures reproducibility: when the runner varies
    seeds across simulations (seed + i), each simulation is independently
    reproducible while the full run is deterministic given the base seed.

    Parameters
    ----------
    seed : int | None
        Base random seed. None means non-deterministic.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate one synthetic OHLCV scenario.

        Parameters
        ----------
        df : pd.DataFrame
            Original OHLCV DataFrame with columns Open, High, Low, Close, Volume.

        Returns
        -------
        pd.DataFrame
            Synthetic OHLCV DataFrame, same index and columns as ``df``.
        """
        ...

    def _make_rng(self) -> np.random.Generator:
        """Construct a NumPy random generator from the current seed."""
        return np.random.default_rng(self.seed)

    def with_seed(self, seed: int | None) -> 'BaseGenerator':
        """Return a shallow copy of this generator with a different seed.

        Used by MonteCarloRunner to create per-simulation generators without
        mutating the original.
        """
        gen = copy.copy(self)
        gen.seed = seed
        return gen


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

class ReturnShuffler(BaseGenerator):
    """Naive Monte Carlo baseline: randomly permutes daily log returns.

    This completely destroys temporal structure (autocorrelation, volatility
    clustering, trend) but preserves the marginal return distribution exactly.
    Use as a baseline: if a more sophisticated generator produces meaningfully
    different results, the strategy is sensitive to temporal structure.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.
    """

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self._make_rng()
        close = df['Close'].to_numpy(dtype=float)
        log_ret = _log_returns(close)

        shuffled = rng.permutation(log_ret)
        synthetic_close = _close_from_log_returns(shuffled, close[0])

        return _reconstruct_ohlcv(synthetic_close, df, rng)


class BlockBootstrap(BaseGenerator):
    """Stationary Block Bootstrap (Politis & Romano, 1994).

    Resamples contiguous blocks of log returns with replacement. Block lengths
    are drawn from a Geometric(p=1/block_size) distribution, giving a mean
    block length of ``block_size``. Using a random rather than fixed block
    length is what makes the bootstrap 'stationary' — the resulting series is
    second-order stationary — and preserves short-range autocorrelation and
    volatility clustering that a fixed-length block bootstrap would distort.

    When a sampled block would extend past the end of the series, it is simply
    truncated (no wrapping). See CircularBlockBootstrap for the wrapping variant.

    Parameters
    ----------
    block_size : int | None
        Mean block length. If None, the Politis & Romano cube-root rule is
        applied: block_size = max(2, round(n^(1/3))).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, block_size: int | None = None, seed: int | None = None):
        super().__init__(seed)
        self.block_size = block_size

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self._make_rng()
        close = df['Close'].to_numpy(dtype=float)
        log_ret = _log_returns(close)
        n = len(log_ret)

        block_size = (
            self.block_size
            if self.block_size is not None
            else _default_block_size(n)
        )
        p = 1.0 / block_size

        resampled: list[float] = []
        while len(resampled) < n:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(p))
            end = min(start + length, n)
            resampled.extend(log_ret[start:end].tolist())

        synthetic_log_ret = np.array(resampled[:n])
        synthetic_close = _close_from_log_returns(synthetic_log_ret, close[0])

        return _reconstruct_ohlcv(synthetic_close, df, rng)


class CircularBlockBootstrap(BaseGenerator):
    """Circular Block Bootstrap (Politis & Romano, 1992).

    Like BlockBootstrap, but the return series is treated as circular: when a
    block would extend past the last observation, it wraps around to the
    beginning. This eliminates the boundary bias of the standard block
    bootstrap — in the standard version, observations near the end of the
    series are systematically under-represented because fewer block start
    positions can reach them.

    Uses a fixed block length (not randomised) because circularity already
    ensures all observations are equally likely block-start candidates.

    Parameters
    ----------
    block_size : int | None
        Fixed block length. If None, the cube-root rule is applied.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, block_size: int | None = None, seed: int | None = None):
        super().__init__(seed)
        self.block_size = block_size

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self._make_rng()
        close = df['Close'].to_numpy(dtype=float)
        log_ret = _log_returns(close)
        n = len(log_ret)

        block_size = (
            self.block_size
            if self.block_size is not None
            else _default_block_size(n)
        )
        if block_size >= n:
            raise ValueError(
                f"CircularBlockBootstrap: block_size ({block_size}) must be "
                f"less than the number of return observations ({n}). "
                f"Reduce block_size or use a longer data series."
            )

        # Tile the series so any start + block_size slice is always valid
        circular = np.tile(log_ret, 2)

        resampled: list[float] = []
        while len(resampled) < n:
            # Start positions in [0, n) so wrap-around is symmetric
            start = int(rng.integers(0, n))
            block = circular[start: start + block_size]
            resampled.extend(block.tolist())

        synthetic_log_ret = np.array(resampled[:n])
        synthetic_close = _close_from_log_returns(synthetic_log_ret, close[0])

        return _reconstruct_ohlcv(synthetic_close, df, rng)


class GBMSimulator(BaseGenerator):
    """Geometric Brownian Motion simulator.

    Fits drift (μ) and volatility (σ) from historical log returns, then
    generates a fully synthetic price path under GBM assumptions.

    GBM models log-returns as:
        r_t = (μ - σ²/2) * dt + σ * √dt * Z_t,   Z_t ~ N(0,1)

    With dt = 1 bar this simplifies to:
        r_t = (μ - σ²/2) + σ * Z_t

    The (μ - σ²/2) correction (Itô's lemma) ensures the expected price level
    grows at μ, not at μ + σ²/2 — a common modelling error to avoid.

    GBM is the Black-Scholes world: log-normal returns, constant parameters,
    no fat tails, no volatility clustering. It underestimates tail risk. Use
    it as an analytical baseline, not as a primary robustness test.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.
    """

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self._make_rng()
        close = df['Close'].to_numpy(dtype=float)
        log_ret = _log_returns(close)
        n = len(log_ret)

        mu = float(np.mean(log_ret))
        sigma = float(np.std(log_ret))

        # Itô-corrected drift per bar
        drift = mu - 0.5 * sigma ** 2
        innovations = rng.standard_normal(n)
        synthetic_log_ret = drift + sigma * innovations

        synthetic_close = _close_from_log_returns(synthetic_log_ret, close[0])

        return _reconstruct_ohlcv(synthetic_close, df, rng)
