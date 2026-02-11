"""Monte Carlo methods for generating variations of market data.

Three methods are implemented:
1. Bootstrap resampling — resample returns with replacement and reconstruct prices.
2. GBM with fitted parameters — estimate drift and vol from data, generate new paths.
3. Noise injection — add Gaussian noise to returns preserving statistical properties.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

from market_lab.utils.logging import get_logger

log = get_logger("data.monte_carlo")


class MCMethod(str, Enum):
    BOOTSTRAP = "bootstrap"
    GBM_FITTED = "gbm_fitted"
    NOISE_INJECTION = "noise_injection"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series."""
    return np.log(prices / prices.shift(1)).dropna()


def _reconstruct_ohlcv(
    original: pd.DataFrame,
    new_closes: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a full OHLCV DataFrame from new close prices and original structure."""
    n = len(new_closes)
    orig_trimmed = original.iloc[:n].copy()

    # Compute ratios from original data
    orig_close = orig_trimmed["close"].values
    ratio = new_closes / (orig_close + 1e-12)

    opens = orig_trimmed["open"].values * ratio
    highs = orig_trimmed["high"].values * ratio
    lows = orig_trimmed["low"].values * ratio

    # Add small noise to preserve realistic OHLC relationships
    noise = rng.normal(1.0, 0.001, n)
    highs = np.maximum(highs * np.abs(noise), np.maximum(opens, new_closes))
    lows = np.minimum(lows / np.abs(noise), np.minimum(opens, new_closes))

    # Volume: perturb slightly
    vol_noise = rng.normal(1.0, 0.1, n)
    volumes = np.maximum(0, orig_trimmed["volume"].values * vol_noise).astype(int)

    df = pd.DataFrame(
        {
            "open": np.round(opens, 4),
            "high": np.round(highs, 4),
            "low": np.round(lows, 4),
            "close": np.round(new_closes, 4),
            "volume": volumes,
        },
        index=orig_trimmed.index,
    )
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Method 1: Bootstrap resampling
# ---------------------------------------------------------------------------

def bootstrap_resample(
    df: pd.DataFrame,
    n_simulations: int = 10,
    seed: int | None = 42,
) -> list[pd.DataFrame]:
    """Resample log returns with replacement and reconstruct price series.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with ``close`` column.
    n_simulations : int
        Number of Monte Carlo paths.
    seed : int or None
        Random seed.

    Returns
    -------
    list of DataFrames, each a synthetic OHLCV variation.
    """
    rng = np.random.default_rng(seed)
    log_rets = _log_returns(df["close"]).values
    start_price = df["close"].iloc[0]
    n_bars = len(df)
    results: list[pd.DataFrame] = []

    for i in range(n_simulations):
        sampled = rng.choice(log_rets, size=n_bars - 1, replace=True)
        cum = np.concatenate([[0.0], np.cumsum(sampled)])
        new_closes = start_price * np.exp(cum)
        results.append(_reconstruct_ohlcv(df, new_closes, rng))

    log.info("Bootstrap: generated %d simulations (%d bars each)", n_simulations, n_bars)
    return results


# ---------------------------------------------------------------------------
# Method 2: GBM with fitted parameters
# ---------------------------------------------------------------------------

def gbm_fitted(
    df: pd.DataFrame,
    n_simulations: int = 10,
    seed: int | None = 42,
) -> list[pd.DataFrame]:
    """Fit GBM parameters from data and generate new paths.

    Estimates annualised drift (mu) and volatility (sigma) from the close
    series, then generates new GBM paths.
    """
    rng = np.random.default_rng(seed)
    log_rets = _log_returns(df["close"]).values
    n_bars = len(df)
    start_price = df["close"].iloc[0]

    # Estimate per-bar statistics
    mu_bar = np.mean(log_rets)
    sigma_bar = np.std(log_rets, ddof=1)

    results: list[pd.DataFrame] = []
    for _ in range(n_simulations):
        sim_rets = rng.normal(mu_bar, sigma_bar, n_bars - 1)
        cum = np.concatenate([[0.0], np.cumsum(sim_rets)])
        new_closes = start_price * np.exp(cum)
        results.append(_reconstruct_ohlcv(df, new_closes, rng))

    log.info(
        "GBM fitted: mu_bar=%.6f sigma_bar=%.6f, %d sims",
        mu_bar, sigma_bar, n_simulations,
    )
    return results


# ---------------------------------------------------------------------------
# Method 3: Noise injection
# ---------------------------------------------------------------------------

def noise_injection(
    df: pd.DataFrame,
    n_simulations: int = 10,
    noise_scale: float = 0.3,
    seed: int | None = 42,
) -> list[pd.DataFrame]:
    """Add Gaussian noise to log returns while preserving statistical properties.

    Parameters
    ----------
    noise_scale : float
        Fraction of return std used as noise amplitude (0.3 = 30% of sigma).
    """
    rng = np.random.default_rng(seed)
    log_rets = _log_returns(df["close"]).values
    n_bars = len(df)
    start_price = df["close"].iloc[0]
    sigma = np.std(log_rets, ddof=1)

    results: list[pd.DataFrame] = []
    for _ in range(n_simulations):
        noise = rng.normal(0, sigma * noise_scale, len(log_rets))
        perturbed = log_rets + noise
        cum = np.concatenate([[0.0], np.cumsum(perturbed)])
        new_closes = start_price * np.exp(cum)
        results.append(_reconstruct_ohlcv(df, new_closes, rng))

    log.info("Noise injection: scale=%.2f, %d sims", noise_scale, n_simulations)
    return results


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def generate_variations(
    df: pd.DataFrame,
    methods: Sequence[MCMethod] | None = None,
    n_simulations: int = 10,
    noise_scale: float = 0.3,
    seed: int | None = 42,
) -> dict[str, list[pd.DataFrame]]:
    """Run one or more Monte Carlo methods on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV data.
    methods : sequence of MCMethod or None
        Methods to use. ``None`` runs all three.
    n_simulations : int
        Simulations per method.
    noise_scale : float
        For noise injection method.
    seed : int or None
        Base seed (incremented per method).

    Returns
    -------
    dict mapping method name to list of simulated DataFrames.
    """
    if methods is None:
        methods = list(MCMethod)

    results: dict[str, list[pd.DataFrame]] = {}
    base_seed = seed

    for i, method in enumerate(methods):
        s = None if base_seed is None else base_seed + i * 1000
        if method == MCMethod.BOOTSTRAP:
            results[method.value] = bootstrap_resample(df, n_simulations, seed=s)
        elif method == MCMethod.GBM_FITTED:
            results[method.value] = gbm_fitted(df, n_simulations, seed=s)
        elif method == MCMethod.NOISE_INJECTION:
            results[method.value] = noise_injection(df, n_simulations, noise_scale, seed=s)

    return results
