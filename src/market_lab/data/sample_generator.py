"""Generate sample OHLCV bar and tick CSV data using geometric Brownian motion.

Importable API — no CLI entry point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_lab.utils.config import EXAMPLES_DIR, ensure_dirs
from market_lab.utils.logging import get_logger

log = get_logger("data.sample_generator")


def generate_ohlcv_bars(
    n_bars: int = 500,
    start_price: float = 100.0,
    annual_drift: float = 0.05,
    annual_vol: float = 0.20,
    start_date: str = "2023-01-01",
    freq: str = "1D",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars using geometric Brownian motion.

    Parameters
    ----------
    n_bars : int
        Number of bars to generate.
    start_price : float
        Initial close price.
    annual_drift : float
        Annualised drift (mu).
    annual_vol : float
        Annualised volatility (sigma).
    start_date : str
        Start timestamp for the index.
    freq : str
        Pandas frequency string (``"1D"``, ``"1h"``, etc.).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Canonical OHLCV DataFrame with DatetimeIndex.
    """
    rng = np.random.default_rng(seed)

    # Convert annual params to per-bar
    if "D" in freq.upper() or "B" in freq.upper():
        bars_per_year = 252
    elif "h" in freq.lower() or "H" in freq:
        bars_per_year = 252 * 6.5  # ~6.5 trading hours/day
    else:
        bars_per_year = 252

    dt = 1.0 / bars_per_year
    mu_bar = (annual_drift - 0.5 * annual_vol**2) * dt
    sigma_bar = annual_vol * np.sqrt(dt)

    # Generate close prices via GBM
    log_returns = rng.normal(mu_bar, sigma_bar, size=n_bars)
    log_prices = np.cumsum(log_returns)
    log_prices = np.insert(log_prices, 0, 0.0)[: n_bars]
    closes = start_price * np.exp(log_prices)

    # Synthetic OHLV from close
    intrabar_vol = sigma_bar * 0.6
    opens = closes * np.exp(rng.normal(0, intrabar_vol * 0.3, n_bars))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, intrabar_vol * 0.5, n_bars)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, intrabar_vol * 0.5, n_bars)))

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    # Volume: random with slight trend correlation
    base_vol = 1_000_000
    volume = np.abs(rng.normal(base_vol, base_vol * 0.3, n_bars)).astype(int)

    idx = pd.date_range(start=start_date, periods=n_bars, freq=freq, tz="UTC")

    df = pd.DataFrame(
        {
            "open": np.round(opens, 4),
            "high": np.round(highs, 4),
            "low": np.round(lows, 4),
            "close": np.round(closes, 4),
            "volume": volume,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def generate_tick_data(
    n_ticks: int = 2000,
    mid_start: float = 100.0,
    spread_bps: float = 5.0,
    annual_vol: float = 0.20,
    start_date: str = "2023-01-01",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate synthetic tick data with bid/ask/mid.

    Parameters
    ----------
    n_ticks : int
        Number of ticks.
    mid_start : float
        Starting mid price.
    spread_bps : float
        Average bid-ask spread in basis points.
    annual_vol : float
        Annualised volatility.
    start_date : str
        Start timestamp.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Canonical tick DataFrame (bid, ask, mid, volume).
    """
    rng = np.random.default_rng(seed)

    # Simulate mid price via random walk
    tick_vol = annual_vol / np.sqrt(252 * 6.5 * 3600)  # per-second vol
    mid_returns = rng.normal(0, tick_vol * 10, n_ticks)  # scale up for tick intervals
    mids = mid_start * np.exp(np.cumsum(mid_returns))

    half_spread = mids * (spread_bps / 10_000) / 2
    spread_noise = rng.uniform(0.8, 1.2, n_ticks)
    half_spread *= spread_noise

    bids = mids - half_spread
    asks = mids + half_spread
    volumes = np.abs(rng.normal(100, 50, n_ticks)).astype(int)

    # Irregular tick timestamps
    intervals_ms = rng.exponential(500, n_ticks).astype(int)  # avg 500ms
    cum_ms = np.cumsum(intervals_ms)
    idx = pd.to_datetime(start_date, utc=True) + pd.to_timedelta(cum_ms, unit="ms")

    df = pd.DataFrame(
        {
            "bid": np.round(bids, 4),
            "ask": np.round(asks, 4),
            "mid": np.round(mids, 4),
            "volume": volumes,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def save_example_csvs(output_dir: Path | None = None) -> dict[str, Path]:
    """Generate and save example CSV files for bars and ticks.

    Returns
    -------
    dict mapping name to saved path.
    """
    ensure_dirs()
    out = output_dir or EXAMPLES_DIR
    out.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # Bars
    bars = generate_ohlcv_bars()
    bars_path = out / "sample_ohlcv.csv"
    bars.to_csv(bars_path)
    saved["bars"] = bars_path
    log.info("Saved sample bars: %s (%d rows)", bars_path, len(bars))

    # Ticks
    ticks = generate_tick_data()
    ticks_path = out / "sample_ticks.csv"
    ticks.to_csv(ticks_path)
    saved["ticks"] = ticks_path
    log.info("Saved sample ticks: %s (%d rows)", ticks_path, len(ticks))

    return saved
