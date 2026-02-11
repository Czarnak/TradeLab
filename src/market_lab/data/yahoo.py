"""Yahoo Finance OHLCV downloader with local Parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from market_lab.utils.config import YAHOO_CACHE_DIR, ensure_dirs
from market_lab.utils.logging import get_logger

log = get_logger("data.yahoo")

VALID_INTERVALS = ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo")
VALID_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")


def _cache_path(symbol: str, interval: str, period: str) -> Path:
    """Deterministic cache path for a download."""
    ensure_dirs()
    safe = symbol.upper().replace("/", "_")
    return YAHOO_CACHE_DIR / f"{safe}_{interval}_{period}.parquet"


def download(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"AAPL"``).
    period:
        Period string (e.g. ``"1y"``, ``"6mo"``).
    interval:
        Bar interval (e.g. ``"1d"``, ``"1h"``).
    use_cache:
        If *True*, return cached parquet if it exists.

    Returns
    -------
    pd.DataFrame in canonical bar schema.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{interval}'. Choose from {VALID_INTERVALS}")
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Choose from {VALID_PERIODS}")

    cp = _cache_path(symbol, interval, period)
    if use_cache and cp.exists():
        log.info("Cache hit: %s", cp)
        df = pd.read_parquet(cp)
        return df

    log.info("Downloading %s period=%s interval=%s", symbol, period, interval)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} (period={period}, interval={interval})")

    # Normalise to canonical schema
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "high", "low", "close", "volume"):
            rename[col] = lc
    df = df.rename(columns=rename)

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    df.index.name = "timestamp"

    # Ensure tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Save cache
    df.to_parquet(cp)
    log.info("Cached %d bars to %s", len(df), cp)
    return df
