"""Application-wide configuration and path management."""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file to find the repo root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: use cwd
    return Path.cwd()


PROJECT_ROOT: Path = _find_project_root()

DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = PROJECT_ROOT / "cache"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

YAHOO_CACHE_DIR: Path = CACHE_DIR / "yahoo"
INSIDER_CACHE_DIR: Path = CACHE_DIR / "insiders"
MONTE_CARLO_CACHE_DIR: Path = CACHE_DIR / "monte_carlo"

BACKTEST_OUTPUTS_DIR: Path = OUTPUTS_DIR / "backtests"
ML_OUTPUTS_DIR: Path = OUTPUTS_DIR / "ml"

EXAMPLES_DIR: Path = DATA_DIR / "examples"
TICKERS_FILE: Path = DATA_DIR / "tickers_watchlist.txt"
SENATE_FILE: Path = DATA_DIR / "senate_insiders.json"

# SEC EDGAR compliance
SEC_USER_AGENT: str = os.getenv(
    "SEC_USER_AGENT",
    "MarketLab/0.1 (research; contact@example.com)",
)
SEC_MAX_REQUESTS_PER_SECOND: int = 10

# Defaults
DEFAULT_COMMISSION_BPS: float = 5.0
DEFAULT_SLIPPAGE_BPS: float = 2.0


def ensure_dirs() -> None:
    """Create all required runtime directories."""
    for d in (
        CACHE_DIR,
        YAHOO_CACHE_DIR,
        INSIDER_CACHE_DIR,
        MONTE_CARLO_CACHE_DIR,
        OUTPUTS_DIR,
        BACKTEST_OUTPUTS_DIR,
        ML_OUTPUTS_DIR,
        EXAMPLES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
