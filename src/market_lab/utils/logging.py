"""Centralised logging configuration for Market-Lab."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialised = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once. Safe to call multiple times."""
    global _initialised
    if _initialised:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger("market_lab")
    root.setLevel(level)
    root.addHandler(handler)
    _initialised = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``market_lab`` namespace."""
    setup_logging()
    return logging.getLogger(f"market_lab.{name}")
