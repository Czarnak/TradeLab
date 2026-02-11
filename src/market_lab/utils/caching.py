"""Simple file-based caching helpers."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from market_lab.utils.logging import get_logger

log = get_logger("cache")


def cache_key(*parts: str) -> str:
    """Deterministic cache key from arbitrary string parts."""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def read_json_cache(path: Path, max_age_seconds: float | None = None) -> Any | None:
    """Read a JSON cache file. Returns *None* if missing or expired."""
    if not path.exists():
        return None
    if max_age_seconds is not None:
        age = time.time() - path.stat().st_mtime
        if age > max_age_seconds:
            log.debug("Cache expired: %s (age=%.0fs)", path, age)
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to read cache %s: %s", path, exc)
        return None


def write_json_cache(path: Path, data: Any) -> None:
    """Write data as JSON to a cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    log.debug("Cache written: %s", path)


def read_text_cache(path: Path, max_age_seconds: float | None = None) -> str | None:
    """Read a text/HTML cache file. Returns *None* if missing or expired."""
    if not path.exists():
        return None
    if max_age_seconds is not None:
        age = time.time() - path.stat().st_mtime
        if age > max_age_seconds:
            return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def write_text_cache(path: Path, text: str) -> None:
    """Write text to a cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
