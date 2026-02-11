"""CSV / Parquet import with robust schema validation.

Canonical schemas
-----------------
Bars:  index=timestamp  columns=[open, high, low, close, volume]
Ticks: index=timestamp  columns=[bid, ask, mid, volume]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

from market_lab.utils.logging import get_logger

log = get_logger("data.loaders")

# ---------------------------------------------------------------------------
# Header alias maps (lowercase)
# ---------------------------------------------------------------------------
_BAR_ALIASES: dict[str, str] = {
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "adj close": "close",
    "adj_close": "close",
    "volume": "volume",
    "vol": "volume",
    "v": "volume",
}

_TICK_ALIASES: dict[str, str] = {
    "bid": "bid",
    "ask": "ask",
    "offer": "ask",
    "volume": "volume",
    "vol": "volume",
    "v": "volume",
}

_TIMESTAMP_ALIASES: set[str] = {
    "timestamp",
    "datetime",
    "date",
    "time",
    "dt",
    "ts",
}

SchemaType = Literal["bars", "ticks"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_columns(columns: list[str]) -> dict[str, str]:
    """Return mapping from original column name to normalised name."""
    out: dict[str, str] = {}
    for col in columns:
        key = re.sub(r"[^a-z0-9]", " ", col.lower()).strip()
        key = re.sub(r"\s+", " ", key)
        out[col] = key
    return out


def _detect_timestamp_col(df: pd.DataFrame) -> str | None:
    """Find the timestamp column (index or column)."""
    for col in df.columns:
        if col.lower().strip() in _TIMESTAMP_ALIASES:
            return col
    return None


def _parse_timestamps(series: pd.Series) -> pd.DatetimeIndex:
    """Try multiple formats to parse a timestamp series."""
    return pd.to_datetime(series, utc=True, format="mixed")


def _apply_alias_map(
    df: pd.DataFrame,
    alias_map: dict[str, str],
) -> pd.DataFrame:
    """Rename columns using an alias map, keeping only matched ones + extras."""
    norm = _normalise_columns(list(df.columns))
    rename: dict[str, str] = {}
    for orig, normed in norm.items():
        if normed in alias_map:
            rename[orig] = alias_map[normed]
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(
    path: str | Path,
    schema: SchemaType = "bars",
) -> pd.DataFrame:
    """Load a CSV file into canonical bar or tick schema.

    Parameters
    ----------
    path:
        Path to CSV file.
    schema:
        ``"bars"`` or ``"ticks"``.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and canonical columns.

    Raises
    ------
    ValueError
        If required columns cannot be resolved.
    """
    path = Path(path)
    log.info("Loading CSV: %s (schema=%s)", path, schema)

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    # --- Detect and set timestamp index ---
    ts_col = _detect_timestamp_col(df)
    if ts_col is not None:
        df.index = _parse_timestamps(df[ts_col])
        df = df.drop(columns=[ts_col])
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try parsing the existing index
        try:
            df.index = _parse_timestamps(df.index.to_series())
        except Exception:
            raise ValueError(
                f"Cannot find or parse a timestamp column in {path}. "
                f"Expected one of: {sorted(_TIMESTAMP_ALIASES)}"
            )
    df.index.name = "timestamp"

    # --- Apply alias map ---
    alias_map = _BAR_ALIASES if schema == "bars" else _TICK_ALIASES
    df = _apply_alias_map(df, alias_map)

    # --- Validate required columns ---
    if schema == "bars":
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required bar columns after alias resolution: {missing}. "
                f"Available: {sorted(df.columns)}"
            )
        df = df[["open", "high", "low", "close", "volume"]].copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    elif schema == "ticks":
        required = {"bid", "ask"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required tick columns: {missing}. "
                f"Available: {sorted(df.columns)}"
            )
        cols = ["bid", "ask"]
        if "volume" in df.columns:
            cols.append("volume")
        df = df[cols].copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        if "volume" not in df.columns:
            df["volume"] = 0.0

    df = df.dropna(how="all")
    df = df.sort_index()
    log.info("Loaded %d rows from %s", len(df), path.name)
    return df


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file (assumed canonical schema)."""
    path = Path(path)
    log.info("Loading Parquet: %s", path)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df
