"""Dataset registry: manage multiple loaded datasets with unique IDs and metadata."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

from market_lab.utils.logging import get_logger

log = get_logger("data.registry")


@dataclass
class DatasetEntry:
    """Single dataset entry in the registry."""

    id: str
    name: str
    schema_type: str  # "bars" or "ticks"
    source: str  # "csv", "yahoo", "generated", "monte_carlo"
    df: pd.DataFrame
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    @property
    def date_range(self) -> tuple[str, str]:
        if isinstance(self.df.index, pd.DatetimeIndex) and len(self.df) > 0:
            return str(self.df.index[0]), str(self.df.index[-1])
        return ("N/A", "N/A")

    def summary(self) -> dict:
        start, end = self.date_range
        return {
            "id": self.id,
            "name": self.name,
            "schema": self.schema_type,
            "source": self.source,
            "rows": self.rows,
            "start": start,
            "end": end,
        }


class DatasetRegistry:
    """Thread-safe* in-memory registry of loaded datasets.

    *Note: thread safety relies on GIL for simple dict ops;
    for full safety, wrap with a lock if needed.
    """

    def __init__(self) -> None:
        self._datasets: dict[str, DatasetEntry] = {}

    def register(
        self,
        df: pd.DataFrame,
        name: str,
        schema_type: str = "bars",
        source: str = "csv",
        metadata: dict | None = None,
    ) -> str:
        """Add a dataset and return its unique ID."""
        ds_id = uuid.uuid4().hex[:8]
        entry = DatasetEntry(
            id=ds_id,
            name=name,
            schema_type=schema_type,
            source=source,
            df=df,
            metadata=metadata or {},
        )
        self._datasets[ds_id] = entry
        log.info("Registered dataset '%s' (id=%s, %d rows)", name, ds_id, len(df))
        return ds_id

    def get(self, ds_id: str) -> DatasetEntry:
        """Retrieve a dataset entry by ID."""
        if ds_id not in self._datasets:
            raise KeyError(f"Dataset '{ds_id}' not found in registry.")
        return self._datasets[ds_id]

    def remove(self, ds_id: str) -> None:
        """Remove a dataset from the registry."""
        if ds_id in self._datasets:
            del self._datasets[ds_id]
            log.info("Removed dataset id=%s", ds_id)

    def list_all(self) -> list[dict]:
        """Return summary dicts for all datasets."""
        return [e.summary() for e in self._datasets.values()]

    def ids(self) -> list[str]:
        return list(self._datasets.keys())

    def __len__(self) -> int:
        return len(self._datasets)

    def __iter__(self) -> Iterator[DatasetEntry]:
        return iter(self._datasets.values())

    def __contains__(self, ds_id: str) -> bool:
        return ds_id in self._datasets


# Module-level singleton
registry = DatasetRegistry()
