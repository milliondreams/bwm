"""Entity registry — the canonical id ↔ external id table.

Seeded from SEC's tickers.json (~10K US public companies). Cross-jurisdiction
merge with LEI/ISIN is deferred (OQ-5).
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from data.schemas.entity import Entity
from data.storage import Storage

REGISTRY_PATH = "entities/registry.parquet"


class EntityRegistry:
    def __init__(self, storage: Storage):
        self.storage = storage

    def write(self, entities: Iterable[Entity]) -> None:
        df = pd.DataFrame([e.model_dump() for e in entities])
        self.storage.write_parquet(REGISTRY_PATH, df)

    def load(self) -> pd.DataFrame:
        return self.storage.read_parquet(REGISTRY_PATH)

    def by_ticker(self, ticker: str) -> Optional[Entity]:
        df = self.load()
        row = df[df["ticker"].str.upper() == ticker.upper()]
        if row.empty:
            return None
        return Entity(**row.iloc[0].to_dict())

    def by_cik(self, cik: str) -> Optional[Entity]:
        df = self.load()
        normalized = str(cik).lstrip("0").zfill(10)
        row = df[df["cik"] == normalized]
        if row.empty:
            return None
        return Entity(**row.iloc[0].to_dict())

    @staticmethod
    def entity_id_from_cik(cik: str | int) -> str:
        return f"us-cik-{str(cik).lstrip('0').zfill(10)}"
