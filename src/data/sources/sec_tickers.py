"""Seed the entity registry from SEC's tickers.json."""
from __future__ import annotations

import json
from typing import Iterator

import requests

from data.schemas.entity import Entity
from data.entity.registry import EntityRegistry

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def fetch_sec_tickers(user_agent: str) -> list[dict]:
    """Pull the SEC's master list of US public company tickers.

    SEC requires a User-Agent header identifying the requester (name + email).
    """
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    r = requests.get(TICKERS_URL, headers=headers, timeout=30)
    r.raise_for_status()
    payload = r.json()
    # payload is {"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...}
    return list(payload.values())


def to_entities(rows: list[dict]) -> Iterator[Entity]:
    for r in rows:
        cik = str(r["cik_str"]).zfill(10)
        yield Entity(
            entity_id=EntityRegistry.entity_id_from_cik(cik),
            name=r["title"],
            cik=cik,
            ticker=r["ticker"],
            country="US",
        )


def seed_registry(registry: EntityRegistry, user_agent: str) -> int:
    rows = fetch_sec_tickers(user_agent)
    entities = list(to_entities(rows))
    registry.write(entities)
    return len(entities)
