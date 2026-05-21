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


def main() -> None:
    """CLI entrypoint: `python -m data.sources.sec_tickers`.

    Seeds the entity registry from SEC's `company_tickers.json` into the
    configured storage backend. Idempotent — re-running overwrites with the
    latest tickers list. The User-Agent header satisfies SEC's fair-use rule.
    """
    import os
    from data.storage import get_storage

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    registry = EntityRegistry(storage)
    ua = os.environ.get("SEC_USER_AGENT", "norn-bwm research rohit@dragonscale.ai")
    print(f"seeding entity registry from {TICKERS_URL}")
    n = seed_registry(registry, ua)
    print(f"wrote {n} entities to entities/registry.parquet")


if __name__ == "__main__":
    main()
