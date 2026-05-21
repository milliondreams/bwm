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
    """Yield one Entity per unique CIK.

    SEC's `tickers.json` lists multiple ticker rows per CIK for issuers with
    multiple share classes (Federal Home Loan Mortgage Corp has 25 ticker
    entries for its 25 preferred series under one CIK; Bank of America, Fannie
    Mae, ProShares ETF families similarly). The convention in `tickers.json`
    is that the common-stock ticker appears first per CIK; for issuers with
    multiple "common-like" tickers we additionally prefer the SHORTEST one,
    since preferred-share tickers always carry a suffix (BAC-PB, BAC-PK, ...)
    and ETF family symbols are typically longer than the parent's common.

    This is the canonical dedup point — `EntityRegistry.write` then persists
    one row per CIK. Downstream code (e.g. `ingest_edgar_full`) also dedups
    defensively so a non-canonical registry doesn't reintroduce the dispatch
    duplication that caused Stage 5's first --all run to do 30% redundant SEC
    requests.
    """
    seen: dict[str, dict] = {}  # cik -> chosen-row
    for r in rows:
        cik = str(r["cik_str"]).zfill(10)
        ticker = str(r.get("ticker", "") or "")
        if cik not in seen:
            seen[cik] = {"cik": cik, "name": r["title"], "ticker": ticker}
            continue
        # Already have an entry — prefer the shorter ticker (common stock
        # tends to have the cleanest symbol; preferred shares carry suffixes).
        if len(ticker) < len(seen[cik]["ticker"]):
            seen[cik]["ticker"] = ticker
            # Keep first-seen name (SEC's convention); name doesn't vary
            # within a CIK anyway.
    for cik in sorted(seen.keys()):
        e = seen[cik]
        yield Entity(
            entity_id=EntityRegistry.entity_id_from_cik(cik),
            name=e["name"],
            cik=cik,
            ticker=e["ticker"],
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
