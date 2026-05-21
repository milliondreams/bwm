"""SEC EDGAR companyfacts connector — the financials modality.

Endpoint: GET https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json

Returns every XBRL fact ever reported by the company across all filings, grouped
by taxonomy and concept. Each fact carries `end` (period end = effective_date),
`filed` (filing date = availability_date), `accn` (accession), `form` (10-K/10-Q),
`fp` (FY/Q1/Q2/Q3/Q4), `fy`, `val`, and `unit` (USD/shares/pure).

Restatements: the same (concept, end, fp) appears multiple times when the
company restates, each with its own `filed` and `accn`. We preserve every
row — the PIT engine handles the "which one wins for as-of date D" logic.

Rate limits: SEC asks for ≤ 10 req/sec and a User-Agent identifying the requester.
"""
from __future__ import annotations

import time
from datetime import date, datetime
from typing import Iterable, Iterator, Optional

import pandas as pd
import requests

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.financial import FinancialFact
from data.schemas.pit import Modality

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
RATE_LIMIT_SLEEP = 0.11  # ~9 req/sec, safely under SEC's 10/sec cap


def _parse_iso(s: str) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def fetch_companyfacts(cik: str, user_agent: str) -> dict:
    """Pull all XBRL facts for one company. `cik` is 10-digit zero-padded."""
    url = COMPANYFACTS_URL.format(cik=cik.zfill(10))
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def iter_facts(payload: dict, entity_id: str) -> Iterator[FinancialFact]:
    """Flatten the nested companyfacts JSON into FinancialFact rows."""
    facts = payload.get("facts", {})
    for taxonomy, concepts in facts.items():
        for concept_name, concept_data in concepts.items():
            units = concept_data.get("units", {})
            for unit, observations in units.items():
                for obs in observations:
                    end = _parse_iso(obs.get("end"))
                    filed = _parse_iso(obs.get("filed"))
                    if end is None or filed is None:
                        continue
                    start = _parse_iso(obs.get("start"))
                    fy = obs.get("fy")
                    fp = obs.get("fp")
                    if fy is None or fp is None:
                        continue
                    accession = obs.get("accn", "")
                    yield FinancialFact(
                        entity_id=entity_id,
                        effective_date=end,
                        availability_date=filed,
                        source="sec_edgar:companyfacts",
                        source_ref=accession,
                        concept=f"{taxonomy}:{concept_name}",
                        taxonomy=taxonomy,
                        period_start=start,
                        period_end=end,
                        fiscal_year=int(fy),
                        fiscal_period=str(fp),
                        unit=unit,
                        value=float(obs["val"]),
                        form=obs.get("form", ""),
                        accession=accession,
                    )


def _mark_restatements(rows: list[FinancialFact]) -> pd.DataFrame:
    """Within a single ingest, populate `restated_at` for superseded rows.

    For each (entity, concept, taxonomy, fiscal_year, fiscal_period) group, the
    successor row (next-later `availability_date`) provides the `restated_at`
    timestamp for the predecessor. Cross-ingest restatements are handled the
    same way by the PIT engine at query time (it picks the latest filed).
    """
    df = pd.DataFrame([r.model_dump() for r in rows])
    if df.empty:
        return df

    key = ["entity_id", "concept", "fiscal_year", "fiscal_period"]
    df = df.sort_values(key + ["availability_date"])
    df["restated_at"] = (
        df.groupby(key)["availability_date"].shift(-1)
    )
    return df


def ingest_company(
    cik: str,
    user_agent: str,
    pit: PITEngine,
    registry: EntityRegistry,
) -> int:
    """Pull one company's facts and persist them. Returns row count written."""
    entity_id = EntityRegistry.entity_id_from_cik(cik)
    payload = fetch_companyfacts(cik, user_agent)
    rows = list(iter_facts(payload, entity_id))
    df = _mark_restatements(rows)
    if df.empty:
        return 0
    pit.write(Modality.FINANCIALS, df, partition_keys=FinancialFact.PARTITION_KEYS)
    return len(df)


def ingest_companies(
    ciks: Iterable[str],
    user_agent: str,
    pit: PITEngine,
    registry: EntityRegistry,
) -> dict[str, int]:
    """Sequentially ingest a list of CIKs respecting SEC rate limits."""
    counts: dict[str, int] = {}
    for cik in ciks:
        try:
            counts[cik] = ingest_company(cik, user_agent, pit, registry)
        except requests.HTTPError as e:
            counts[cik] = -1
            print(f"  ! {cik}: HTTP {e.response.status_code if e.response else '?'}")
        time.sleep(RATE_LIMIT_SLEEP)
    return counts
