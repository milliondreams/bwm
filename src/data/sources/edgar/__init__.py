"""SEC EDGAR ingestion package.

Modules:
    client                async HTTP client with token-bucket rate limit
    archive               content-addressed raw response storage
    state                 watermark store for incremental ingest
    companyfacts_legacy   the original sync companyfacts pull (kept for
                          cross-validation against the new per-filing path)

Public re-exports preserve back-compat with callers that still import
`ingest_company` / `ingest_companies` from `data.sources.edgar`.
"""
from data.sources.edgar.companyfacts_legacy import (
    ingest_company,
    ingest_companies,
    fetch_companyfacts,
    iter_facts,
)

__all__ = [
    "ingest_company",
    "ingest_companies",
    "fetch_companyfacts",
    "iter_facts",
]
