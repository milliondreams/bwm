"""End-to-end smoke test for Phase A.

  uv run python -m data.cli.ingest_smoke

Seeds entity registry from SEC, ingests 10 large-cap companies' XBRL facts,
and runs PIT round-trip queries to verify restatement-aware as-of behavior.
"""
from __future__ import annotations

import os
from datetime import date

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.pit import Modality
from data.sources.edgar import ingest_companies
from data.sources.sec_tickers import seed_registry
from data.storage import get_storage

SMOKE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BRK-B", "WMT"]


def _user_agent() -> str:
    ua = os.environ.get("BWM_SEC_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "Set BWM_SEC_USER_AGENT='Your Name your@email' — SEC requires identifying UA"
        )
    return ua


def main() -> None:
    ua = _user_agent()
    storage = get_storage()
    registry = EntityRegistry(storage)
    pit = PITEngine(storage)

    print("[1/4] seeding entity registry from SEC tickers.json …")
    n = seed_registry(registry, ua)
    print(f"      wrote {n} entities")

    df = registry.load()
    smoke = df[df["ticker"].isin([t.replace("-", ".") for t in SMOKE_TICKERS] + SMOKE_TICKERS)]
    if smoke.empty:
        raise RuntimeError("smoke tickers not found in registry")
    ciks = smoke["cik"].tolist()
    print(f"[2/4] resolved {len(ciks)} CIKs: {dict(zip(smoke['ticker'], ciks))}")

    print("[3/4] ingesting EDGAR companyfacts …")
    counts = ingest_companies(ciks, ua, pit, registry)
    for cik, n in counts.items():
        ticker = smoke[smoke["cik"] == cik]["ticker"].iloc[0]
        print(f"      {ticker} ({cik}): {n} facts")

    print("[4/4] PIT round-trip on AAPL FY2020 Q2 Revenue (ASC 606 concept):")
    aapl_id = EntityRegistry.entity_id_from_cik(smoke[smoke["ticker"] == "AAPL"]["cik"].iloc[0])
    concept = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    # FY2020 Q2 ended 2020-03-28; filing landed 2020-05-01. Pre-filing as-of
    # should return nothing; post-filing returns the as-reported number.
    for as_of in (date(2020, 4, 30), date(2020, 8, 15), date(2024, 1, 1)):
        result = pit.query(
            Modality.FINANCIALS,
            entity_ids=[aapl_id],
            as_of=as_of,
            effective_between=(date(2020, 3, 1), date(2020, 7, 1)),
            extra_filters={"concept": concept, "fiscal_period": "Q2", "fiscal_year": 2020},
        )
        if result.empty:
            print(f"      as_of {as_of}: <not yet knowable>")
        else:
            r = result.iloc[0]
            print(
                f"      as_of {as_of}: ${r['value']:,.0f} "
                f"(filed {r['availability_date']}, accn {r['accession']}, form {r['form']})"
            )


if __name__ == "__main__":
    main()
