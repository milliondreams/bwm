"""Stage 2 driver: ingest one specific filing via the new per-filing iXBRL path.

Given a (CIK, accession) pair, fetch the submissions index to locate the
primary document, download the iXBRL HTML, parse facts with full context
dimensions, and write to canonical financials. Designed for the targeted
validation on AAPL Q2-FY2020 10-Q before scaling to the full universe.

Run:
    uv run python -m data.cli.ingest_xbrl_one \
        --cik 0000320193 --accession 0000320193-20-000052
"""
from __future__ import annotations

import argparse
import asyncio
import os
from datetime import date
from typing import Iterable

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.financial import FinancialFact
from data.schemas.pit import Modality
from data.sources.edgar.archive import archive
from data.sources.edgar.client import edgar_client
from data.sources.edgar.submissions import Filing, list_filings
from data.sources.edgar.xbrl import ParsedFact, parse_ixbrl
from data.storage import get_storage

# DEI (Document and Entity Information) concept names carry fiscal year/period.
DEI_FY = "DocumentFiscalYearFocus"
DEI_FP = "DocumentFiscalPeriodFocus"
DEI_TYPE = "DocumentType"


def _find_filing(filings: Iterable[Filing], accession: str) -> Filing:
    for f in filings:
        if f.accession == accession:
            return f
    raise SystemExit(f"accession {accession} not in submissions index")


def _extract_fiscal_meta(facts: list[ParsedFact]) -> tuple[int, str, str]:
    """Pull (fiscal_year, fiscal_period, document_type) from DEI facts.

    DEI concepts always live in the dei:* taxonomy and are reported once per
    filing. They are themselves XBRL facts in the iXBRL document — the parser
    surfaces them like any other fact. We just look them up here.
    """
    fy: int = 0
    fp: str = ""
    doc_type: str = ""
    for f in facts:
        if f.taxonomy != "dei":
            continue
        bare = f.concept.split(":", 1)[-1]
        if bare == DEI_FY:
            try:
                fy = int(f.value)
            except (ValueError, TypeError):
                pass
    # FP and doc type are non-numeric — they come through as ix:nonNumeric and
    # our parser deliberately ignores those. As a fallback, infer from the
    # form code (10-K → FY, 10-Q + report_date → Q1/Q2/Q3 won't work for
    # non-calendar fiscal years like Apple, so we accept "Q?" and let it be
    # corrected in a later pass that adds nonNumeric extraction).
    return fy, fp, doc_type


def _to_pit_rows(
    filing: Filing,
    entity_id: str,
    facts: list[ParsedFact],
    fy: int,
    fp_hint: str,
) -> pd.DataFrame:
    if not facts:
        return pd.DataFrame()
    rows = []
    # Heuristic: 10-K → FY; 10-Q without DEI → "Q" placeholder. The real fp
    # comes from DEI:DocumentFiscalPeriodFocus once nonNumeric parsing lands.
    if filing.form.startswith("10-K"):
        fp = "FY"
    elif filing.form.startswith("10-Q"):
        fp = fp_hint or "Q"
    else:
        fp = fp_hint or filing.form
    for f in facts:
        # Match the period in the context to the filing's reporting period
        # so we don't pull in comparative-period facts as if they were the
        # current quarter. For the current period, the context's period_end
        # equals the filing's report_date.
        if f.context.period_end != filing.report_date:
            continue
        rows.append({
            "entity_id": entity_id,
            "modality": Modality.FINANCIALS.value,
            "effective_date": f.context.period_end,
            "availability_date": filing.filed,
            "restated_at": None,
            "source": "sec_edgar:xbrl_instance",
            "source_ref": filing.accession,
            "concept": f.concept,
            "taxonomy": f.taxonomy,
            "period_start": f.context.period_start,
            "period_end": f.context.period_end,
            "fiscal_year": fy or filing.report_date.year,
            "fiscal_period": fp,
            "unit": f.unit,
            "value": f.value,
            "form": filing.form,
            "accession": filing.accession,
            "context_id": f.context.context_id,
            "dimensions_json": f.context.dimensions_json,
        })
    return pd.DataFrame(rows)


async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cik", required=True)
    ap.add_argument("--accession", required=True)
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    entity_id = EntityRegistry.entity_id_from_cik(args.cik)

    async with edgar_client() as client:
        print(f"fetching submissions for CIK {args.cik}…")
        filings = await list_filings(client, args.cik)
        print(f"  {len(filings)} XBRL-bearing filings in recent index")
        filing = _find_filing(filings, args.accession)
        print(f"  target: {filing.form} accn={filing.accession} filed={filing.filed} report_date={filing.report_date}")
        print(f"  fetching primary doc: {filing.primary_doc_url}")
        body = await client.get_bytes(filing.primary_doc_url)
        print(f"  {len(body):,} bytes downloaded")

    path, was_new = archive(
        storage, args.cik.zfill(10), "primary", body, accession=args.accession
    )
    print(f"  archived: {path}  (was_new={was_new})")

    facts = list(parse_ixbrl(body))
    print(f"  parsed {len(facts):,} numeric facts from iXBRL")

    fy, fp, _ = _extract_fiscal_meta(facts)
    if fy:
        print(f"  DEI fiscal year focus: {fy}")
    df = _to_pit_rows(filing, entity_id, facts, fy, fp)
    print(f"  filtered to {len(df):,} rows matching report_date={filing.report_date}")

    if not df.empty:
        pit.write(Modality.FINANCIALS, df, partition_keys=FinancialFact.PARTITION_KEYS)
        print(f"  wrote {len(df):,} rows to canonical financials")

    # Quick summary by dimensions
    if not df.empty:
        consolidated = df[df["dimensions_json"] == ""]
        dimensional = df[df["dimensions_json"] != ""]
        print(f"\nconsolidated rows: {len(consolidated):,}")
        print(f"dimensional rows:  {len(dimensional):,}")
        if not dimensional.empty:
            print("\ndistinct dimension axes seen:")
            axes = set()
            for d in dimensional["dimensions_json"]:
                import json as _json
                for k in _json.loads(d).keys():
                    axes.add(k)
            for a in sorted(axes):
                print(f"  - {a}")


if __name__ == "__main__":
    asyncio.run(amain())
