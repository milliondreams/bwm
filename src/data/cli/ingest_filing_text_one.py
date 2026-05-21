"""Stage 4 driver: extract per-Item text chunks from one filing.

Re-uses the primary-doc HTML already archived by Stage 2 ingest where
possible, falling back to a fresh fetch.

Run:
    uv run python -m data.cli.ingest_filing_text_one \
        --cik 0000320193 --accession 0000320193-20-000052
"""
from __future__ import annotations

import argparse
import asyncio
import os

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.filing_text import FilingTextChunk
from data.schemas.pit import Modality
from data.sources.edgar.archive import archive
from data.sources.edgar.client import edgar_client
from data.sources.edgar.filings_text import CHARS_PER_TOKEN, extract_chunks
from data.sources.edgar.submissions import TEXT_FORMS, list_filings
from data.storage import get_storage


def _find_filing(filings, accession):
    for f in filings:
        if f.accession == accession:
            return f
    raise SystemExit(f"accession {accession} not in submissions index")


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
        filings = await list_filings(client, args.cik, forms=TEXT_FORMS)
        filing = _find_filing(filings, args.accession)
        print(f"target: {filing.form} accn={filing.accession} filed={filing.filed} report_date={filing.report_date}")
        body = await client.get_bytes(filing.primary_doc_url)
        print(f"  {len(body):,} bytes downloaded")

    archive(storage, args.cik.zfill(10), "primary", body, accession=args.accession)

    chunks = list(extract_chunks(body, filing.form))
    print(f"  extracted {len(chunks)} chunks across {len(set(c.item_id for c in chunks))} items")

    rows = []
    for c in chunks:
        rows.append({
            "entity_id": entity_id,
            "modality": Modality.FILINGS_TEXT.value,
            "effective_date": filing.report_date,
            "availability_date": filing.filed,
            "restated_at": None,
            "source": "sec_edgar:filing_html",
            "source_ref": filing.accession,
            "form": filing.form,
            "accession": filing.accession,
            "item_id": c.item_id,
            "item_title": c.item_title,
            "chunk_idx": c.chunk_idx,
            "n_chunks_in_item": c.n_chunks_in_item,
            "text": c.text,
            "n_chars": len(c.text),
            "n_tokens_approx": len(c.text) // CHARS_PER_TOKEN,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        pit.write(
            Modality.FILINGS_TEXT,
            df,
            partition_keys=FilingTextChunk.PARTITION_KEYS,
        )
        print(f"  wrote {len(df)} chunks to canonical filings_text")

        # Brief stats by item
        by_item = (
            df.groupby("item_id")
              .agg(chunks=("chunk_idx", "count"), chars=("n_chars", "sum"))
              .sort_index()
        )
        print("\nchunks per item:")
        for item_id, row in by_item.iterrows():
            print(f"  Item {item_id:>4}: {row['chunks']:>3} chunk(s)  {row['chars']:>7,} chars")


if __name__ == "__main__":
    asyncio.run(amain())
