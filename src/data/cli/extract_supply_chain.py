"""Stage 6 driver: extract supply-chain edges from one filing's canonical text.

Loads the chunks for a given (CIK, accession) from canonical/filings_text,
filters to Item 1 (Business) and Item 7 (MD&A) which carry counterparty
disclosures, runs the LLM extractor, resolves names against the registry,
and writes edges to canonical/supply_chain.

Run:
    uv run python -m data.cli.extract_supply_chain \
        --cik 0000320193 --accession 0000320193-23-000106 \
        --items 1 7

This produces a small per-filing canonical that downstream snapshots roll
up to quarterly graph slices via `build_graph.py`.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.filing_text import FilingTextChunk
from data.schemas.pit import Modality
from data.schemas.supply_chain import SupplyChainEdge
from data.sources.edgar.ie.extract import EdgeExtractor, _evidence_hash
from data.sources.edgar.ie.resolve import NameResolver
from data.storage import get_storage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cik", required=True)
    ap.add_argument("--accession", required=True)
    ap.add_argument("--items", nargs="+", default=["1", "7"],
                    help="Item IDs to extract from (default: 1 7)")
    ap.add_argument("--max-chunks-per-item", type=int, default=3,
                    help="Cap LLM calls — default 3 chunks per item (budget control)")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    registry = EntityRegistry(storage)
    entity_id = EntityRegistry.entity_id_from_cik(args.cik)

    # Load this entity's filing-text canonical and filter to the target filing.
    text_rows = storage.read_parquet(
        f"canonical/{Modality.FILINGS_TEXT.value}/entity={entity_id}.parquet"
    )
    rows = text_rows[
        (text_rows["accession"] == args.accession)
        & (text_rows["item_id"].isin(args.items))
    ].sort_values(["item_id", "chunk_idx"])
    if rows.empty:
        raise SystemExit(
            f"no canonical text chunks for entity={entity_id} accession={args.accession}"
        )

    # Look up issuer name for prompt context.
    reg_df = registry.load()
    issuer_row = reg_df[reg_df["cik"] == args.cik.zfill(10)]
    issuer_name = str(issuer_row.iloc[0]["name"]) if not issuer_row.empty else "Issuer"
    print(f"Issuer: {issuer_name}  ({entity_id})")

    resolver = NameResolver(reg_df)
    extractor = EdgeExtractor()

    # Walk chunks; cap per-item to control LLM budget.
    per_item_counts: dict[str, int] = {}
    edges_written = 0
    edge_rows: list[dict] = []
    filing_form = ""
    filing_filed = None
    filing_report = None
    for r in rows.itertuples():
        if filing_form == "":
            filing_form, filing_filed, filing_report = r.form, r.availability_date, r.effective_date
        item_id = r.item_id
        per_item_counts.setdefault(item_id, 0)
        if per_item_counts[item_id] >= args.max_chunks_per_item:
            continue
        per_item_counts[item_id] += 1
        print(f"  extracting Item {item_id} chunk {r.chunk_idx}…")
        raw_edges = extractor.extract(issuer_name, item_id, r.text)
        for re_edge in raw_edges:
            res = resolver.resolve(re_edge.target_name)
            evhash = _evidence_hash(re_edge.evidence_span)
            edge_rows.append({
                "entity_id": entity_id,
                "modality": Modality.SUPPLY_CHAIN.value,
                "effective_date": filing_report,
                "availability_date": filing_filed,
                "restated_at": None,
                "source": "sec_edgar:ie",
                "source_ref": args.accession,
                "target_entity_id": res.entity_id,
                "target_name_as_filed": re_edge.target_name,
                "relationship_type": re_edge.relationship_type,
                "weight_qualifier": re_edge.weight_qualifier,
                "evidence_span": re_edge.evidence_span,
                "evidence_hash": evhash,
                "extractor_model": extractor.model,
                "confidence": re_edge.confidence,
                "form": filing_form,
                "accession": args.accession,
                "item_id": item_id,
            })
            edges_written += 1
            tag = "RES" if res.is_resolved else "unr"
            print(
                f"    + [{tag} {res.score:>5.1f}] {re_edge.relationship_type:>10s}: "
                f"{re_edge.target_name!r} → {res.entity_id}"
            )

    if edge_rows:
        df = pd.DataFrame(edge_rows)
        pit.write(Modality.SUPPLY_CHAIN, df, partition_keys=SupplyChainEdge.PARTITION_KEYS)
        print(f"\nwrote {edges_written} edges to canonical supply_chain")
    else:
        print("\nno edges extracted")


if __name__ == "__main__":
    main()
