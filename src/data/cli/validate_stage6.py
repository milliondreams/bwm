"""Stage 6 validation: supply-chain extraction pipeline.

Validates every component of the Stage 6 pipeline **except** the live LLM
call, which is gated by the Anthropic account's billing state and is not a
code concern. Specifically asserts:

  1. `parse_response` correctly extracts edges from realistic LLM JSON output,
     and tolerates leading/trailing slop, partial fields, and malformed rows.
  2. `NameResolver` resolves common supplier names ("Foxconn", "TSMC",
     "Samsung Electronics") to the right CIKs when present in registry, and
     yields stable `unresolved:` placeholders otherwise.
  3. End-to-end: synthetic LLM edges → resolver → PIT canonical → query.
     The full data path (schema, PIT engine partition keys, dedup,
     query) is exercised under realistic Apple/10-K shape.

For the live LLM extraction, run `extract_supply_chain.py` directly; it
will raise on insufficient credit so the failure mode is visible.
"""
from __future__ import annotations

import hashlib
import os
from datetime import date

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.pit import Modality
from data.schemas.supply_chain import SupplyChainEdge
from data.sources.edgar.ie.extract import RawEdge, parse_response
from data.sources.edgar.ie.resolve import NameResolver, normalize
from data.storage import get_storage

AAPL_CIK = "0000320193"
AAPL_FY24_10K = "0000320193-24-000123"


# Realistic LLM output for a Stage 6 extraction on AAPL Item 1 / Item 7.
# Apple's 10-K explicitly discusses these relationships.
SAMPLE_LLM_OUTPUT = """[
  {
    "target_name": "Foxconn",
    "relationship_type": "supplier",
    "weight_qualifier": "primary contract manufacturer",
    "evidence_span": "Substantially all of the Company's hardware products are manufactured by outsourcing partners that are located primarily in Asia, including Foxconn.",
    "confidence": 0.92
  },
  {
    "target_name": "TSMC",
    "relationship_type": "supplier",
    "weight_qualifier": "primary foundry for custom silicon",
    "evidence_span": "The Company's custom-designed processors are manufactured by TSMC.",
    "confidence": 0.94
  },
  {
    "target_name": "Samsung Electronics",
    "relationship_type": "supplier",
    "weight_qualifier": "display and memory components",
    "evidence_span": "The Company sources certain memory components from Samsung Electronics.",
    "confidence": 0.85
  },
  {
    "target_name": "Qualcomm",
    "relationship_type": "supplier",
    "weight_qualifier": "modems",
    "evidence_span": "Qualcomm supplies wireless modems used in certain iPhone models.",
    "confidence": 0.88
  },
  {
    "target_name": "Google",
    "relationship_type": "competitor",
    "weight_qualifier": "smartphone OS",
    "evidence_span": "We compete with Google in mobile operating systems and services.",
    "confidence": 0.90
  }
]"""


# Edge cases the parser should handle gracefully.
MALFORMED_OUTPUTS = [
    # Slop before/after the JSON
    'Here are the edges:\n```json\n[{"target_name":"X","relationship_type":"supplier","evidence_span":"foo","confidence":0.5}]\n```\nDone.',
    # Empty array
    "[]",
    # Missing required fields
    '[{"target_name":""}]',
    # Wrong outer type
    '{"oops": []}',
    # Garbage
    "totally not json",
]


def test_parse_response() -> None:
    print("\n--- testing parse_response ---")
    edges = parse_response(SAMPLE_LLM_OUTPUT)
    print(f"  parsed {len(edges)} edges from sample LLM output")
    assert len(edges) == 5, f"expected 5 edges, got {len(edges)}"
    # The first one is Foxconn (a supplier)
    assert edges[0].target_name == "Foxconn"
    assert edges[0].relationship_type == "supplier"
    assert edges[0].confidence == 0.92
    print("  [OK] sample extraction matches expected shape")

    # Malformed inputs should either yield empty list or skip bad rows
    for i, bad in enumerate(MALFORMED_OUTPUTS):
        out = parse_response(bad)
        # Edge 0 (slop+json) should yield 1 edge; others should yield 0
        expected = 1 if i == 0 else 0
        assert len(out) == expected, (
            f"malformed case {i} ({bad[:40]!r}…): expected {expected} edges, got {len(out)}"
        )
    print(f"  [OK] {len(MALFORMED_OUTPUTS)} malformed-input cases handled gracefully")


def test_resolver() -> None:
    print("\n--- testing NameResolver ---")
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    registry = EntityRegistry(storage)
    reg_df = registry.load()
    resolver = NameResolver(reg_df, min_score=80.0)

    # Apple Inc. is definitely in the SEC registry
    r = resolver.resolve("Apple Inc.")
    print(f"  'Apple Inc.' → {r.entity_id}  score={r.score:.1f}  resolved={r.is_resolved}")
    assert r.is_resolved and r.entity_id == "us-cik-0000320193"

    # Tickers route through the alias side-index
    r = resolver.resolve("AAPL")
    print(f"  'AAPL'      → {r.entity_id}  score={r.score:.1f}  resolved={r.is_resolved}")
    assert r.is_resolved and r.entity_id == "us-cik-0000320193"

    # Microsoft via slightly varied name
    r = resolver.resolve("Microsoft Corporation")
    print(f"  'Microsoft Corporation' → {r.entity_id}  score={r.score:.1f}  resolved={r.is_resolved}")
    assert r.is_resolved and r.entity_id == "us-cik-0000789019"

    # Foxconn — SEC name is "HON HAI PRECISION INDUSTRY CO LTD" or it may not
    # be registered at all (Taiwan-listed; not SEC-listed). The unresolved
    # path is the correct outcome — we want a stable placeholder, not a wrong
    # match.
    r = resolver.resolve("Foxconn")
    print(f"  'Foxconn'   → {r.entity_id}  score={r.score:.1f}  resolved={r.is_resolved}")
    # We accept either outcome but assert determinism: same input → same id
    r2 = resolver.resolve("Foxconn")
    assert r.entity_id == r2.entity_id, "resolver is non-deterministic"

    # Empty string → stable unresolved placeholder
    r = resolver.resolve("")
    print(f"  ''          → {r.entity_id}  resolved={r.is_resolved}")
    assert not r.is_resolved
    assert r.entity_id.startswith("unresolved:")

    print("  [OK] resolver produces stable, type-consistent results")


def _evidence_hash(span: str) -> str:
    return hashlib.sha1(span.encode("utf-8")).hexdigest()[:12]


def test_e2e_pipeline() -> None:
    """End-to-end: simulated LLM edges → resolve → write → query → assert."""
    print("\n--- testing end-to-end pipeline (no LLM call) ---")
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    registry = EntityRegistry(storage)
    reg_df = registry.load()
    resolver = NameResolver(reg_df, min_score=80.0)

    entity_id = EntityRegistry.entity_id_from_cik(AAPL_CIK)

    # Wipe any previous Stage 6 rows for this entity so this test is repeatable
    path = (
        f".data/canonical/{Modality.SUPPLY_CHAIN.value}/entity={entity_id}.parquet"
    )
    from pathlib import Path
    p = Path(path)
    if p.exists():
        p.unlink()

    # Parse the sample LLM output, resolve, write to canonical
    edges = parse_response(SAMPLE_LLM_OUTPUT)
    rows = []
    for e in edges:
        res = resolver.resolve(e.target_name)
        rows.append({
            "entity_id": entity_id,
            "modality": Modality.SUPPLY_CHAIN.value,
            "effective_date": date(2024, 9, 28),  # Apple FY24 period end
            "availability_date": date(2024, 11, 1),  # 10-K filing date
            "restated_at": None,
            "source": "sec_edgar:ie",
            "source_ref": AAPL_FY24_10K,
            "target_entity_id": res.entity_id,
            "target_name_as_filed": e.target_name,
            "relationship_type": e.relationship_type,
            "weight_qualifier": e.weight_qualifier,
            "evidence_span": e.evidence_span,
            "evidence_hash": _evidence_hash(e.evidence_span),
            "extractor_model": "mock:azure-openai:gpt-5.5",
            "confidence": e.confidence,
            "form": "10-K",
            "accession": AAPL_FY24_10K,
            "item_id": "1",
        })
    df = pd.DataFrame(rows)
    pit.write(Modality.SUPPLY_CHAIN, df, partition_keys=SupplyChainEdge.PARTITION_KEYS)
    print(f"  wrote {len(df)} mock edges to canonical supply_chain")

    # PIT before-filed: nothing knowable
    before = pit.query(
        Modality.SUPPLY_CHAIN,
        entity_ids=[entity_id],
        as_of=date(2024, 10, 31),
        partition_keys=SupplyChainEdge.PARTITION_KEYS,
    )
    assert before.empty, "edges visible before filing date"
    print("  [OK] PIT: no edges knowable as of 2024-10-31 (day before filing)")

    # PIT after-filed
    after = pit.query(
        Modality.SUPPLY_CHAIN,
        entity_ids=[entity_id],
        as_of=date(2024, 11, 2),
        partition_keys=SupplyChainEdge.PARTITION_KEYS,
    )
    assert len(after) == len(edges), f"expected {len(edges)} edges as-of-after, got {len(after)}"
    print(f"  [OK] PIT: {len(after)} edges knowable as of 2024-11-02")

    # Relationship type breakdown
    rels = after["relationship_type"].value_counts().to_dict()
    print(f"  relationship-type distribution: {rels}")
    assert "supplier" in rels and rels["supplier"] >= 3
    assert "competitor" in rels and rels["competitor"] >= 1
    print("  [OK] supplier and competitor edges both present")

    # Resolution rate
    resolved = after[~after["target_entity_id"].str.startswith("unresolved:")]
    unresolved = after[after["target_entity_id"].str.startswith("unresolved:")]
    print(f"  resolved: {len(resolved)} / {len(after)}  unresolved: {len(unresolved)}")
    # We want at least Google (a public co) to resolve.
    assert len(resolved) >= 1, "no edges resolved — resolver may be broken"
    print("  [OK] at least one target resolved to a registry CIK")

    # Determinism: re-writing the same rows is a no-op (idempotent)
    pit.write(Modality.SUPPLY_CHAIN, df, partition_keys=SupplyChainEdge.PARTITION_KEYS)
    after2 = pit.query(
        Modality.SUPPLY_CHAIN,
        entity_ids=[entity_id],
        as_of=date(2030, 1, 1),
        partition_keys=SupplyChainEdge.PARTITION_KEYS,
    )
    assert len(after2) == len(after), "duplicate writes inflated canonical"
    print(f"  [OK] idempotency: re-write produced no duplicates ({len(after2)} rows)")


def main() -> None:
    print("=== Stage 6 validation: supply-chain IE pipeline ===")
    print("\n(Note: live LLM extraction is gated by Anthropic credit; this validation")
    print(" exercises every other component including the JSON parser, resolver, schema,")
    print(" PIT engine, and canonical layer. Switching to the live extractor is a")
    print(" one-line change in extract_supply_chain.py once credits are available.)")

    test_parse_response()
    test_resolver()
    test_e2e_pipeline()

    print("\n=== Stage 6 validation: PASS ===")


if __name__ == "__main__":
    main()
