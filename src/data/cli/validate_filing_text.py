"""Stage 4 validation: filing text extraction is correct and deterministic.

Asserts:
  - Item 1A and Item 2 are present for AAPL Q2-FY2020 10-Q
  - At least one chunk contains a known risk-factor phrase
  - Re-running extraction produces byte-identical chunks (determinism)
  - PIT: chunks query as-of after filed date; not before
"""
from __future__ import annotations

import gzip
import os
from datetime import date
from pathlib import Path

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.filing_text import FilingTextChunk
from data.schemas.pit import Modality
from data.sources.edgar.filings_text import extract_chunks
from data.storage import get_storage

AAPL_CIK = "0000320193"
AAPL_Q2_FY20_ACCN = "0000320193-20-000052"


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    entity_id = EntityRegistry.entity_id_from_cik(AAPL_CIK)

    print("=== Stage 4 validation: filings text extraction ===\n")

    # PIT before-filed check
    pre = pit.query(
        Modality.FILINGS_TEXT,
        entity_ids=[entity_id],
        as_of=date(2020, 4, 30),
        partition_keys=FilingTextChunk.PARTITION_KEYS,
    )
    assert pre.empty, f"text chunks visible before filing date: {len(pre)} rows"
    print(f"  [OK] PIT: nothing knowable as of 2020-04-30 (one day before filing)")

    # PIT after-filed check
    post = pit.query(
        Modality.FILINGS_TEXT,
        entity_ids=[entity_id],
        as_of=date(2020, 5, 2),
        partition_keys=FilingTextChunk.PARTITION_KEYS,
    )
    chunks = post[post["accession"] == AAPL_Q2_FY20_ACCN]
    print(f"  [OK] PIT: {len(chunks)} chunks knowable as of 2020-05-02")
    assert len(chunks) > 0

    # Items present
    items = set(chunks["item_id"])
    print(f"  items extracted: {sorted(items)}")
    assert "1A" in items, "Item 1A (Risk Factors) missing"
    assert "2" in items, "Item 2 (MD&A) missing"
    print("  [OK] required items 1A and 2 both present")

    # Content sanity: Item 1A in AAPL 10-Qs references COVID-19 in 2020
    item1a_text = " ".join(chunks[chunks["item_id"] == "1A"]["text"].tolist()).lower()
    item2_text = " ".join(chunks[chunks["item_id"] == "2"]["text"].tolist()).lower()
    # Pick phrases known to be in AAPL Q2-FY20 10-Q.
    assert "covid" in (item1a_text + item2_text), "expected 'COVID' to appear somewhere"
    print("  [OK] content sanity: 'COVID' present (expected for Q2-FY20 filing)")

    # Determinism: re-extract from the archived raw bytes; new chunks must
    # match the canonical ones byte-for-byte.
    archive_dir = Path(storage.root) / "raw" / "sec_edgar" / AAPL_CIK / "primary"
    bodies = list(archive_dir.glob("*.html.gz"))
    assert bodies, "expected archived primary HTML to be present"
    body = gzip.decompress(bodies[0].read_bytes())
    re_chunks = list(extract_chunks(body, "10-Q"))
    canonical_chunks = chunks.sort_values(["item_id", "chunk_idx"])
    re_chunks_sorted = sorted(re_chunks, key=lambda c: (c.item_id, c.chunk_idx))
    assert len(re_chunks) == len(chunks), (
        f"chunk count drift: re-extracted {len(re_chunks)} vs stored {len(chunks)}"
    )
    diffs = 0
    for stored, fresh in zip(canonical_chunks.itertuples(), re_chunks_sorted):
        if stored.text != fresh.text:
            diffs += 1
    assert diffs == 0, f"non-deterministic extraction: {diffs} chunks differ on re-run"
    print(f"  [OK] determinism: {len(re_chunks)} chunks byte-identical across two extractions")

    print("\n=== Stage 4 validation: PASS ===")

    # Stage 1.6 A8: item-boundary inline-mention robustness.
    test_item_boundary_inline_mention()


def test_item_boundary_inline_mention() -> None:
    """If "Item 1A" appears inline in body text (mid-paragraph, NOT at line
    start) as well as a real section heading, only the heading should produce
    an ExtractedItem and its body should be the real section text.

    The regex anchor `(?im)^\\s*item\\s+...` requires a line-start, so an
    inline mention should NOT match. This test pins that contract — if anyone
    loosens the regex in the future, this catches it."""
    from data.sources.edgar.filings_text import extract_items, _ITEM_10KQ_RE
    print("\n=== Stage 1.6 A8: item-boundary inline mention ===")

    # Synthetic plain text — mimics what html_to_text would produce.
    text = (
        "PART I\n\n"
        "Item 1. Business\n\n"
        "The Company sells widgets. Discussion is similar to Item 1A above; "
        "see the risk factors there. We are subject to litigation as noted.\n\n"
        "Item 1A. Risk Factors\n\n"
        "The Company faces a number of significant risks. Competition is intense "
        "and supplier concentration is high.\n\n"
        "Item 2. Properties\n\n"
        "The Company owns offices in California.\n"
    )

    # Direct check: regex should match 3 headings, not the inline mention.
    matches = list(_ITEM_10KQ_RE.finditer(text))
    matched_ids = [m.group("id").upper() for m in matches]
    assert matched_ids == ["1", "1A", "2"], (
        f"expected 3 line-start headings, got: {matched_ids}. "
        f"Inline mention of 'Item 1A above' must NOT match."
    )
    print(f"  [OK] line-start anchor matches {len(matched_ids)} headings; inline mention skipped")

    # End-to-end: extract_items should return one entry per item with the
    # right body text.
    items = extract_items(text, form="10-K")
    by_id = {it.item_id: it for it in items}
    assert "1A" in by_id, "Item 1A heading missing from extraction"
    body_1a = by_id["1A"].text
    assert "significant risks" in body_1a, (
        f"Item 1A body is the real section text, got: {body_1a[:80]!r}"
    )
    # The inline mention text ("Discussion is similar to Item 1A above") is in
    # Item 1's body, not Item 1A's body. Verify.
    body_1 = by_id["1"].text
    assert "Item 1A above" in body_1, "inline mention should remain inside Item 1's body"
    print(f"  [OK] extract_items produced {len(items)} items; bodies are the real sections")
    print("\n=== A8 item-boundary inline mention: PASS ===")


if __name__ == "__main__":
    main()
