"""Stage M5 driver: derive earnings_calls canonical from existing filings_text.

Reads chunks from canonical/filings_text where form starts with "8-K" and
item_id == "2.02" (Results of Operations and Financial Condition — the
standard earnings-release 8-K item), writes them as EarningsCallExcerpt
rows to canonical/earnings_calls.

This is pure derivation: no SEC API calls, no new data, just a structural
projection of existing canonical data into a new modality. Re-running is
idempotent because PITEngine dedups on (accession, chunk_idx).

Run:
    uv run python -m data.cli.derive_earnings_calls
    uv run python -m data.cli.derive_earnings_calls --cik 0000320193
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine, _ENTITY_FILE_RE
from data.schemas.earnings_call import EarningsCallExcerpt
from data.schemas.pit import Modality
from data.storage import get_storage


def _project_chunks_to_excerpts(rows: pd.DataFrame) -> pd.DataFrame:
    """Project filings_text rows (form startswith '8-K', item_id == '2.02')
    into EarningsCallExcerpt rows."""
    out = []
    for r in rows.itertuples(index=False):
        out.append({
            "entity_id": r.entity_id,
            "modality": Modality.EARNINGS_CALLS.value,
            "effective_date": r.effective_date,
            "availability_date": r.availability_date,
            "restated_at": getattr(r, "restated_at", None),
            "source": "derived:filings_text/8-K_item_2.02",
            "source_ref": r.accession,
            "coverage_class": "press_release",
            "form": r.form,
            "accession": r.accession,
            "item_id": r.item_id,
            "chunk_idx": int(r.chunk_idx),
            "text": r.text,
            "n_chars": int(r.n_chars),
            "n_tokens_approx": int(r.n_tokens_approx),
        })
    return pd.DataFrame(out)


def derive_for_entity(storage, pit: PITEngine, entity_id: str) -> int:
    """Read one entity's filings_text canonical, derive press-release rows.

    Returns the number of EarningsCallExcerpt rows written.
    """
    from data.pit.engine import _entity_path
    src_path = _entity_path(Modality.FILINGS_TEXT, entity_id)
    if not storage.exists(src_path):
        return 0
    src = storage.read_parquet(src_path)
    matched = src[
        src["form"].astype(str).str.startswith("8-K")
        & (src["item_id"].astype(str) == "2.02")
    ]
    if matched.empty:
        return 0
    derived = _project_chunks_to_excerpts(matched)
    pit.write(
        Modality.EARNINGS_CALLS, derived, partition_keys=EarningsCallExcerpt.PARTITION_KEYS
    )
    return len(derived)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cik", default="", help="single CIK; empty → all entities with filings_text")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    if args.cik:
        eids = [EntityRegistry.entity_id_from_cik(args.cik)]
    else:
        # Discover entities by scanning the filings_text directory.
        eids = []
        for p in storage.list(f"canonical/{Modality.FILINGS_TEXT.value}"):
            fname = p.rsplit("/", 1)[-1]
            m = _ENTITY_FILE_RE.match(fname)
            if m:
                eids.append(m.group("eid"))

    print(f"deriving earnings_calls for {len(eids)} entit(ies)…")
    total = 0
    for eid in eids:
        n = derive_for_entity(storage, pit, eid)
        if n:
            print(f"  + {eid}: {n} excerpts derived")
        total += n
    print(f"\ntotal earnings_call excerpts written: {total}")


if __name__ == "__main__":
    main()
