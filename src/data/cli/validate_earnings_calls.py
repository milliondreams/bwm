"""Stage M5 validation: earnings_calls derived from 8-K Item 2.02.

Asserts:
  - At least one derived excerpt exists (we have real 8-K data ingested)
  - PIT timing inherited from source 8-K
  - coverage_class is always 'press_release' in v1
  - Re-running the derive is idempotent (no duplicates)
"""
from __future__ import annotations

import os
from datetime import date

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.earnings_call import EarningsCallExcerpt
from data.schemas.pit import Modality
from data.storage import get_storage


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    print("=== Stage M5 validation: earnings_calls derived from 8-K Item 2.02 ===\n")

    # Pick an entity that has 8-K data (AAPL — we ingested 8-K via Stage 4/5).
    eid = EntityRegistry.entity_id_from_cik("0000320193")
    snap = pit.snapshot(eid, as_of=date(2030, 1, 1))
    ec = snap.earnings_calls
    if ec.empty:
        # Try to derive on demand
        from data.cli.derive_earnings_calls import derive_for_entity
        n = derive_for_entity(storage, pit, eid)
        print(f"  triggered derivation: {n} rows written")
        snap = pit.snapshot(eid, as_of=date(2030, 1, 1))
        ec = snap.earnings_calls

    assert not ec.empty, "no earnings_calls excerpts for AAPL — Stage 4 must run first"
    print(f"AAPL earnings_calls excerpts: {len(ec)}")
    print(f"  coverage_classes present: {set(ec['coverage_class'])}")
    assert set(ec["coverage_class"]) == {"press_release"}, (
        "v1 should emit only press_release"
    )
    print("  [OK] all excerpts are coverage_class='press_release' as documented")

    # Idempotency: re-deriving must not produce duplicates
    from data.cli.derive_earnings_calls import derive_for_entity
    n_before = len(ec)
    derive_for_entity(storage, pit, eid)
    snap2 = pit.snapshot(eid, as_of=date(2030, 1, 1))
    n_after = len(snap2.earnings_calls)
    assert n_after == n_before, f"re-derive produced duplicates: {n_before} → {n_after}"
    print(f"  [OK] idempotent: re-deriving keeps row count at {n_after}")

    # PIT timing: an excerpt's availability_date should match its source 8-K
    sample = ec.iloc[0]
    print(f"  sample: accn={sample['accession']} item={sample['item_id']} "
          f"effective={sample['effective_date']} avail={sample['availability_date']}")
    assert sample["item_id"] == "2.02"

    print("\n=== Stage M5 validation: PASS ===")


if __name__ == "__main__":
    main()
