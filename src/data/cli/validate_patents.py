"""Stage M3 validation: patents pipeline end-to-end.

PatentsView v1 was deprecated in 2024. The v2 API at search.patentsview.org
isn't reachable from this sandbox; the live fetch path is therefore tested
in environments with internet access by running `ingest_patents_one`. This
validation exercises everything OTHER than the network call: schema,
PIT contract, restatement-free expectation, snapshot integration.

Asserts:
  - Synthetic patents land in canonical with correct shape and PIT timing
    (application_date as effective; grant_date as availability — granted
    patents become public on grant_date)
  - PIT before grant_date returns nothing; PIT after returns the patent
  - The CPC + inventors JSON columns roundtrip correctly
"""
from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.patent import Patent
from data.schemas.pit import Modality
from data.storage import get_storage


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    print("=== Stage M3 validation: patents pipeline ===")
    print("\n(Note: live PatentsView fetch hits search.patentsview.org which")
    print(" isn't reachable from this sandbox. Connector code is exercised via")
    print(" `ingest_patents_one` in environments with API access. This validation")
    print(" exercises schema + PIT engine + snapshot integration with synthetic")
    print(" patents that mirror real PatentsView shapes.)\n")

    eid = EntityRegistry.entity_id_from_cik("0000320193")

    # Wipe any prior data for this entity so the test is repeatable.
    from data.pit.engine import _entity_path
    from pathlib import Path
    canon = _entity_path(Modality.PATENTS, eid)
    if storage.exists(canon):
        (Path(storage.root) / canon).unlink()

    # Two synthetic AAPL patents.
    cpc_a = ["G06F 3/0488", "H04L 9/3247"]
    inv_a = ["John Smith", "Jane Doe"]
    rows = pd.DataFrame([
        {
            "entity_id": eid,
            "modality": Modality.PATENTS.value,
            "effective_date": date(2022, 5, 1),  # application_date
            "availability_date": date(2024, 1, 16),  # grant_date
            "restated_at": None,
            "source": "test:patentsview",
            "source_ref": "US11900000",
            "patent_number": "US11900000",
            "application_date": date(2022, 5, 1),
            "grant_date": date(2024, 1, 16),
            "assignee_name_as_filed": "Apple Inc.",
            "title": "Foldable display interaction",
            "abstract": "A foldable display device provides...",
            "claims_count": 18,
            "cpc_classifications": json.dumps(cpc_a),
            "inventors": json.dumps(inv_a),
        },
        {
            "entity_id": eid,
            "modality": Modality.PATENTS.value,
            "effective_date": date(2023, 2, 15),
            "availability_date": date(2024, 6, 4),
            "restated_at": None,
            "source": "test:patentsview",
            "source_ref": "US12000000",
            "patent_number": "US12000000",
            "application_date": date(2023, 2, 15),
            "grant_date": date(2024, 6, 4),
            "assignee_name_as_filed": "Apple Inc.",
            "title": "Neural co-processor pipelining",
            "abstract": "A neural co-processor for on-device inference...",
            "claims_count": 22,
            "cpc_classifications": json.dumps(["G06N 3/063"]),
            "inventors": json.dumps(["Alice Lee"]),
        },
    ])
    pit.write(Modality.PATENTS, rows, partition_keys=Patent.PARTITION_KEYS)
    print(f"  wrote {len(rows)} synthetic patents to canonical")

    # PIT contract probes
    pre = pit.snapshot(eid, as_of=date(2024, 1, 15))
    assert len(pre.patents) == 0, f"patents visible before grant: {len(pre.patents)}"
    print(f"  [OK] PIT: no patents knowable on 2024-01-15 (one day before first grant)")

    mid = pit.snapshot(eid, as_of=date(2024, 3, 1))
    assert len(mid.patents) == 1
    assert mid.patents.iloc[0]["patent_number"] == "US11900000"
    print(f"  [OK] PIT: 1 patent visible after first grant, before second")

    after = pit.snapshot(eid, as_of=date(2024, 12, 31))
    assert len(after.patents) == 2
    patent_numbers = sorted(after.patents["patent_number"])
    assert patent_numbers == ["US11900000", "US12000000"]
    print(f"  [OK] PIT: both patents visible after both grants")

    # CPC + inventors JSON roundtrip
    p1 = after.patents[after.patents["patent_number"] == "US11900000"].iloc[0]
    assert json.loads(p1["cpc_classifications"]) == cpc_a, "CPC classifications lost in roundtrip"
    assert json.loads(p1["inventors"]) == inv_a, "inventors lost in roundtrip"
    print(f"  [OK] JSON-encoded list columns (CPC, inventors) roundtrip cleanly")

    print("\n=== Stage M3 validation: PASS ===")


if __name__ == "__main__":
    main()
