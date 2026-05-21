"""Stage M2 validation: FRED macro vintage-aware PIT semantics.

The live `fetch_vintages` call requires `FRED_API_KEY`. This validation
exercises the schema, restatement logic, and PIT contract with synthetic
vintages so the pipeline is provable independently of the API key state.

Asserts:
  - Synthetic CPI vintages for 2020-06 (first release, then a revision)
    land in canonical with the second vintage's release_date marked as
    `restated_at` on the original.
  - PIT contract: as-of the original release date returns the original
    value; as-of after the revision returns the revised value.
  - Snapshot integration: `pit.snapshot(entity="fred:CPIAUCSL")` returns
    a non-empty macro DataFrame.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from data.pit.engine import PITEngine
from data.schemas.macro_observation import MacroObservation
from data.schemas.pit import Modality
from data.storage import get_storage


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    print("=== Stage M2 validation: FRED macro PIT vintages ===")
    print("\n(Note: live FRED fetch requires FRED_API_KEY. This validation runs")
    print(" synthetic vintages through the full pipeline to prove the schema +")
    print(" restatement logic + PIT contract work end-to-end. Switching to the")
    print(" live source is a `fetch_vintages(...)` call once the key is set.)\n")

    # Synthetic CPI for observation 2020-06:
    # - First release: 2020-07-14, value 257.797
    # - Revised:       2020-08-12, value 258.094 (small upward revision)
    # - Re-revised:    2021-02-26, value 258.273
    series = "CPIAUCSL"
    eid = f"fred:{series}"
    obs = date(2020, 6, 30)

    rows = pd.DataFrame([
        {
            "entity_id": eid,
            "modality": Modality.MACRO.value,
            "effective_date": obs,
            "availability_date": date(2020, 7, 14),
            "restated_at": None,
            "source": "test:fred-vintage",
            "source_ref": f"{series}@2020-07-14",
            "series_id": series,
            "value": 257.797,
            "series_title": "CPI for All Urban Consumers: All Items",
            "units": "Index 1982-1984=100",
            "frequency": "Monthly",
        },
        {
            "entity_id": eid,
            "modality": Modality.MACRO.value,
            "effective_date": obs,
            "availability_date": date(2020, 8, 12),
            "restated_at": None,
            "source": "test:fred-vintage",
            "source_ref": f"{series}@2020-08-12",
            "series_id": series,
            "value": 258.094,
            "series_title": "CPI for All Urban Consumers: All Items",
            "units": "Index 1982-1984=100",
            "frequency": "Monthly",
        },
        {
            "entity_id": eid,
            "modality": Modality.MACRO.value,
            "effective_date": obs,
            "availability_date": date(2021, 2, 26),
            "restated_at": None,
            "source": "test:fred-vintage",
            "source_ref": f"{series}@2021-02-26",
            "series_id": series,
            "value": 258.273,
            "series_title": "CPI for All Urban Consumers: All Items",
            "units": "Index 1982-1984=100",
            "frequency": "Monthly",
        },
    ])
    # Mark restatements: each non-latest vintage gets its successor's release as restated_at.
    from data.cli.ingest_macro import _mark_restatements
    rows = _mark_restatements(rows)
    got = list(rows["restated_at"])
    # Normalize NaN → None for stable comparison; groupby+shift produces NaN for the tail.
    got_norm = [d if isinstance(d, date) else None for d in got]
    expected_restated = [date(2020, 8, 12), date(2021, 2, 26), None]
    assert got_norm == expected_restated, f"restated_at marking wrong: {got_norm}"
    print(f"  [OK] restated_at marking: {got_norm}")

    # Wipe any prior state for this entity to make the test repeatable.
    from data.pit.engine import _entity_path
    from pathlib import Path
    canon = _entity_path(Modality.MACRO, eid)
    if storage.exists(canon):
        (Path(storage.root) / canon).unlink()

    pit.write(Modality.MACRO, rows, partition_keys=MacroObservation.PARTITION_KEYS)
    print(f"  wrote {len(rows)} vintaged observations to canonical")

    # PIT probes
    probes = [
        ("just after original release", date(2020, 7, 15), 257.797),
        ("between original and revision", date(2020, 7, 30), 257.797),
        ("just after first revision", date(2020, 8, 13), 258.094),
        ("just after second revision", date(2021, 3, 1), 258.273),
    ]
    for label, as_of, expected in probes:
        result = pit.query(
            Modality.MACRO,
            entity_ids=[eid],
            as_of=as_of,
            extra_filters={"series_id": series},
            partition_keys=MacroObservation.PARTITION_KEYS,
        )
        result = result[result["effective_date"] == obs]
        assert len(result) == 1, f"{label}: expected 1 row, got {len(result)}"
        got_value = float(result.iloc[0]["value"])
        assert abs(got_value - expected) < 1e-6, (
            f"{label}: as_of={as_of}, expected {expected}, got {got_value}"
        )
        print(f"  [OK] as_of={as_of}  ({label:>32s}): value = {got_value}")

    # Snapshot integration
    snap = pit.snapshot(eid, as_of=date(2021, 1, 1))
    assert not snap.macro.empty, "snapshot.macro is empty after ingest"
    print(f"  [OK] snapshot integration: snap.macro has {len(snap.macro)} row(s)")

    print("\n=== Stage M2 validation: PASS ===")


if __name__ == "__main__":
    main()
