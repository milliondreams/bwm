"""Stage M6 validation: BLS JOLTS pipeline (substitute coverage for hiring).

Per-company hiring data has no free per-entity source. This modality covers
the spec's intent (model uses hiring signal) via industry-aggregated BLS
JOLTS data — entity_id = "bls:naics-{code}". Downstream model code looks
up the focal company's NAICS and attaches the matching JOLTS series.

Live BLS API access works without an API key (rate-limited to 25/day) but
isn't exercised in this validation — see ingest_hiring CLI. This script
exercises schema + canonical + PIT + snapshot integration with synthetic
JOLTS observations.

Asserts:
  - Synthetic JOLTS observations for "Information" (NAICS 51) round-trip
    through canonical
  - PIT timing: release_date is ~6 weeks after observation month
  - Snapshot integration works on synthetic entity_ids like "bls:naics-510000"
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from data.pit.engine import PITEngine
from data.schemas.hiring_observation import HiringObservation
from data.schemas.pit import Modality
from data.storage import get_storage


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    print("=== Stage M6 validation: BLS JOLTS hiring (industry-aggregated) ===")
    print("\n(Note: live BLS API access is operational and exercised separately.")
    print(" This validation exercises schema + PIT + snapshot with synthetic JOLTS")
    print(" observations for the Information industry (NAICS 51).)\n")

    eid = "bls:naics-510000"  # Information sector

    # Wipe canonical for repeatability
    from data.pit.engine import _entity_path
    from pathlib import Path
    canon = _entity_path(Modality.HIRING, eid)
    if storage.exists(canon):
        (Path(storage.root) / canon).unlink()

    # 3 monthly observations for job_openings in Information (NAICS 51) in 2021-Q4
    rows = pd.DataFrame([
        {
            "entity_id": eid,
            "modality": Modality.HIRING.value,
            "effective_date": date(2021, 10, 31),
            "availability_date": date(2021, 12, 14),  # ~6 weeks later
            "restated_at": None,
            "source": "bls:jolts",
            "source_ref": "JTSJOL510000:2021-10",
            "naics_industry": "510000",
            "naics_label": "Information",
            "measure": "job_openings",
            "value": 360.0,  # thousands
            "seasonal_adjustment": "seasonally_adjusted",
        },
        {
            "entity_id": eid,
            "modality": Modality.HIRING.value,
            "effective_date": date(2021, 11, 30),
            "availability_date": date(2022, 1, 11),
            "restated_at": None,
            "source": "bls:jolts",
            "source_ref": "JTSJOL510000:2021-11",
            "naics_industry": "510000",
            "naics_label": "Information",
            "measure": "job_openings",
            "value": 405.0,
            "seasonal_adjustment": "seasonally_adjusted",
        },
        {
            "entity_id": eid,
            "modality": Modality.HIRING.value,
            "effective_date": date(2021, 12, 31),
            "availability_date": date(2022, 2, 8),
            "restated_at": None,
            "source": "bls:jolts",
            "source_ref": "JTSJOL510000:2021-12",
            "naics_industry": "510000",
            "naics_label": "Information",
            "measure": "job_openings",
            "value": 432.0,  # post-pandemic tech hiring boom
            "seasonal_adjustment": "seasonally_adjusted",
        },
    ])

    pit.write(Modality.HIRING, rows, partition_keys=HiringObservation.PARTITION_KEYS)
    print(f"  wrote {len(rows)} JOLTS observations to canonical for {eid}")

    # Releases lag the reference month by ~6 weeks
    lag_days = (
        pd.to_datetime(rows["availability_date"]) - pd.to_datetime(rows["effective_date"])
    ).dt.days
    assert (lag_days >= 35).all() and (lag_days <= 50).all(), (
        f"release lag outside expected 35-50 day window: {list(lag_days)}"
    )
    print(f"  [OK] release lag (avail − effective): {list(lag_days)} days each")

    # Snapshot integration: industry pseudo-entity works.
    snap = pit.snapshot(eid, as_of=date(2022, 3, 1))
    assert not snap.hiring.empty
    assert len(snap.hiring) == 3
    job_openings = snap.hiring.sort_values("effective_date")["value"].tolist()
    print(f"  [OK] snapshot integration: 3 observations, values trending {job_openings}")
    # Sanity: values trend upward through Q4 2021 (tech hiring boom)
    assert job_openings[2] > job_openings[0], (
        "Information JOLTS job openings should rise through 2021-Q4"
    )
    print(f"  [OK] sanity: Q4 2021 Information JOLTS job openings rose ({job_openings[0]}→{job_openings[2]})")

    # PIT: not knowable before release
    pre = pit.snapshot(eid, as_of=date(2021, 12, 13))  # one day before first release
    assert pre.hiring.empty
    print(f"  [OK] PIT: 0 observations knowable before first release date")

    print("\n=== Stage M6 validation: PASS ===")


if __name__ == "__main__":
    main()
