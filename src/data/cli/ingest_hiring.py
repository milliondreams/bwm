"""Stage M6 driver: ingest BLS JOLTS for the curated NAICS industry set.

Run:
    BLS_API_KEY=... uv run python -m data.cli.ingest_hiring \
        --start-year 2015 --end-year 2026

This is an industry-aggregated modality (no per-CIK fan-out). Each NAICS
industry × measure becomes a pseudo-entity with entity_id
"bls:naics-{6-digit-code}". Downstream training looks up the focal
company's NAICS code and attaches the matching JOLTS series as a context
feature, per Phase A.2 / Stage M6 of the plan.

PIT semantics inherited from the source: effective_date = period_end,
availability_date = release_date (Tuesday ~42 days post month-end).

The BLS v2 API accepts ≤25 series/request without a key (≤50 with). We
group the curated series list into ≤25-series chunks so unkeyed access
still completes in one or two requests.
"""
from __future__ import annotations

import argparse
import os
from datetime import date as date_cls

import pandas as pd

from data.observability.run_id import tag as _tag
from data.pit.engine import PITEngine
from data.schemas.hiring_observation import HiringObservation
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.hiring.bls import (
    DEFAULT_NAICS_INDUSTRIES,
    MEASURES,
    build_default_seriesids,
    fetch_jolts_series,
)
from data.storage import get_storage


# Convenient lookup: 6-digit NAICS → human label, and BLS DataElement → measure name.
_NAICS_LABEL = dict(DEFAULT_NAICS_INDUSTRIES)
_MEASURE_NAME = dict(MEASURES)  # "JOL" → "job_openings", etc.


def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=date_cls.today().year)
    ap.add_argument(
        "--chunk-size", type=int, default=25,
        help="series per BLS request (25 unkeyed, 50 with --api-key/BLS_API_KEY)",
    )
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    api_key = os.environ.get("BLS_API_KEY")
    series_tuples = build_default_seriesids()  # [(series_id, naics, measure_name), ...]
    print(
        f"=== BLS JOLTS ingest: {len(series_tuples)} series, "
        f"{args.start_year}-{args.end_year}, chunk={args.chunk_size}, "
        f"key={'set' if api_key else 'unset'} ==="
    )

    # Map series_id back to (naics, measure_name) since the BLS response
    # only carries the seriesID, not our metadata tuples.
    series_meta: dict[str, tuple[str, str]] = {
        sid: (naics, measure) for (sid, naics, measure) in series_tuples
    }
    sids = [t[0] for t in series_tuples]

    total_obs = 0
    total_series_with_data = 0
    by_entity: dict[str, list[dict]] = {}

    for chunk_idx, sid_chunk in enumerate(_chunked(sids, args.chunk_size)):
        try:
            observations = list(
                fetch_jolts_series(
                    sid_chunk,
                    start_year=args.start_year,
                    end_year=args.end_year,
                    api_key=api_key,
                )
            )
        except Exception as e:  # noqa: BLE001
            print(_tag(f"  [ERR] chunk {chunk_idx}: {type(e).__name__}: {e}"))
            log_parse(
                storage, "bls", f"chunk-{chunk_idx}", "hiring",
                "fetch_failed", error_message=f"{type(e).__name__}: {e}",
            )
            continue
        print(_tag(f"  chunk {chunk_idx}: {len(sid_chunk)} series → {len(observations)} obs"))
        for obs in observations:
            meta = series_meta.get(obs.series_id)
            if meta is None:
                # BLS sometimes returns a series we didn't request; skip.
                continue
            naics, measure = meta
            entity_id = f"bls:naics-{naics}"
            by_entity.setdefault(entity_id, []).append(
                {
                    "entity_id": entity_id,
                    "modality": Modality.HIRING.value,
                    "effective_date": obs.period_end,
                    "availability_date": obs.release_date,
                    "restated_at": None,
                    "source": "bls:jolts",
                    "source_ref": f"{obs.series_id}:{obs.period_end.isoformat()}",
                    "naics_industry": naics,
                    "naics_label": _NAICS_LABEL.get(naics, ""),
                    "measure": measure,
                    "value": obs.value,
                    "seasonal_adjustment": "seasonally_adjusted",
                }
            )
            total_obs += 1

    for entity_id, rows in by_entity.items():
        df = pd.DataFrame(rows)
        pit.write(Modality.HIRING, df, partition_keys=HiringObservation.PARTITION_KEYS)
        log_parse(
            storage, entity_id, f"jolts:{entity_id}", "hiring",
            "ok_with_records", n_records=len(rows),
        )
        total_series_with_data += 1
        print(_tag(f"  [OK] {entity_id}: wrote {len(rows):,} observations"))

    # Report which requested series returned nothing.
    requested_entities = {f"bls:naics-{naics}" for (_, naics, _) in series_tuples}
    empty = sorted(requested_entities - set(by_entity.keys()))
    for entity_id in empty:
        log_parse(storage, entity_id, f"jolts:{entity_id}", "hiring", "ok_no_records")

    print("\n=== summary ===")
    print(f"  entities populated: {total_series_with_data}/{len(requested_entities)}")
    print(f"  total observations: {total_obs:,}")
    if empty:
        print(f"  entities with 0 observations: {len(empty)}")
        for eid in empty[:5]:
            print(f"    - {eid}")


if __name__ == "__main__":
    main()
