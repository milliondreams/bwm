"""Stage M3 driver: ingest patents for one entity via PatentsView.

The PatentsView v1 API requires no API key. Some deployments rate-limit
heavy use; pass `PATENTSVIEW_API_KEY` for elevated quota if you have one.

Run:
    uv run python -m data.cli.ingest_patents_one --cik 0000320193

Resolves the CIK → assignee name via the registry (entity name), queries
PatentsView, writes per-CIK partitioned canonical patents.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date as date_cls

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.patent import Patent
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.patents.uspto import fetch_patents_by_assignee
from data.storage import get_storage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cik", required=True)
    ap.add_argument("--assignee-name", default="",
                    help="override assignee name (default: lookup from registry)")
    ap.add_argument("--start-grant", default="2000-01-01")
    ap.add_argument("--end-grant", default=date_cls.today().isoformat())
    ap.add_argument("--max-patents", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    registry = EntityRegistry(storage)

    reg = registry.load()
    row = reg[reg["cik"] == args.cik.zfill(10)]
    if row.empty:
        raise SystemExit(f"CIK {args.cik} not in registry")
    assignee = args.assignee_name or str(row.iloc[0]["name"])
    entity_id = EntityRegistry.entity_id_from_cik(args.cik)
    print(f"querying PatentsView for assignee={assignee!r}  entity_id={entity_id}")

    api_key = os.environ.get("PATENTSVIEW_API_KEY")
    patents = []
    try:
        for p in fetch_patents_by_assignee(
            assignee,
            start_grant_date=date_cls.fromisoformat(args.start_grant),
            end_grant_date=date_cls.fromisoformat(args.end_grant),
            api_key=api_key,
        ):
            patents.append(p)
            if args.max_patents > 0 and len(patents) >= args.max_patents:
                break
    except Exception as e:  # noqa: BLE001
        log_parse(storage, args.cik.zfill(10), f"patentsview:{assignee}",
                  "patents", "fetch_failed", error_message=f"{type(e).__name__}: {e}")
        raise

    if not patents:
        log_parse(storage, args.cik.zfill(10), f"patentsview:{assignee}",
                  "patents", "ok_no_records")
        print(f"  no patents found for assignee={assignee!r}")
        return

    rows = [
        {
            "entity_id": entity_id,
            "modality": Modality.PATENTS.value,
            "effective_date": p.application_date,
            "availability_date": p.grant_date,
            "restated_at": None,
            "source": "uspto:patentsview",
            "source_ref": p.patent_number,
            "patent_number": p.patent_number,
            "application_date": p.application_date,
            "grant_date": p.grant_date,
            "assignee_name_as_filed": p.assignee_name_as_filed,
            "title": p.title,
            "abstract": p.abstract,
            "claims_count": p.claims_count,
            "cpc_classifications": json.dumps(p.cpc_classifications),
            "inventors": json.dumps(p.inventors),
        }
        for p in patents
    ]
    df = pd.DataFrame(rows)
    pit.write(Modality.PATENTS, df, partition_keys=Patent.PARTITION_KEYS)
    log_parse(storage, args.cik.zfill(10), f"patentsview:{assignee}",
              "patents", "ok_with_records", n_records=len(rows))
    print(f"  wrote {len(rows):,} patents; grant_date range "
          f"{min(p.grant_date for p in patents)} to {max(p.grant_date for p in patents)}")


if __name__ == "__main__":
    main()
