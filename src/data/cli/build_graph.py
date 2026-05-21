"""Roll up canonical/supply_chain into quarterly graph snapshots.

Spec §5.7 calls for time-varying edge weights; the planning layer reads
`graph/{quarter}/edges.parquet`. This CLI walks canonical supply_chain edges
and produces one snapshot per fiscal quarter, where each snapshot contains
the most recently-filed edge for every (source_entity, target_entity,
relationship_type) tuple known as of that quarter's end.

Quarter labels follow the convention `YYYY-Qn` based on calendar quarter
of the snapshot date. The snapshot date is the last day of the quarter; an
edge is included if availability_date <= snapshot_date and (no
restated_at or restated_at > snapshot_date).
"""
from __future__ import annotations

import argparse
import os
from datetime import date

import pandas as pd

from data.pit.engine import PITEngine, _ENTITY_FILE_RE
from data.schemas.pit import Modality
from data.schemas.supply_chain import SupplyChainEdge
from data.storage import get_storage


def _quarter_ends(start: date, end: date) -> list[date]:
    """Yield the last calendar day of every quarter in [start, end]."""
    out = []
    y, q = start.year, (start.month - 1) // 3 + 1
    while True:
        m = q * 3
        # Last day of month m
        if m in (3, 6, 9):
            d = date(y, m, 30)
        else:
            d = date(y, 12, 31)
        if d > end:
            break
        if d >= start:
            out.append(d)
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out


def _quarter_label(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{q}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01", help="ISO date")
    ap.add_argument("--end", default=date.today().isoformat(), help="ISO date")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)
    quarters = _quarter_ends(start_d, end_d)

    # Find all entities that have supply_chain rows
    sc_dir = f"canonical/{Modality.SUPPLY_CHAIN.value}"
    entity_paths = [
        p for p in storage.list(sc_dir)
        if _ENTITY_FILE_RE.match(p.rsplit("/", 1)[-1])
    ]
    if not entity_paths:
        print("no supply_chain canonical files yet — run extract_supply_chain first")
        return

    print(f"building graph snapshots across {len(quarters)} quarters from "
          f"{len(entity_paths)} source entities…")
    total_edges = 0
    for q_end in quarters:
        # PIT query the supply_chain modality across ALL entities at this date.
        df = pit.query(
            Modality.SUPPLY_CHAIN,
            as_of=q_end,
            partition_keys=SupplyChainEdge.PARTITION_KEYS,
        )
        if df.empty:
            continue
        # Each row already names (entity_id=source, target_entity_id, relationship_type)
        # so we just emit the columns needed for graph operations.
        edges = df[[
            "entity_id", "target_entity_id", "relationship_type",
            "weight_qualifier", "confidence", "evidence_span",
            "accession", "form", "item_id", "availability_date",
        ]].rename(columns={"entity_id": "source_entity_id"}).copy()
        label = _quarter_label(q_end)
        out_path = f"graph/{label}/edges.parquet"
        storage.write_parquet(out_path, edges)
        print(f"  {label}: {len(edges):>5,} edges → {out_path}")
        total_edges += len(edges)

    print(f"\ntotal edge-quarters written: {total_edges:,}")


if __name__ == "__main__":
    main()
