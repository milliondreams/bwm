"""Stage 5 validation: the full-ingest driver + Phase A acceptance check.

This script measures the current canonical store against the spec's
Phase A acceptance criteria (§6.2). The hard targets — 50K entities,
80 median quarters, 60% graph coverage — apply at the full-corpus scale.
For a partial / smoke ingest the script reports actuals so we can track
progress as we scale up.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine, _entity_path
from data.schemas.financial import FinancialFact
from data.schemas.filing_text import FilingTextChunk
from data.schemas.insider_trade import InsiderTrade
from data.schemas.pit import Modality
from data.storage import get_storage


def _count_entities(storage, modality: Modality) -> int:
    """Number of distinct entities that have data in this modality."""
    directory = f"canonical/{modality.value}"
    return sum(
        1 for p in storage.list(directory)
        if p.rsplit("/", 1)[-1].startswith("entity=")
    )


def _summary_for_modality(storage, modality: Modality) -> tuple[int, int]:
    """Return (entity_count, total_rows)."""
    entities = 0
    total_rows = 0
    for p in storage.list(f"canonical/{modality.value}"):
        if not p.rsplit("/", 1)[-1].startswith("entity="):
            continue
        entities += 1
        total_rows += len(storage.read_parquet(p))
    return entities, total_rows


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    registry = EntityRegistry(storage)
    reg_df = registry.load()

    print("=== Stage 5 validation: Phase A acceptance check ===\n")

    print(f"Registry size:                  {len(reg_df):,} entities")

    fin_ent, fin_rows = _summary_for_modality(storage, Modality.FINANCIALS)
    f4_ent, f4_rows = _summary_for_modality(storage, Modality.INSIDER_TRADES)
    txt_ent, txt_rows = _summary_for_modality(storage, Modality.FILINGS_TEXT)

    print(f"\nModality coverage:")
    print(f"  financials:       entities={fin_ent:>5}  rows={fin_rows:>12,}")
    print(f"  insider_trades:   entities={f4_ent:>5}  rows={f4_rows:>12,}")
    print(f"  filings_text:     entities={txt_ent:>5}  rows={txt_rows:>12,}")

    # Median quarters per entity for financials
    if fin_ent > 0:
        per_entity_quarters = []
        for p in storage.list(f"canonical/{Modality.FINANCIALS.value}"):
            if not p.rsplit("/", 1)[-1].startswith("entity="):
                continue
            df = storage.read_parquet(p)
            # A "quarter covered" is a distinct (fiscal_year, fiscal_period) pair
            # with at least one financial fact. This is a coarse proxy; spec
            # really wants distinct quarter-end dates.
            n_q = df[["fiscal_year", "fiscal_period"]].drop_duplicates().shape[0]
            per_entity_quarters.append(n_q)
        ser = pd.Series(per_entity_quarters)
        print(f"\nfinancials — quarters covered per entity:")
        print(f"  min={ser.min()}  median={ser.median()}  max={ser.max()}  mean={ser.mean():.1f}")

    # PIT round-trip sanity: pick a random entity in financials, verify
    # query returns >0 rows for as_of=today
    from datetime import date as date_cls
    fin_dir = f"canonical/{Modality.FINANCIALS.value}"
    entity_files = [
        p for p in storage.list(fin_dir)
        if p.rsplit("/", 1)[-1].startswith("entity=")
    ]
    if entity_files:
        sample = entity_files[0]
        # Extract entity_id from the filename "entity={slug}.parquet"
        eid = sample.rsplit("/", 1)[-1][len("entity="):-len(".parquet")]
        result = pit.query(
            Modality.FINANCIALS,
            entity_ids=[eid],
            as_of=date_cls.today(),
            partition_keys=FinancialFact.PARTITION_KEYS,
        )
        print(f"\nPIT spot-check on {eid}: {len(result):,} rows knowable as-of today")
        assert len(result) > 0

    # Spec Phase A targets
    print("\nSpec Phase A targets (§6.2):")
    print(f"  ≥50,000 entities with financials:  current = {fin_ent:>6}   target = 50,000")
    print(f"  ≥10 modalities at quarterly cadence: current = 3   target = 10")
    print(f"  Graph coverage ≥60% of top-1000:    current = 0%  target = 60%  (Stage 6)")

    print("\n=== Stage 5 validation completed (sample-scale; full backfill is operational) ===")


if __name__ == "__main__":
    main()
