"""Stage 1.7A migration: back-fill v3 PIT field columns onto existing parquet.

Walks every `canonical/{modality}/entity={eid}.parquet`, adds missing v3
columns with per-modality defaults from the schema's `pit_default_fields()`,
writes back. Idempotent — already-migrated files are detected by all-columns-
present check and skipped.

Run:
    uv run python -m data.cli.migrate_v3_fields
    uv run python -m data.cli.migrate_v3_fields --modality financials
    uv run python -m data.cli.migrate_v3_fields --dry-run
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from data.pit.engine import _ENTITY_FILE_RE, _backfill_v3_fields
from data.schemas.earnings_call import EarningsCallExcerpt
from data.schemas.filing_text import FilingTextChunk
from data.schemas.financial import FinancialFact
from data.schemas.hiring_observation import HiringObservation
from data.schemas.insider_trade import InsiderTrade
from data.schemas.macro_observation import MacroObservation
from data.schemas.market_bar import MarketBar
from data.schemas.news_event import NewsEvent
from data.schemas.patent import Patent
from data.schemas.pit import Modality
from data.schemas.supply_chain import SupplyChainEdge
from data.storage import get_storage

# One mapping for all 10 modalities. Adding the next modality is a single
# row here — no other migration code changes.
SCHEMA_BY_MODALITY = {
    Modality.FINANCIALS: FinancialFact,
    Modality.INSIDER_TRADES: InsiderTrade,
    Modality.FILINGS_TEXT: FilingTextChunk,
    Modality.SUPPLY_CHAIN: SupplyChainEdge,
    Modality.MARKET: MarketBar,
    Modality.MACRO: MacroObservation,
    Modality.NEWS: NewsEvent,
    Modality.PATENTS: Patent,
    Modality.EARNINGS_CALLS: EarningsCallExcerpt,
    Modality.HIRING: HiringObservation,
}

V3_COLUMNS = (
    "recorded_at",
    "belief",
    "perspective",
    "policy_tags",
    "source_certainty",
    "valid_from_uncertainty_days",
)


def migrate_modality_file(storage, path: str, schema_cls, dry_run: bool) -> tuple[int, int]:
    """Back-fill v3 columns in one canonical parquet. Returns (n_rows, n_added_cols)."""
    df = storage.read_parquet(path)
    missing = [c for c in V3_COLUMNS if c not in df.columns]
    if not missing:
        return len(df), 0
    if dry_run:
        return len(df), len(missing)
    defaults = schema_cls.pit_default_fields()
    df = _backfill_v3_fields(df, defaults)
    storage.write_parquet(path, df)
    return len(df), len(missing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="", help="single modality (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()

    if args.modality:
        try:
            modalities = [Modality(args.modality)]
        except ValueError as e:
            raise SystemExit(f"unknown modality: {args.modality}") from e
    else:
        modalities = list(SCHEMA_BY_MODALITY.keys())

    total_files = 0
    total_rows = 0
    total_files_changed = 0
    print(f"v3 fields migration{' (DRY RUN)' if args.dry_run else ''} — modalities: {[m.value for m in modalities]}")
    for m in modalities:
        schema_cls = SCHEMA_BY_MODALITY[m]
        dir_path = f"canonical/{m.value}"
        files = [p for p in storage.list(dir_path) if _ENTITY_FILE_RE.match(p.rsplit("/", 1)[-1])]
        if not files:
            print(f"  [{m.value:>15s}] no parquet files yet")
            continue
        modality_rows = 0
        modality_changed = 0
        for p in files:
            n_rows, n_added = migrate_modality_file(storage, p, schema_cls, args.dry_run)
            modality_rows += n_rows
            if n_added > 0:
                modality_changed += 1
        print(
            f"  [{m.value:>15s}] {len(files)} file(s), {modality_rows:>8,} rows; "
            f"{modality_changed} needed back-fill"
        )
        total_files += len(files)
        total_rows += modality_rows
        total_files_changed += modality_changed

    print(
        f"\nsummary: {total_files} file(s) inspected, {total_rows:,} rows; "
        f"{total_files_changed} file(s) {'would be' if args.dry_run else 'were'} migrated"
    )


if __name__ == "__main__":
    main()
