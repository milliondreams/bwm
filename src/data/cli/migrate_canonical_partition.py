"""One-time migration: convert canonical/{modality}/data.parquet (single big
file) into canonical/{modality}/cik={cik10}.parquet (one file per entity).

Idempotent: skips entities that already have a per-CIK parquet at the target
path. Deletes the legacy single file after verification.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from data.pit.engine import _entity_path
from data.schemas.pit import Modality
from data.storage import get_storage

_PARTITION_RE = re.compile(r"^entity=[A-Za-z0-9._-]+\.parquet$")


def migrate_modality(modality: Modality) -> None:
    storage = get_storage()
    legacy_path = f"canonical/{modality.value}/data.parquet"
    if not storage.exists(legacy_path):
        print(f"[{modality.value}] no legacy file; skipping")
        return

    df = storage.read_parquet(legacy_path)
    n_before = len(df)
    entities = df["entity_id"].unique()
    print(f"[{modality.value}] legacy file has {n_before:,} rows across {len(entities)} entities")

    n_after = 0
    skipped = 0
    written = 0
    for entity_id in entities:
        target = _entity_path(modality, str(entity_id))
        batch = df[df["entity_id"] == entity_id].reset_index(drop=True)
        if storage.exists(target):
            # Merge legacy rows into the existing per-CIK file (idempotent).
            existing = storage.read_parquet(target)
            combined = pd.concat([existing, batch], ignore_index=True)
            # Use a conservative dedup: full row equality. Per-modality logical
            # dedup happens in PITEngine.write going forward; this is just to
            # avoid bloating the file with verbatim duplicates from re-runs.
            combined = combined.drop_duplicates().reset_index(drop=True)
            storage.write_parquet(target, combined)
            n_after += len(combined) - len(existing)
            skipped += 1
        else:
            storage.write_parquet(target, batch)
            n_after += len(batch)
            written += 1

    # Sanity: round-trip count check.
    total = 0
    for p in storage.list(f"canonical/{modality.value}"):
        fname = p.rsplit("/", 1)[-1]
        if _PARTITION_RE.match(fname):
            total += len(storage.read_parquet(p))
    print(f"[{modality.value}] wrote {written} new files, merged into {skipped} existing; total per-CIK rows = {total:,}")

    if total < n_before:
        print(f"[{modality.value}] !! row count regressed ({n_before:,} → {total:,}); not deleting legacy file")
        return

    # Delete the legacy single-file blob via the bytes API (no Storage.delete
    # exists yet; fall back to a Path operation when LocalStorage is in use).
    _delete_path(storage, legacy_path)
    print(f"[{modality.value}] deleted legacy {legacy_path}")


def _delete_path(storage, path: str) -> None:
    """Best-effort delete. LocalStorage holds a `root` attribute; AzureMLStorage
    is not exercised here because the migration is intended for the dev fixture.
    """
    root = getattr(storage, "root", None)
    if root is None:
        print(f"  (cannot delete {path} on this backend; leaving in place)")
        return
    p = Path(root) / path
    if p.exists():
        p.unlink()


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    for modality in (Modality.FINANCIALS,):  # only modality with legacy data today
        migrate_modality(modality)


if __name__ == "__main__":
    main()
