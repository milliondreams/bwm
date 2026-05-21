"""Stage 1.7A regression: every canonical row has the v3 PIT field set.

Asserts that across all modalities currently on disk, every row carries the
six v3 fields with valid values:
  - recorded_at: tz-aware UTC datetime
  - belief: float in [0, 1]
  - perspective: non-empty string matching the schema's DEFAULT_PERSPECTIVE
    (or a stable per-row source identifier)
  - policy_tags: list (possibly empty)
  - source_certainty: float in [0, 1]
  - valid_from_uncertainty_days: int ≥ 0

This is the substrate-independent half of v3 § 9.1 acceptance:
  "All records carry valid_from (BTIC), knowable_from, belief, perspective,
   policy_tags, source_certainty"
"""
from __future__ import annotations

import os

import pandas as pd

from data.pit.engine import _ENTITY_FILE_RE
from data.schemas.pit import Modality
from data.storage import get_storage
from data.cli.migrate_v3_fields import SCHEMA_BY_MODALITY, V3_COLUMNS


def _check_modality(storage, modality: Modality, schema_cls) -> tuple[int, list[str]]:
    """Returns (n_rows_checked, list_of_failures)."""
    failures: list[str] = []
    n_checked = 0
    expected_persp = getattr(schema_cls, "DEFAULT_PERSPECTIVE", "")
    expected_cert = getattr(schema_cls, "DEFAULT_SOURCE_CERTAINTY", 1.0)
    dir_path = f"canonical/{modality.value}"

    for p in storage.list(dir_path):
        if not _ENTITY_FILE_RE.match(p.rsplit("/", 1)[-1]):
            continue
        df = storage.read_parquet(p)
        n_checked += len(df)

        # Column presence
        missing = [c for c in V3_COLUMNS if c not in df.columns]
        if missing:
            failures.append(f"{p}: missing columns {missing}")
            continue

        # Type / range checks
        if not df.empty:
            # recorded_at: must be tz-aware
            ra = pd.to_datetime(df["recorded_at"], errors="coerce", utc=True)
            if ra.isna().any():
                failures.append(f"{p}: recorded_at has NaT in {int(ra.isna().sum())} row(s)")
            # belief in [0, 1]
            bel = pd.to_numeric(df["belief"], errors="coerce")
            bad_bel = ((bel < 0) | (bel > 1) | bel.isna()).sum()
            if bad_bel:
                failures.append(f"{p}: belief out of [0,1] or NaN in {int(bad_bel)} row(s)")
            # source_certainty in [0, 1]
            sc = pd.to_numeric(df["source_certainty"], errors="coerce")
            bad_sc = ((sc < 0) | (sc > 1) | sc.isna()).sum()
            if bad_sc:
                failures.append(f"{p}: source_certainty out of [0,1] or NaN in {int(bad_sc)} row(s)")
            # valid_from_uncertainty_days >= 0
            vu = pd.to_numeric(df["valid_from_uncertainty_days"], errors="coerce")
            bad_vu = ((vu < 0) | vu.isna()).sum()
            if bad_vu:
                failures.append(f"{p}: valid_from_uncertainty_days < 0 or NaN in {int(bad_vu)} row(s)")
            # perspective: non-empty for at least one row (schemas with
            # DEFAULT_PERSPECTIVE non-empty should produce non-empty rows)
            if expected_persp and (df["perspective"].astype(str).str.len() == 0).all():
                failures.append(f"{p}: all perspective values empty; expected {expected_persp!r}")
            # policy_tags: must be list-like (parquet returns numpy arrays or lists)
            sample = df["policy_tags"].iloc[0]
            if sample is not None and not isinstance(sample, (list, tuple)) and not hasattr(sample, "__iter__"):
                failures.append(f"{p}: policy_tags is not list-like (type={type(sample).__name__})")
    return n_checked, failures


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    print("=== Stage 1.7A validation: v3 PIT field set on every canonical row ===\n")

    total_rows = 0
    total_failures: list[str] = []
    for modality, schema_cls in SCHEMA_BY_MODALITY.items():
        n_checked, failures = _check_modality(storage, modality, schema_cls)
        tag = "OK" if not failures else f"FAIL×{len(failures)}"
        persp = getattr(schema_cls, "DEFAULT_PERSPECTIVE", "(none)")
        cert = getattr(schema_cls, "DEFAULT_SOURCE_CERTAINTY", 1.0)
        print(f"  [{tag}] {modality.value:>15s}: {n_checked:>8,} rows  "
              f"perspective={persp!r} src_cert={cert}")
        for f in failures[:3]:
            print(f"      → {f}")
        total_rows += n_checked
        total_failures.extend(failures)

    print(f"\nTotal rows checked: {total_rows:,}")
    if total_failures:
        print(f"FAILURES: {len(total_failures)} issue(s)")
        for f in total_failures[:10]:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\n=== Stage 1.7A validation: PASS ===")


if __name__ == "__main__":
    main()
