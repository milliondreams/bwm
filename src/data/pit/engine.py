"""Point-in-time engine.

Given a per-CIK partitioned set of canonical PIT parquet tables, answers "what
was knowable about X on date D" — the contract from spec §5.6 and FR-2. The
non-trivial part is restatement handling: when multiple rows share the same
(entity, modality, effective_date, *modality_partition_keys*), the engine picks
the *most recently filed* row whose availability_date ≤ as_of (and that has
not itself been restated before as_of).

This is the single most operationally important data engineering primitive
in the project; getting it wrong silently invalidates every backtest.

### Storage layout

    canonical/{modality}/cik={cik10}.parquet     one file per entity

Writes touch only the affected entity's file; reads (when entity-scoped) touch
only the requested entities' files. Cross-entity queries glob the directory.

### Restatement-aware as-of selection

For each logical fact (uniquely identified by entity_id, effective_date, and
the modality's PARTITION_KEYS), the engine returns at most one row: the
last-filed surviving row. Ties on `availability_date` are broken by
`source_ref` (always present on PITRecord) to guarantee determinism.

### Schema-declared partition keys

Concrete modality schemas (FinancialFact, FilingTextChunk, …) declare a
PARTITION_KEYS list. The connector passes it to write/query so the engine
does not need to know modality-specific column names. If no list is provided,
the engine falls back to inferring from a small allowlist — used only by the
legacy companyfacts path.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Iterable, Optional

import pandas as pd

from data.schemas.pit import Modality
from data.storage import Storage


REQUIRED_COLS = {
    "entity_id",
    "modality",
    "effective_date",
    "availability_date",
    "restated_at",
    "source",
    "source_ref",
}

# Used only when the caller does not pass `partition_keys` (legacy path).
_LEGACY_INFERRED_KEYS = (
    "concept", "taxonomy", "fiscal_period", "fiscal_year", "unit",
    "context_id", "dimensions_json",
)

# File naming is content-agnostic: the entity_id slug becomes the partition
# value verbatim, with characters that are awkward on filesystems mapped to "-".
# Any entity_id matching this regex is recognized by the directory scanner.
_ENTITY_FILE_RE = re.compile(r"^entity=(?P<eid>[A-Za-z0-9._-]+)\.parquet$")


def _backfill_v3_fields(rows: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    """Add v3 PIT field columns that are missing from `rows`, using `defaults`.

    Scalar defaults broadcast across all rows; list defaults (e.g. policy_tags)
    are deep-copied per row so callers' mutations don't leak across records.
    """
    if not defaults:
        return rows
    n = len(rows)
    out = rows
    for col, default in defaults.items():
        if col in out.columns:
            continue
        if isinstance(default, list):
            out = out.assign(**{col: [list(default) for _ in range(n)]})
        else:
            out = out.assign(**{col: default})
    return out


def _safe_slug(entity_id: str) -> str:
    """Map an entity_id to a filesystem-safe slug.

    EntityRegistry currently produces ids like 'us-cik-0000320193'. The
    characters [A-Za-z0-9.-_] are all preserved as-is; any others would be
    replaced with '-' so the slug stays unambiguous and stable.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", entity_id)


def _entity_path(modality: Modality, entity_id: str) -> str:
    return f"canonical/{modality.value}/entity={_safe_slug(entity_id)}.parquet"


def _modality_dir(modality: Modality) -> str:
    return f"canonical/{modality.value}"


class PITEngine:
    """Restatement-aware as-of queries against per-CIK partitioned canonical tables."""

    def __init__(self, storage: Storage):
        self.storage = storage

    # ----- snapshots (PIT-safe entry point for training code) -------------

    def snapshot(
        self,
        entity_id: str,
        as_of: date,
        modalities: Optional[Iterable[Modality]] = None,
    ):
        """Return a frozen, PIT-bounded EntitySnapshot for `entity_id` at `as_of`.

        See data.pit.snapshot.EntitySnapshot for the invariants enforced.
        This is the supported entry point for training data loaders — going
        through `query()` and joining DataFrames manually skips the
        invariant assertions and is unsafe.
        """
        # Lazy import to avoid circular import (snapshot imports schema modules
        # which are independent of engine.py, but engine.py doesn't need them).
        from data.pit.snapshot import build_snapshot_from_pit

        return build_snapshot_from_pit(self, entity_id, as_of, modalities)

    def snapshots(
        self,
        entity_ids: Iterable[str],
        as_of: date,
        modalities: Optional[Iterable[Modality]] = None,
    ) -> dict:
        """Multi-entity snapshot at a single as_of. Returns {entity_id: snapshot}.

        Useful for graph-aware models that need neighbor states at the same
        point in time as the focal entity.
        """
        return {eid: self.snapshot(eid, as_of, modalities) for eid in entity_ids}

    # ----- write ----------------------------------------------------------

    def write(
        self,
        modality: Modality,
        rows: pd.DataFrame,
        partition_keys: Optional[list[str]] = None,
        schema_cls: Optional[type] = None,
    ) -> None:
        """Persist canonical rows for a modality, partitioned per entity.

        Each entity_id present in `rows` triggers exactly one read+rewrite of
        its own parquet file. Entities not in this batch are untouched.

        `partition_keys` is the modality's logical-fact key beyond
        (entity_id, effective_date). If None, the legacy column-inference
        path is used (preserves `companyfacts_legacy.py` callers).

        `schema_cls` is the concrete `PITRecord` subclass for this modality.
        When provided, any v3 spec fields missing from `rows.columns` are
        back-filled with per-schema defaults via `cls.pit_default_fields()`.
        Pre-Stage-1.7 callers that pass row dicts without v3 fields keep
        working — they just inherit the schema's source defaults.
        """
        missing = REQUIRED_COLS - set(rows.columns)
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        if rows.empty:
            return

        # Resolve schema_cls from modality if the caller didn't supply one.
        # This lets every write benefit from the v3 field back-fill without
        # each ingest CLI needing to import the schema explicitly. Imported
        # lazily to keep engine.py free of schema dependencies.
        if schema_cls is None:
            from data.pit.snapshot import lazy_schema_cls
            schema_cls = lazy_schema_cls(modality)
        if schema_cls is not None and hasattr(schema_cls, "pit_default_fields"):
            rows = _backfill_v3_fields(rows, schema_cls.pit_default_fields())

        keys = self._effective_partition_keys(partition_keys, rows.columns)
        dedup_keys = ["entity_id", "effective_date", "source_ref", *keys]

        for entity_id, batch in rows.groupby("entity_id", sort=False):
            path = _entity_path(modality, str(entity_id))
            if self.storage.exists(path):
                existing = self.storage.read_parquet(path)
                combined = pd.concat([existing, batch], ignore_index=True)
            else:
                combined = batch.reset_index(drop=True)

            # Keep last occurrence: on re-ingest we trust the newest row.
            combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
            self.storage.write_parquet(path, combined)

    # ----- query ----------------------------------------------------------

    def query(
        self,
        modality: Modality,
        entity_ids: Optional[Iterable[str]] = None,
        as_of: Optional[date] = None,
        effective_between: Optional[tuple[date, date]] = None,
        extra_filters: Optional[dict] = None,
        partition_keys: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return the as-of view of a modality.

        For each (entity_id, effective_date, *partition_keys*) group, returns
        the single row that was knowable on `as_of` — the most-recently-filed
        surviving row. Ties on availability_date broken by source_ref.
        """
        df = self._load(modality, entity_ids)
        if df.empty:
            return df.reset_index(drop=True)

        # Normalize date-like columns to Timestamp dtype so comparisons with
        # the user's pure-Python `date` arguments are type-safe. The result
        # frame is post-processed back to `date` so callers see the same
        # types they always did.
        for col in ("effective_date", "availability_date", "restated_at"):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        if effective_between is not None:
            lo, hi = (pd.Timestamp(d) for d in effective_between)
            df = df[(df["effective_date"] >= lo) & (df["effective_date"] <= hi)]
        if extra_filters:
            for k, v in extra_filters.items():
                df = df[df[k] == v]

        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            df = df[df["availability_date"] <= as_of_ts]
            df = df[df["restated_at"].isna() | (df["restated_at"] > as_of_ts)]

        keys = self._effective_partition_keys(partition_keys, df.columns)
        partition_keys_full = ["entity_id", "effective_date", *keys]

        # Stable sort: latest filing wins; ties broken by source_ref so that
        # two callers see the same answer even when filings collide on the
        # same day (rare but observed in cross-quarter amendments).
        df = df.sort_values(
            ["availability_date", "source_ref"], kind="mergesort"
        ).drop_duplicates(subset=partition_keys_full, keep="last")

        # Restore date columns to native `date` so callers' downstream code
        # that compares against pure-Python dates keeps working.
        for col in ("effective_date", "availability_date", "restated_at"):
            df[col] = df[col].dt.date
        return df.reset_index(drop=True)

    # ----- internals ------------------------------------------------------

    def _effective_partition_keys(
        self, partition_keys: Optional[list[str]], columns: Iterable[str]
    ) -> list[str]:
        if partition_keys is not None:
            present = [k for k in partition_keys if k in columns]
            return present
        # Legacy: infer from a known allowlist of column names.
        return [k for k in _LEGACY_INFERRED_KEYS if k in columns]

    def _load(
        self, modality: Modality, entity_ids: Optional[Iterable[str]]
    ) -> pd.DataFrame:
        if entity_ids is not None:
            frames = []
            for eid in entity_ids:
                p = _entity_path(modality, str(eid))
                if self.storage.exists(p):
                    frames.append(self.storage.read_parquet(p))
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

        # No filter — glob the modality directory.
        directory = _modality_dir(modality)
        paths = [
            p for p in self.storage.list(directory)
            if _ENTITY_FILE_RE.match(p.rsplit("/", 1)[-1])
        ]
        if not paths:
            return pd.DataFrame()
        return pd.concat(
            (self.storage.read_parquet(p) for p in paths), ignore_index=True
        )
