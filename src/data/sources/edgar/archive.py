"""Content-addressed raw archival for SEC EDGAR responses.

Layout under the storage root:

    raw/sec_edgar/{cik10}/{kind}/{sha12}.{ext}.gz

`{sha12}` is the first 12 hex chars of SHA-1(body). Identical content always
lands at the identical path, so `was_new` reliably reflects "we already have
this byte-for-byte." Re-fetches on a later day are no-ops, not new files.

### Forensic trail (NFR-1)

    state/edgar/archive_log/{cik10}.jsonl    one JSON line per archive call
    state/edgar/parse_log/{cik10}.jsonl      one JSON line per parse attempt

Both are append-only JSONL: each call writes a single line via the storage
backend's `append_bytes` (O(1) on local; falls back to read-modify-write on
cloud blob backends — see Storage.append_bytes).

The archive log answers "when did we observe this content?" The parse log
answers "what did we extract from it, or why did it fail?" Splitting them
lets us distinguish silent-zero-facts from real parse errors without coupling
the fetch path to per-modality parser semantics.

### Why JSONL not parquet

The prior implementation used parquet, doing read-modify-write on every
archive call. At 10K CIKs × hundreds of events each, that's O(n²) writes per
CIK. JSONL append is O(1) and the log becomes queryable by streaming OR by
the optional `cli/compact_logs.py` script that snapshots JSONL → parquet on
demand.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from data.storage import Storage


def _digest(body: bytes) -> str:
    return hashlib.sha1(body).hexdigest()[:12]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def archive_path(
    cik: str,
    kind: str,
    body: bytes,
    accession: Optional[str] = None,  # noqa: ARG001 — accepted for API stability
) -> str:
    """Build the canonical archive path for one document.

    Args:
        cik: 10-digit zero-padded CIK.
        kind: short doc tag — e.g. "submissions", "xbrl", "primary", "form4".
        body: raw bytes (used to compute the content hash).
        accession: accepted for API compatibility but no longer in the path —
            the SHA already uniquely identifies content. The accession is
            still written to the archive log for human-readable audit.
    """
    sha = _digest(body)
    cik10 = cik.zfill(10)
    ext = _extension_for(kind)
    return f"raw/sec_edgar/{cik10}/{kind}/{sha}.{ext}.gz"


def _extension_for(kind: str) -> str:
    return {
        "submissions": "json",
        "xbrl": "xml",
        "primary": "html",
        "form4": "xml",
        "companyfacts": "json",
    }.get(kind, "bin")


def archive_log_path(cik: str) -> str:
    return f"state/edgar/archive_log/{cik.zfill(10)}.jsonl"


def parse_log_path(cik: str) -> str:
    return f"state/edgar/parse_log/{cik.zfill(10)}.jsonl"


def _append_log(storage: Storage, log_path: str, payload: dict) -> None:
    """Write one JSON line to `log_path`, terminated with '\\n'."""
    line = (json.dumps(payload, default=str) + "\n").encode("utf-8")
    storage.append_bytes(log_path, line)


def archive(
    storage: Storage,
    cik: str,
    kind: str,
    body: bytes,
    accession: Optional[str] = None,
) -> tuple[str, bool]:
    """Persist a raw response. Returns (path, was_new).

    Always records an event in the archive log, even when the body is a
    byte-for-byte duplicate of one already on disk — the log captures *when*
    we observed the content, while the path captures *what* we observed.
    """
    sha = _digest(body)
    path = archive_path(cik, kind, body, accession=accession)
    if storage.exists(path):
        was_new = False
    else:
        storage.write_gz_bytes(path, body)
        was_new = True
    _append_log(
        storage,
        archive_log_path(cik),
        {
            "ingest_at_utc": _now_utc_iso(),
            "kind": kind,
            "accession": accession or "",
            "path": path,
            "sha12": sha,
            "was_new": was_new,
        },
    )
    return path, was_new


# ---- parse log (Stage 1.6 A5) ----------------------------------------------


# Statuses: ok_with_records (parsed N>0 records), ok_no_records (parsed
# successfully but zero records — distinguishes empty filings from failures),
# parse_failed (exception during parse), fetch_failed (network error).
PARSE_STATUSES = ("ok_with_records", "ok_no_records", "parse_failed", "fetch_failed")


def log_parse(
    storage: Storage,
    cik: str,
    accession: str,
    modality: str,
    status: str,
    n_records: int = 0,
    error_message: str = "",
) -> None:
    """Record one parse attempt. Called by every connector/extractor.

    `status` must be one of PARSE_STATUSES. The parser doesn't have to know
    about modality enum values — pass the string label so this is callable
    from any connector without circular schema imports.
    """
    if status not in PARSE_STATUSES:
        raise ValueError(f"unknown parse status: {status!r}; must be one of {PARSE_STATUSES}")
    _append_log(
        storage,
        parse_log_path(cik),
        {
            "ingest_at_utc": _now_utc_iso(),
            "accession": accession,
            "modality": modality,
            "status": status,
            "n_records": int(n_records),
            "error_message": error_message,
        },
    )
