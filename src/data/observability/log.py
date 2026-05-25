"""Structured (JSON-line) job event log.

A unified per-event stream that complements the existing print-based
progress lines and per-CIK forensic logs (`state/edgar/parse_log/*.jsonl`).
Stored at `state/job_log.jsonl` so it sits one level above the EDGAR-
specific logs — this is a cross-pipeline log shared by DERA, Feed,
M-Batch, validators, and the swap CLI.

### When to emit

- `emit_start(pipeline, modality)` at CLI entry — returns an event_id that
  callers thread through subsequent events as `parent_event_id`.
- `emit_event(pipeline, modality, status, **fields)` at each unit boundary
  (per quarter for DERA, per day for Feed, etc.).
- `emit_end(pipeline, modality, status, event_id, wall_seconds)` at exit.

### Schema (every event)

    {
      "ts": "2026-05-25T02:13:12.345Z",        # UTC, microsecond
      "event_id": "uuid4",                      # this event's correlator
      "parent_event_id": "uuid4" | null,        # ties end→start
      "pipeline": "ingest_dera",
      "modality": "financials",
      "status": "start" | "progress" | "ok_with_records" | "ok_no_records"
              | "parse_failed" | "fetch_failed" | "swap" | "validation" | ...,
      "n_records": 13520,                       # nullable
      "wall_seconds": 21.9,                     # nullable; on end events
      ... arbitrary extra fields ...
    }

### Storage

Always appends to `state/job_log.jsonl` via `Storage.append_bytes` so this
works on both local filesystem and AzureML mounted blobs (cloud append
falls back to read-modify-write, which is fine for the low-frequency event
rate we expect — at most ~hundreds of events per day per job).

Stdout mirroring is gated by `BWM_STRUCTURED_LOG=1` to avoid noisy double-
printing during dev. Operators flip it on for AML runs where machine-
consumable streams matter.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

# Lazy storage import to avoid circular deps if observability is imported
# from inside the storage layer itself.
_LOG_PATH = "state/job_log.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _get_storage():
    """Defer the import so module-import time stays light."""
    from data.storage import get_storage
    return get_storage()


def _structured_stdout_enabled() -> bool:
    return os.environ.get("BWM_STRUCTURED_LOG", "").lower() in ("1", "true", "yes")


def emit_event(
    pipeline: str,
    modality: str,
    status: str,
    *,
    event_id: str | None = None,
    parent_event_id: str | None = None,
    **fields: Any,
) -> str:
    """Append one structured event to the unified job log; return its event_id.

    Storage failures are swallowed (with a stderr note) — observability must
    never break the actual pipeline. If you depend on events landing, run
    against a writable storage root.
    """
    eid = event_id or str(uuid.uuid4())
    event: dict[str, Any] = {
        "ts": _utc_now_iso(),
        "event_id": eid,
        "parent_event_id": parent_event_id,
        "pipeline": pipeline,
        "modality": modality,
        "status": status,
    }
    # Add caller-supplied fields, deduping against the reserved keys above.
    for k, v in fields.items():
        if k not in event:
            event[k] = v

    line = json.dumps(event, default=str, separators=(",", ":")) + "\n"
    try:
        _get_storage().append_bytes(_LOG_PATH, line.encode("utf-8"))
    except Exception as e:
        # Don't crash the pipeline because the log write failed; surface a
        # one-time note on stderr instead.
        import sys
        print(f"[log] warning: append to {_LOG_PATH} failed: {e}", file=sys.stderr)

    if _structured_stdout_enabled():
        # Print the same line to stdout so AML log tailing surfaces events
        # without needing to mount the storage root.
        print(line, end="")

    return eid


def emit_start(pipeline: str, modality: str, **fields: Any) -> str:
    """Emit a `start` event. Return event_id for use as parent_event_id
    on subsequent progress + end events.
    """
    return emit_event(pipeline, modality, "start", **fields)


def emit_end(
    pipeline: str,
    modality: str,
    status: str,
    event_id: str | None,
    wall_seconds: float,
    **fields: Any,
) -> None:
    """Emit a paired `end` event linked to a prior `start` via parent_event_id.

    `event_id` is the START event's id (becomes parent_event_id here); we
    mint a fresh uuid for this end event so the two are correlated but
    distinct.
    """
    emit_event(
        pipeline, modality, status,
        parent_event_id=event_id, wall_seconds=wall_seconds, **fields,
    )
