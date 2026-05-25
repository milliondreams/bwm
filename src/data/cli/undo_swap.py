"""Reverse the most-recent `swap_canonical` for a given modality.

Reads `state/swap_log.jsonl`, finds the most-recent `swap` event for the
target modality that does not have a matching `undo` event after it, and
reverses the rename:

  - canonical/{modality}            → canonical/{modality}_undone_{ts}
  - canonical/{archived_to}         → canonical/{modality}

Refuses to act if the original swap had no prior canonical (archived_to
is null) — undoing would leave canonical empty with nothing to restore.

LocalStorage / LanceStorage with a local root only. Cloud-backed
storage requires Azure SDK rename + manifest update; emits a clear error.

Run:
    BWM_DATA_ROOT=.data uv run python -m data.cli.undo_swap \\
        --modality financials [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from data.observability.log import emit_event
from data.storage import get_storage
from data.storage.backend import LocalStorage


def _resolve_local_root(storage) -> Path:
    if not isinstance(storage, LocalStorage):
        if hasattr(storage, "root") and isinstance(storage.root, str):
            return Path(storage.root)
        raise SystemExit(
            "undo_swap requires a local-filesystem storage root. "
            "Cloud backends need Azure SDK rename; out of scope here."
        )
    return storage.root


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    events: list[dict] = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _find_pending_swap(events: list[dict], modality: str) -> dict | None:
    """Walk newest-first; return the latest `swap` for `modality` that
    isn't already followed by an `undo` for the same modality.

    The swap event today doesn't carry an explicit `event` field — it has
    `modality`, `promoted_from`, `archived_to`, `ts`. Undo events we write
    here carry `event: "undo"`. We treat any non-`undo` entry for the
    modality as a swap.
    """
    # Track undone swap timestamps so we don't reverse a swap twice.
    undone_ts: set[str] = set()
    # Walk newest → oldest
    for ev in reversed(events):
        if ev.get("modality") != modality:
            continue
        if ev.get("event") == "undo":
            undone_ts.add(ev.get("undone_swap_ts", ""))
            continue
        if ev.get("ts") in undone_ts:
            continue
        return ev
    return None


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modality", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    storage = get_storage()
    root = _resolve_local_root(storage)
    log_path = root / "state" / "swap_log.jsonl"

    events = _read_log(log_path)
    swap = _find_pending_swap(events, args.modality)
    if swap is None:
        print(f"[undo_swap] no pending swap to undo for modality={args.modality}")
        return 1

    archived = swap.get("archived_to")
    if not archived:
        print(
            f"[undo_swap] swap at ts={swap.get('ts')} had no prior canonical "
            "(archived_to is null) — undoing would leave canonical empty. "
            "Refusing."
        )
        return 2

    canonical = root / "canonical" / args.modality
    archived_path = root / "canonical" / archived
    if not canonical.exists():
        print(f"[undo_swap] current canonical missing: {canonical}")
        return 2
    if not archived_path.exists():
        print(f"[undo_swap] archived path missing: {archived_path}")
        return 2

    undone_ts = _utc_now_compact()
    set_aside = root / "canonical" / f"{args.modality}_undone_{undone_ts}"

    if args.dry_run:
        print(f"[undo_swap] DRY RUN:")
        print(f"  would rename {canonical} → {set_aside}")
        print(f"  would rename {archived_path} → {canonical}")
        return 0

    print(f"[undo_swap] setting aside current canonical → {set_aside.name}")
    canonical.rename(set_aside)
    print(f"[undo_swap] restoring {archived_path.name} → {canonical.name}")
    archived_path.rename(canonical)

    # Record the undo
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "undo",
            "modality": args.modality,
            "undone_swap_ts": swap.get("ts"),
            "restored_from": archived,
            "set_aside_as": set_aside.name,
        }) + "\n")
    emit_event(
        "undo_swap", args.modality, "undo",
        undone_swap_ts=swap.get("ts"),
        restored_from=archived, set_aside_as=set_aside.name,
    )
    print(f"[undo_swap] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
