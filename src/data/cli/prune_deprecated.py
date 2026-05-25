"""Prune `canonical/{modality}_v1_deprecated_*` and `_undone_*` archives.

Keeps the most-recent `--keep-count` (default 3) regardless of age, plus
anything younger than `--age-days` (default 30). Deletes the rest.

LocalStorage / LanceStorage with a local root only. Cloud-backed storage
needs Azure Blob lifecycle rules (a separate, orthogonal workstream);
this CLI emits a clear warning and skips on those.

Run:
    BWM_DATA_ROOT=.data uv run python -m data.cli.prune_deprecated \\
        [--modality financials] [--age-days 30] [--keep-count 3] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from data.observability.log import emit_event
from data.storage import get_storage
from data.storage.backend import LocalStorage

_DEPRECATED_RE = re.compile(
    r"^(?P<modality>[a-z_]+)_v1_deprecated_(?P<date>\d{8})$"
)
_UNDONE_RE = re.compile(
    r"^(?P<modality>[a-z_]+)_undone_(?P<ts>\d{8}T\d{6}Z)$"
)


def _resolve_local_root(storage) -> Path:
    if not isinstance(storage, LocalStorage):
        if hasattr(storage, "root") and isinstance(storage.root, str):
            return Path(storage.root)
        raise SystemExit(
            "prune_deprecated requires a local-filesystem storage root. "
            "Cloud backends should use Azure Blob lifecycle policies."
        )
    return storage.root


def _parse_age_days(name: str, today: date) -> tuple[str, int] | None:
    """Return (modality, age_days) if `name` matches a deprecated/undone
    pattern; None otherwise.
    """
    m = _DEPRECATED_RE.match(name)
    if m:
        d = datetime.strptime(m.group("date"), "%Y%m%d").date()
        return m.group("modality"), (today - d).days
    m = _UNDONE_RE.match(name)
    if m:
        d = datetime.strptime(m.group("ts"), "%Y%m%dT%H%M%SZ").date()
        return m.group("modality"), (today - d).days
    return None


def _dir_size_bytes(p: Path) -> int:
    total = 0
    for child in p.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _classify(
    root: Path, modality_filter: str | None, today: date
) -> dict[str, list[tuple[Path, int]]]:
    """Group archives by modality. Each entry: (path, age_days), newest first."""
    canonical = root / "canonical"
    if not canonical.exists():
        return {}
    grouped: dict[str, list[tuple[Path, int]]] = {}
    for entry in canonical.iterdir():
        if not entry.is_dir():
            continue
        parsed = _parse_age_days(entry.name, today)
        if parsed is None:
            continue
        modality, age = parsed
        if modality_filter and modality != modality_filter:
            continue
        grouped.setdefault(modality, []).append((entry, age))
    for items in grouped.values():
        # Newest first (smallest age_days first)
        items.sort(key=lambda t: t[1])
    return grouped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modality", default="",
                    help="restrict to one modality (default: all)")
    ap.add_argument("--age-days", type=int, default=30,
                    help="anything younger than this is kept (default: 30)")
    ap.add_argument("--keep-count", type=int, default=3,
                    help="always keep the N newest per modality (default: 3)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    storage = get_storage()
    root = _resolve_local_root(storage)
    today = date.today()

    grouped = _classify(root, args.modality or None, today)
    if not grouped:
        print("[prune] nothing to prune")
        return 0

    log_path = root / "state" / "prune_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_pruned = 0
    total_freed = 0
    for modality, entries in sorted(grouped.items()):
        # Keep the newest --keep-count regardless of age.
        kept = entries[: args.keep_count]
        kept_names = {p.name for p, _ in kept}
        # Of the rest, keep anything younger than --age-days.
        candidates = []
        for p, age in entries[args.keep_count:]:
            if age < args.age_days:
                kept_names.add(p.name)
            else:
                candidates.append((p, age))

        if not candidates:
            print(f"[prune] {modality}: nothing eligible (kept {len(kept_names)})")
            continue

        freed = 0
        pruned_paths: list[str] = []
        for p, age in candidates:
            size = _dir_size_bytes(p)
            freed += size
            pruned_paths.append(p.name)
            if args.dry_run:
                print(f"[prune] {modality} DRY: would delete {p.name} "
                      f"(age={age}d, size={size:,}B)")
            else:
                shutil.rmtree(p, ignore_errors=True)
                print(f"[prune] {modality}: deleted {p.name} "
                      f"(age={age}d, freed={size:,}B)")
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "modality": modality,
                        "path": p.name,
                        "age_days": age,
                        "freed_bytes": size,
                    }) + "\n")

        total_pruned += len(pruned_paths)
        total_freed += freed
        emit_event(
            "prune_deprecated", modality, "prune",
            kept=len(kept_names), pruned=len(pruned_paths),
            freed_bytes=freed, dry_run=args.dry_run,
        )

    print(f"[prune] total: pruned={total_pruned}, freed={total_freed:,}B"
          + (" (DRY)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
