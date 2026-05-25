"""Disk + memory pre-flight checks for ingest CLIs.

The Phase A2 v2 ingest path can OOM or fill disk under realistic loads:

  - `ingest_dera` parses `num.txt` (527 MB uncompressed for a recent quarter)
    into a pandas DataFrame — ~1-2 GB peak RAM per quarter.
  - `ingest_feed` buffers per-document rows in memory before flushing; peak
    earnings-season days are ~3 GB compressed / 8-12 GB resident.

Without pre-flight checks, an under-provisioned host fails with an opaque
OOM kill or a partial write that breaks downstream reads. With them, the
operator gets a single actionable error before the slow download begins.

Both functions raise `RuntimeError` on insufficiency. The CLIs expose a
`--skip-resource-check` flag for emergency overrides; missing capacity is
recoverable (run on a bigger node, free disk, etc.) so a hard fail is the
right default.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import psutil


def check_disk_space(path: str | Path, min_gb: float, hint: str = "") -> None:
    """Raise RuntimeError if free disk at `path` is below `min_gb` GiB.

    `path` doesn't have to exist; we resolve to its nearest existing parent
    so callers can check the spool dir before creating it.
    """
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    free_bytes = shutil.disk_usage(p).free
    free_gb = free_bytes / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(
            f"insufficient disk space at {p}: "
            f"{free_gb:.2f} GiB free, {min_gb:.2f} GiB required"
            + (f" ({hint})" if hint else "")
            + ". Free space or use --skip-resource-check to override."
        )


def check_memory(min_gb: float, hint: str = "") -> None:
    """Raise RuntimeError if available system RAM is below `min_gb` GiB.

    Uses `psutil.virtual_memory().available` which subtracts kernel caches
    and other reclaimable memory — the realistic "process can grab this much
    without triggering swap" number.
    """
    avail_bytes = psutil.virtual_memory().available
    avail_gb = avail_bytes / (1024**3)
    if avail_gb < min_gb:
        raise RuntimeError(
            f"insufficient memory: {avail_gb:.2f} GiB available, "
            f"{min_gb:.2f} GiB required"
            + (f" ({hint})" if hint else "")
            + ". Run on a larger node or use --skip-resource-check to override."
        )


def preflight(
    disk_path: str | Path,
    disk_min_gb: float,
    mem_min_gb: float,
    hint: str = "",
) -> None:
    """Combined disk + memory check. Either failure raises RuntimeError with
    a single actionable message.

    Call at CLI startup AND before each large unit (e.g. before downloading
    each DERA quarter or each Feed day) so mid-run capacity loss is caught
    before the next slow download.
    """
    check_disk_space(disk_path, disk_min_gb, hint=hint)
    check_memory(mem_min_gb, hint=hint)
