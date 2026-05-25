"""Bulk SEC Feed ingest: daily .nc.tar.gz → canonical/filings_full.lance.

Replaces the rate-limited per-filing EDGAR API path (which produced the
cap=80 broken corpus). Each day's tarball is ~500MB-2GB compressed and
contains every filing accepted by SEC that day. We mirror the full set
filtered to registry CIKs, dropping only XBRL family docs (covered by DERA).

Run:
    # Smoke: single day, 20-CIK pilot
    BWM_STORAGE_BACKEND=lance \\
        uv run python -m data.cli.ingest_feed --start 2024-01-25 --end 2024-01-25 --limit-ciks 20

    # Full-year backfill
    BWM_STORAGE_BACKEND=lance \\
        uv run python -m data.cli.ingest_feed --start 2024-01-01 --end 2024-12-31

    # Full 25-year historical backfill (split across nodes by year range)
    BWM_STORAGE_BACKEND=lance \\
        uv run python -m data.cli.ingest_feed --start 2001-01-01 --end 2025-12-31

Architecture:
    1. Walk date range day-by-day (skipping weekends + SEC holidays implicitly
       via 404 handling)
    2. For each day: download tarball → stream-extract filings filtered to
       registry CIKs → flatten to per-document rows → batch-write per CIK
    3. Update per-CIK watermark with latest filing date for incremental re-runs

Parallelism: single-process per date range; run multiple jobs on disjoint
ranges to scale. Each day is independent.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd

from data.entity.registry import EntityRegistry
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.edgar.feed_bulk import (
    FeedDay,
    download_day,
    filings_to_rows,
    iter_filings,
)
from data.sources.edgar.state import WatermarkStore
from data.storage import get_storage

WM_SOURCE = "sec_edgar:feed"
FILINGS_MODALITY = "filings_full"  # not in Modality enum; written to canonical/filings_full
BATCH_FLUSH_ROWS = 5_000           # flush per-CIK buffer at this size


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@dataclass
class DayStats:
    day: date
    download_bytes: int = 0
    filings: int = 0
    docs: int = 0
    ciks_touched: int = 0
    elapsed_s: float = 0.0
    status: str = "pending"
    error: str = ""


def _write_per_cik_batch(storage, rows_by_cik: dict[tuple[str, int], list[dict]]) -> int:
    """Flush buffered rows to per-CIK / per-year Lance datasets.

    The Lance write semantic is read-merge-write to dedup by sha256 (same
    document bytes from a corrected re-filing shouldn't create duplicate
    rows). For brand-new (cik, year) shards the dedup is a no-op.
    """
    n_written = 0
    for (cik, year), rows in rows_by_cik.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        path = f"canonical/filings_full/cik={cik}/year={year}"
        # Read-merge-dedup by sha256 (document content fingerprint)
        if storage.exists(path):
            try:
                existing = storage.read_table(path)
                combined = pd.concat([existing, df], ignore_index=True)
            except Exception:
                combined = df
        else:
            combined = df
        combined = combined.drop_duplicates(subset=["accession", "document_name", "sha256"], keep="last")
        storage.write_table(path, combined)
        n_written += len(df)
    return n_written


def _ingest_day(
    day: date,
    keep_ciks: set[str],
    storage,
    watermark: WatermarkStore,
) -> DayStats:
    stats = DayStats(day=day)
    t0 = time.monotonic()
    feed_day = FeedDay(day=day)
    try:
        tarball = download_day(feed_day)
    except FileNotFoundError:
        stats.status = "missing"
        stats.elapsed_s = time.monotonic() - t0
        return stats
    except Exception as e:
        stats.status = "fetch_failed"
        stats.error = str(e)[:200]
        stats.elapsed_s = time.monotonic() - t0
        return stats
    stats.download_bytes = len(tarball)

    rows_by_cik: dict[tuple[str, int], list[dict]] = defaultdict(list)
    latest_per_cik: dict[str, tuple[date, str]] = {}
    try:
        for filing in iter_filings(tarball, keep_ciks=keep_ciks):
            stats.filings += 1
            for row in filings_to_rows(iter([filing])):
                rows_by_cik[(row["cik"], row["year"])].append(row)
                stats.docs += 1
            # Track watermark per primary CIK
            if filing.filed_date:
                current = latest_per_cik.get(filing.primary_cik, (date.min, ""))
                if filing.filed_date > current[0]:
                    latest_per_cik[filing.primary_cik] = (filing.filed_date, filing.accession)
            # Flush periodically to keep memory bounded for high-activity days
            buffered = sum(len(v) for v in rows_by_cik.values())
            if buffered >= BATCH_FLUSH_ROWS:
                _write_per_cik_batch(storage, rows_by_cik)
                rows_by_cik = defaultdict(list)
    except Exception as e:
        stats.status = "parse_failed"
        stats.error = str(e)[:200]
        stats.elapsed_s = time.monotonic() - t0
        return stats

    # Final flush
    if rows_by_cik:
        _write_per_cik_batch(storage, rows_by_cik)

    stats.ciks_touched = len(latest_per_cik)

    # Update watermarks (best-effort)
    for cik, (filed, accn) in latest_per_cik.items():
        try:
            watermark.update(cik, WM_SOURCE, accn, filed)
        except Exception:
            pass

    stats.status = "ok_with_records" if stats.docs > 0 else "ok_no_records"
    stats.elapsed_s = time.monotonic() - t0
    return stats


def _iter_days(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=_parse_date, required=True, help="First day (YYYY-MM-DD)")
    ap.add_argument("--end", type=_parse_date, required=True, help="Last day inclusive (YYYY-MM-DD)")
    ap.add_argument(
        "--limit-ciks", type=int, default=0,
        help="If >0, restrict to first N CIKs (smoke / debug)",
    )
    args = ap.parse_args()
    if args.end < args.start:
        raise SystemExit("--end must be >= --start")

    storage = get_storage()
    registry = EntityRegistry(storage)
    reg_df = registry.load()
    ciks_all = sorted(set(reg_df["cik"].astype(str).str.zfill(10).tolist()))
    if args.limit_ciks > 0:
        ciks_all = ciks_all[: args.limit_ciks]
    keep_ciks = set(ciks_all)
    print(f"[setup] backend={type(storage).__name__} ciks={len(keep_ciks)}")
    watermark = WatermarkStore(storage)

    days = list(_iter_days(args.start, args.end))
    print(f"[setup] processing {len(days)} days: {days[0]} → {days[-1]}")

    t_start = time.monotonic()
    total_docs = 0
    total_bytes = 0
    for i, day in enumerate(days, 1):
        stats = _ingest_day(day, keep_ciks, storage, watermark)
        total_docs += stats.docs
        total_bytes += stats.download_bytes
        log_parse(
            storage, cik="feed", accession=day.isoformat(),
            modality=FILINGS_MODALITY, status=stats.status,
            n_records=stats.docs, error_message=stats.error,
        )
        pace = i / max(time.monotonic() - t_start, 1e-9) * 3600  # days/hour
        eta_h = (len(days) - i) / max(pace, 1e-9)
        print(
            f"[{i}/{len(days)}] {day} status={stats.status} "
            f"filings={stats.filings:>5} docs={stats.docs:>6} ciks={stats.ciks_touched:>4} "
            f"dl={stats.download_bytes/1e6:>7.1f}MB wall={stats.elapsed_s:>6.1f}s "
            f"pace={pace:.1f}d/h eta={eta_h:.2f}h"
            + (f" ERR={stats.error[:80]}" if stats.error else "")
        )

    print()
    print(
        f"[done] {len(days)} days in {time.monotonic()-t_start:.1f}s; "
        f"total docs={total_docs:,}; download={total_bytes/1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
