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
from data.observability.log import emit_event, emit_start, emit_end
from data.observability.resources import preflight
from data.observability.run_id import tag as _tag
from data.pit.engine import PITEngine
from data.schemas.insider_trade import InsiderTrade
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.edgar.feed_bulk import (
    FORM4_FORMS,
    FeedDay,
    FeedFiling,
    download_day,
    extract_form4,
    filings_to_rows,
    iter_filings,
)
from data.sources.edgar.form4_supersede import supersede_prior_form4
from data.sources.edgar.state import WatermarkStore
from data.storage import get_storage

WM_SOURCE = "sec_edgar:feed"
PIPELINE = "ingest_feed"
FILINGS_MODALITY = "filings_full"  # not in Modality enum; written to canonical/filings_full
INSIDER_MODALITY = Modality.INSIDER_TRADES.value
BATCH_FLUSH_ROWS = 5_000           # flush per-CIK buffer at this size
INSIDER_FLUSH_ROWS = 2_000         # smaller buffer: each row is a small trade record
# Peak earnings-season Feed days are ~3 GB compressed / ~8 GB resident as
# all-filings-of-day is buffered. 10 GB disk / 4 GB RAM is safe.
DISK_MIN_GB = 10.0
MEM_MIN_GB = 4.0


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@dataclass
class DayStats:
    day: date
    download_bytes: int = 0
    filings: int = 0
    docs: int = 0
    ciks_touched: int = 0
    form4_filings: int = 0           # # of Form 4 / 4/A filings seen
    form4_trades: int = 0            # # of insider transaction rows written
    form4_supersedes: int = 0        # # of canonical rows marked restated by 4/A
    elapsed_s: float = 0.0
    status: str = "pending"
    error: str = ""


def _form4_to_insider_rows(filing: FeedFiling, parsed) -> list[dict]:
    """Convert a parsed Form 4 (+ FeedFiling envelope) to insider_trade row dicts.

    Schema mirrors `data.schemas.insider_trade.InsiderTrade`. The PIT required
    columns (entity_id, modality, effective_date, availability_date,
    restated_at, source, source_ref) are populated here so PITEngine.write
    accepts the batch without further back-fill on those columns. Other v3
    PIT fields are filled by PITEngine via `InsiderTrade.pit_default_fields()`.
    """
    entity_id = f"us-cik-{filing.primary_cik}"
    rows: list[dict] = []
    for idx, t in enumerate(parsed.trades):
        rows.append({
            "transaction_index": idx,
            "period_of_report": parsed.period_of_report,
            "entity_id": entity_id,
            "modality": INSIDER_MODALITY,
            "effective_date": t.transaction_date,
            "availability_date": filing.filed_date,
            "restated_at": None,
            "source": "sec_edgar:form4",
            "source_ref": filing.accession,
            "insider_cik": t.insider_cik,
            "insider_name": t.insider_name,
            "is_director": t.is_director,
            "is_officer": t.is_officer,
            "is_ten_percent_owner": t.is_ten_percent_owner,
            "officer_title": t.officer_title,
            "security_title": t.security_title,
            "is_derivative": t.is_derivative,
            "transaction_code": t.transaction_code,
            "shares": t.shares,
            "price_per_share": t.price_per_share,
            "acquired_disposed": t.acquired_disposed,
            "post_transaction_shares": t.post_transaction_shares,
            "direct_or_indirect": t.direct_or_indirect,
            "exercise_price": t.exercise_price,
            "expiration_date": t.expiration_date,
            "form": filing.form,
            "accession": filing.accession,
        })
    return rows


def _flush_insider_rows(pit: PITEngine, rows: list[dict]) -> int:
    """Write buffered insider_trade rows via PITEngine (per-entity dedup).

    PITEngine.write groups by entity_id and runs a read-merge-dedup on the
    canonical per-entity parquet, so re-ingesting the same Form 4 is a no-op.
    """
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    pit.write(
        Modality.INSIDER_TRADES,
        df,
        partition_keys=InsiderTrade.PARTITION_KEYS,
        schema_cls=InsiderTrade,
    )
    return len(rows)


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
    pit: PITEngine,
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
    insider_buffer: list[dict] = []
    # Form 4/A amendments need a supersession sweep AFTER their rows are
    # written. We defer them until the day's writes have flushed so the
    # supersede sweep sees the amendment row in canonical (it filters out
    # `accession != amendment_accession`, so this is a correctness-neutral
    # ordering but it keeps the read in supersede consistent).
    pending_supersedes: list[tuple[str, str, date, str, date]] = []
    latest_per_cik: dict[str, tuple[date, str]] = {}
    try:
        for filing in iter_filings(tarball, keep_ciks=keep_ciks):
            stats.filings += 1
            for row in filings_to_rows(iter([filing])):
                rows_by_cik[(row["cik"], row["year"])].append(row)
                stats.docs += 1
            # Form 4 / 4/A: parse ownership XML into insider_trade rows.
            if filing.form in FORM4_FORMS and filing.filed_date:
                parsed = extract_form4(filing)
                if parsed is not None:
                    stats.form4_filings += 1
                    trades = _form4_to_insider_rows(filing, parsed)
                    insider_buffer.extend(trades)
                    stats.form4_trades += len(trades)
                    if filing.form == "4/A" and parsed.period_of_report is not None:
                        pending_supersedes.append((
                            f"us-cik-{filing.primary_cik}",
                            parsed.trades[0].insider_cik,
                            parsed.period_of_report,
                            filing.accession,
                            filing.filed_date,
                        ))
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
            if len(insider_buffer) >= INSIDER_FLUSH_ROWS:
                _flush_insider_rows(pit, insider_buffer)
                insider_buffer = []
    except Exception as e:
        stats.status = "parse_failed"
        stats.error = str(e)[:200]
        stats.elapsed_s = time.monotonic() - t0
        return stats

    # Final flush
    if rows_by_cik:
        _write_per_cik_batch(storage, rows_by_cik)
    if insider_buffer:
        _flush_insider_rows(pit, insider_buffer)

    # Run amendment supersession sweeps now that all insider rows are written.
    for entity_id, insider_cik, period, amend_accn, amend_filed in pending_supersedes:
        try:
            n = supersede_prior_form4(
                storage=storage,
                entity_id=entity_id,
                insider_cik=insider_cik,
                period_of_report=period,
                amendment_accession=amend_accn,
                amendment_filed=amend_filed,
            )
            stats.form4_supersedes += n
        except Exception as e:  # noqa: BLE001
            # Supersession is a hardening sweep; failures shouldn't abort the day.
            emit_event(
                PIPELINE, INSIDER_MODALITY, "supersede_failed",
                accession=amend_accn, error=str(e)[:200],
            )

    stats.ciks_touched = len(latest_per_cik)

    # Update watermarks. Failure here is non-fatal (the canonical write already
    # succeeded) but must be visible — silent failure causes infinite re-ingest.
    for cik, (filed, accn) in latest_per_cik.items():
        try:
            watermark.set(cik, WM_SOURCE, accn, filed)
        except Exception as e:  # noqa: BLE001
            log_parse(storage, cik, accn or "", "feed",
                      "parse_failed", error_message=f"watermark_set: {e}")

    stats.status = (
        "ok_with_records"
        if (stats.docs > 0 or stats.form4_trades > 0)
        else "ok_no_records"
    )
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
    ap.add_argument(
        "--skip-resource-check", action="store_true",
        help="Bypass disk/memory pre-flight checks (use only on known-good hosts)",
    )
    args = ap.parse_args()
    if args.end < args.start:
        raise SystemExit("--end must be >= --start")

    storage = get_storage()
    pit = PITEngine(storage)
    # Pre-flight before slow downloads; Feed days can be 3 GB compressed and
    # peak resident is ~8 GB while buffering all-filings-of-day.
    if not args.skip_resource_check:
        import tempfile
        preflight(
            tempfile.gettempdir(), disk_min_gb=DISK_MIN_GB, mem_min_gb=MEM_MIN_GB,
            hint="Feed peak days are ~3GB compressed / ~8GB resident",
        )
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

    job_id = emit_start(
        PIPELINE, FILINGS_MODALITY,
        start_day=args.start.isoformat(), end_day=args.end.isoformat(),
        n_days=len(days), n_ciks=len(keep_ciks),
    )

    t_start = time.monotonic()
    total_docs = 0
    total_bytes = 0
    total_form4_trades = 0
    total_form4_supersedes = 0
    for i, day in enumerate(days, 1):
        if not args.skip_resource_check:
            import tempfile
            try:
                preflight(
                    tempfile.gettempdir(), disk_min_gb=DISK_MIN_GB,
                    mem_min_gb=MEM_MIN_GB, hint=f"Feed day {day}",
                )
            except RuntimeError as e:
                emit_event(PIPELINE, FILINGS_MODALITY, "resource_exhausted",
                           parent_event_id=job_id, day=day.isoformat(), error=str(e))
                raise
        stats = _ingest_day(day, keep_ciks, storage, watermark, pit)
        total_docs += stats.docs
        total_bytes += stats.download_bytes
        total_form4_trades += stats.form4_trades
        total_form4_supersedes += stats.form4_supersedes
        log_parse(
            storage, cik="feed", accession=day.isoformat(),
            modality=FILINGS_MODALITY, status=stats.status,
            n_records=stats.docs, error_message=stats.error,
        )
        emit_event(
            PIPELINE, FILINGS_MODALITY, stats.status,
            parent_event_id=job_id, day=day.isoformat(),
            n_filings=stats.filings, n_records=stats.docs,
            ciks_touched=stats.ciks_touched,
            download_bytes=stats.download_bytes, wall_seconds=stats.elapsed_s,
            error=stats.error or None,
        )
        # Separate event surface for insider_trades so the modality has its
        # own per-day timeseries in observability.
        if stats.form4_filings or stats.form4_trades:
            emit_event(
                PIPELINE, INSIDER_MODALITY,
                "ok_with_records" if stats.form4_trades else "ok_no_records",
                parent_event_id=job_id, day=day.isoformat(),
                n_form4_filings=stats.form4_filings,
                n_records=stats.form4_trades,
                n_supersedes=stats.form4_supersedes,
            )
        pace = i / max(time.monotonic() - t_start, 1e-9) * 3600  # days/hour
        eta_h = (len(days) - i) / max(pace, 1e-9)
        print(_tag(
            f"[{i}/{len(days)}] {day} status={stats.status} "
            f"filings={stats.filings:>5} docs={stats.docs:>6} ciks={stats.ciks_touched:>4} "
            f"f4={stats.form4_filings:>4} trades={stats.form4_trades:>5} "
            f"dl={stats.download_bytes/1e6:>7.1f}MB wall={stats.elapsed_s:>6.1f}s "
            f"pace={pace:.1f}d/h eta={eta_h:.2f}h"
            + (f" ERR={stats.error[:80]}" if stats.error else "")
        ))

    wall = time.monotonic() - t_start
    emit_end(PIPELINE, FILINGS_MODALITY, "ok_with_records", job_id, wall,
             n_records=total_docs, download_bytes=total_bytes,
             n_days=len(days), n_insider_trades=total_form4_trades,
             n_form4_supersedes=total_form4_supersedes)
    print()
    print(
        f"[done] {len(days)} days in {wall:.1f}s; "
        f"total docs={total_docs:,}; insider trades={total_form4_trades:,}; "
        f"4/A supersedes={total_form4_supersedes:,}; "
        f"download={total_bytes/1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
