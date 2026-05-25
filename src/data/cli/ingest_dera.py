"""Bulk DERA ingest: SEC Financial Statement Data Sets → canonical/financials.

Replaces the rate-limited per-filing XBRL path with the SEC's quarterly bulk
dumps. ~52 quarters (2012-Q1 → 2024-Q4) ≈ 6 GB compressed download, ~3 GB
of relevant facts after registry filtering, written as per-CIK Lance shards.

Run:
    # Whole 2012→2025 corpus (full Phase A backfill)
    BWM_STORAGE_BACKEND=lance \
        uv run python -m data.cli.ingest_dera --start 2012Q1 --end 2025Q1

    # Single quarter (smoke / debug)
    uv run python -m data.cli.ingest_dera --start 2024Q4 --end 2024Q4

    # Wider pre-2012 sweep (sparse — pre-XBRL-mandate)
    uv run python -m data.cli.ingest_dera --start 2009Q1 --end 2011Q4

Architecture:
    1. Enumerate target quarters
    2. For each: download zip → filter to registry CIKs → parse num × sub join
       → emit FinancialFact-shaped DataFrame → PITEngine.write (per-CIK shards)
    3. Update watermark (cik, "sec_edgar:dera", quarter_id) after each batch

Compared to the per-filing API path that broke us last round:
    - No rate limit (zip download is CDN-served)
    - No cap=80 per-CIK budget — every filing in the quarter lands
    - Preserves XBRL `segments` (dimensions) verbatim, fixing the GE 2017
      consolidated-vs-segment collision

The driver is single-process; parallelism (across quarters) is achieved by
running multiple jobs with disjoint --start/--end ranges. Each quarter is
self-contained — no cross-quarter dependency in the parse — so this scales
trivially across AML compute nodes.
"""
from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.financial import FinancialFact
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.edgar.dera_bulk import (
    DeraQuarter,
    download_quarter,
    enumerate_quarters,
    parse_quarter,
)
from data.sources.edgar.state import WatermarkStore
from data.storage import get_storage

WM_SOURCE = "sec_edgar:dera"
_QUARTER_RE = re.compile(r"^(\d{4})[Qq]([1-4])$")


def _parse_quarter_arg(s: str) -> DeraQuarter:
    m = _QUARTER_RE.match(s)
    if not m:
        raise argparse.ArgumentTypeError(
            f"--start/--end must look like '2024Q3', got: {s!r}"
        )
    return DeraQuarter(int(m.group(1)), int(m.group(2)))


@dataclass
class QuarterStats:
    quarter: DeraQuarter
    download_bytes: int = 0
    rows_written: int = 0
    ciks_touched: int = 0
    elapsed_s: float = 0.0
    status: str = "pending"
    error: str = ""


def _ingest_quarter(
    q: DeraQuarter,
    engine: PITEngine,
    keep_ciks: set[str],
    watermark: WatermarkStore,
    storage,
) -> QuarterStats:
    """Pull + parse + write one DERA quarter. Updates watermarks per CIK touched."""
    stats = QuarterStats(quarter=q)
    t0 = time.monotonic()
    try:
        zip_bytes = download_quarter(q)
    except FileNotFoundError:
        stats.status = "missing"
        stats.elapsed_s = time.monotonic() - t0
        return stats
    except Exception as e:
        stats.status = "fetch_failed"
        stats.error = str(e)
        stats.elapsed_s = time.monotonic() - t0
        return stats

    stats.download_bytes = len(zip_bytes)
    try:
        df = parse_quarter(zip_bytes, keep_ciks=keep_ciks)
    except Exception as e:
        stats.status = "parse_failed"
        stats.error = str(e)
        stats.elapsed_s = time.monotonic() - t0
        return stats

    if df.empty:
        stats.status = "ok_no_records"
        stats.elapsed_s = time.monotonic() - t0
        return stats

    # PITEngine writes per-CIK Lance shards; one write per entity in this batch.
    engine.write(
        Modality.FINANCIALS,
        df,
        partition_keys=FinancialFact.PARTITION_KEYS,
        schema_cls=FinancialFact,
    )
    stats.rows_written = len(df)
    stats.ciks_touched = df["entity_id"].nunique()

    # Watermark per CIK so re-runs are incremental. Watermark key is the
    # accession; here we record the latest filed date per CIK for this quarter.
    for cik_str, batch in df.groupby("entity_id"):
        cik10 = cik_str.removeprefix("us-cik-")
        latest = batch["availability_date"].max()
        latest_accn = (
            batch.sort_values("availability_date").iloc[-1]["source_ref"]
        )
        try:
            watermark.update(cik10, WM_SOURCE, latest_accn, latest)
        except Exception:
            # Watermark update is best-effort; ingest itself succeeded
            pass

    stats.status = "ok_with_records"
    stats.elapsed_s = time.monotonic() - t0
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--start", type=_parse_quarter_arg, required=True,
        help="First quarter to ingest (e.g. 2012Q1)",
    )
    ap.add_argument(
        "--end", type=_parse_quarter_arg, required=True,
        help="Last quarter to ingest, inclusive (e.g. 2024Q4)",
    )
    ap.add_argument(
        "--limit-ciks", type=int, default=0,
        help="If >0, restrict ingest to first N CIKs (smoke / debug)",
    )
    args = ap.parse_args()

    if (args.end.year, args.end.quarter) < (args.start.year, args.start.quarter):
        raise SystemExit("--end must be >= --start")

    storage = get_storage()
    registry = EntityRegistry(storage)
    reg_df = registry.load()
    ciks_all = sorted(set(reg_df["cik"].astype(str).str.zfill(10).tolist()))
    if args.limit_ciks > 0:
        ciks_all = ciks_all[: args.limit_ciks]
    keep_ciks = set(ciks_all)
    print(f"[setup] backend={type(storage).__name__} ciks={len(keep_ciks)}")

    engine = PITEngine(storage)
    watermark = WatermarkStore(storage)

    quarters = [
        q for q in enumerate_quarters(args.start.year, args.end.year)
        if (q.year, q.quarter) >= (args.start.year, args.start.quarter)
        and (q.year, q.quarter) <= (args.end.year, args.end.quarter)
    ]
    print(f"[setup] processing {len(quarters)} quarters: {quarters[0]} → {quarters[-1]}")

    t_start = time.monotonic()
    total_rows = 0
    total_bytes = 0
    for i, q in enumerate(quarters, 1):
        stats = _ingest_quarter(q, engine, keep_ciks, watermark, storage)
        total_rows += stats.rows_written
        total_bytes += stats.download_bytes
        log_parse(
            storage,
            cik="dera",
            accession=str(q),
            modality="financials",
            status=stats.status,
            n_records=stats.rows_written,
            error_message=stats.error,
        )
        pace = i / max(time.monotonic() - t_start, 1e-9) * 3600  # quarters/hour
        eta_h = (len(quarters) - i) / max(pace, 1e-9)
        print(
            f"[{i}/{len(quarters)}] {q} status={stats.status} "
            f"rows={stats.rows_written:>9,} ciks={stats.ciks_touched:>5} "
            f"dl={stats.download_bytes/1e6:>6.1f}MB "
            f"wall={stats.elapsed_s:>5.1f}s "
            f"pace={pace:.1f}/h eta={eta_h:.2f}h"
            + (f" ERR={stats.error[:80]}" if stats.error else "")
        )

    print()
    print(
        f"[done] {len(quarters)} quarters in {time.monotonic()-t_start:.1f}s; "
        f"total rows={total_rows:,}; download={total_bytes/1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
