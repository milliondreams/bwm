"""Stage M3 batch driver: ingest patents across the registry via PatentsView.

Run:
    # Whole registry, grant dates 2000→today
    uv run python -m data.cli.ingest_patents_all --all

    # Wider pilot
    uv run python -m data.cli.ingest_patents_all --limit-ciks 100

Mirrors `ingest_edgar_full` patterns: dedup, asyncio semaphore, [progress]
checkpoints, watermark-gated incremental ingest. PatentsView's free tier
has historically rate-limited heavy use, so concurrency defaults to 2.

Each CIK resolves to an assignee name via the registry. PatentsView matches
names case-insensitively against `assignee_organization`. Entities whose
name returns no patents land with `n_records=0` in the parse log — that's
expected for many CIKs (financial firms, holding companies) and not an
error. R8-style hardening (manual name overrides) is a follow-on if smoke
shows resolution issues.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date as date_cls, timedelta

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.patent import Patent
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.edgar.state import WatermarkStore
from data.sources.patents.uspto import fetch_patents_by_assignee
from data.storage import get_storage

WM_SOURCE = "patentsview"


@dataclass
class CikStats:
    cik: str
    assignee: str = ""
    patents: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def _patent_rows(cik: str, entity_id: str, patents: list) -> list[dict]:
    return [
        {
            "entity_id": entity_id,
            "modality": Modality.PATENTS.value,
            "effective_date": p.application_date,
            "availability_date": p.grant_date,
            "restated_at": None,
            "source": "uspto:patentsview",
            "source_ref": p.patent_number,
            "patent_number": p.patent_number,
            "application_date": p.application_date,
            "grant_date": p.grant_date,
            "assignee_name_as_filed": p.assignee_name_as_filed,
            "title": p.title,
            "abstract": p.abstract,
            "claims_count": p.claims_count,
            "cpc_classifications": json.dumps(p.cpc_classifications),
            "inventors": json.dumps(p.inventors),
        }
        for p in patents
    ]


def _resolve_start(wm: WatermarkStore, cik: str, default_start: date_cls) -> date_cls:
    w = wm.get(cik, WM_SOURCE)
    if w and w.last_filed_date:
        return w.last_filed_date + timedelta(days=1)
    return default_start


async def _process_cik(
    storage,
    pit: PITEngine,
    wm: WatermarkStore,
    registry_row: dict,
    args,
    default_start: date_cls,
    default_end: date_cls,
) -> CikStats:
    cik = registry_row["cik"]
    assignee = str(registry_row.get("name") or "").strip()
    s = CikStats(cik=cik, assignee=assignee)
    t0 = time.monotonic()
    if not assignee:
        s.errors.append("no name in registry")
        s.elapsed_s = time.monotonic() - t0
        return s

    start = _resolve_start(wm, cik, default_start)
    if start >= default_end:
        s.elapsed_s = time.monotonic() - t0
        return s

    entity_id = EntityRegistry.entity_id_from_cik(cik)
    api_key = os.environ.get("PATENTSVIEW_API_KEY")
    try:
        patents = await asyncio.to_thread(
            lambda: list(
                fetch_patents_by_assignee(
                    assignee,
                    start_grant_date=start,
                    end_grant_date=default_end,
                    api_key=api_key,
                )
            )
        )
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        s.errors.append(msg)
        log_parse(storage, cik, f"patentsview:{assignee}", "patents",
                  "fetch_failed", error_message=msg)
        s.elapsed_s = time.monotonic() - t0
        return s

    if args.max_patents_per_cik > 0:
        patents = patents[: args.max_patents_per_cik]

    if not patents:
        log_parse(storage, cik, f"patentsview:{assignee}", "patents", "ok_no_records")
        s.elapsed_s = time.monotonic() - t0
        return s

    rows = _patent_rows(cik, entity_id, patents)
    df = pd.DataFrame(rows)
    pit.write(Modality.PATENTS, df, partition_keys=Patent.PARTITION_KEYS)
    log_parse(storage, cik, f"patentsview:{assignee}", "patents",
              "ok_with_records", n_records=len(rows))
    s.patents = len(rows)
    latest = max(p.grant_date for p in patents)
    await wm.set_async(cik, WM_SOURCE, None, latest)
    s.elapsed_s = time.monotonic() - t0
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true")
    g.add_argument("--limit-ciks", type=int, default=0)
    g.add_argument("--ciks", nargs="*", default=[])

    ap.add_argument("--start-grant", type=str, default="2000-01-01")
    ap.add_argument("--end-grant", type=str, default=date_cls.today().isoformat())
    ap.add_argument("--concurrency", type=int, default=2,
                    help="number of CIKs in flight (PatentsView rate-limits heavy use)")
    ap.add_argument("--max-patents-per-cik", type=int, default=0,
                    help="cap patents per CIK; 0 = no cap")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    wm = WatermarkStore(storage)
    registry = EntityRegistry(storage)

    df = registry.load()
    if args.ciks:
        wanted = sorted({c.zfill(10) for c in args.ciks})
        df = df[df["cik"].isin(wanted)]
    elif args.limit_ciks > 0:
        df = df.head(args.limit_ciks)
    elif not args.all:
        ap.error("specify --all, --limit-ciks N, or --ciks <list>")

    df = df.drop_duplicates(subset=["cik"]).reset_index(drop=True)
    rows = df.to_dict(orient="records")
    total = len(rows)
    if total == 0:
        print("no CIKs to process")
        return

    default_start = date_cls.fromisoformat(args.start_grant)
    default_end = date_cls.fromisoformat(args.end_grant)

    print(f"=== patents ingest: {total} CIKs, concurrency={args.concurrency} ===")
    print(f"grant_date {args.start_grant} → {args.end_grant}\n")

    sem = asyncio.Semaphore(args.concurrency)
    progress_lock = asyncio.Lock()
    start_wall = time.monotonic()
    completed_count = 0
    error_count = 0
    total_patents = 0
    nonzero_entities = 0
    PROGRESS_EVERY = max(1, total // 200) if total > 200 else 10

    async def _wrapped(row: dict) -> CikStats:
        nonlocal completed_count, error_count, total_patents, nonzero_entities
        async with sem:
            s = await _process_cik(storage, pit, wm, row, args,
                                   default_start, default_end)
            tag = "OK" if not s.errors else f"ERR×{len(s.errors)}"
            print(f"  [{tag}] cik={s.cik} assignee={s.assignee[:40]!r} patents={s.patents} in {s.elapsed_s:.1f}s",
                  flush=True)
            if s.errors:
                for e in s.errors[:3]:
                    print(f"      → {e}", flush=True)
            async with progress_lock:
                completed_count += 1
                total_patents += s.patents
                if s.patents > 0:
                    nonzero_entities += 1
                if s.errors:
                    error_count += 1
                n = completed_count
                if n % PROGRESS_EVERY == 0 or n == total:
                    elapsed_h = (time.monotonic() - start_wall) / 3600
                    pace = n / max(elapsed_h, 1e-6)
                    remaining_h = (total - n) / max(pace, 1e-6)
                    print(
                        f"[progress] {n}/{total} ({100.0 * n / total:.1f}%)  "
                        f"pace={pace:.0f}/h  "
                        f"errs={error_count}  "
                        f"patents={total_patents:,}  "
                        f"nonzero={nonzero_entities}  "
                        f"elapsed={elapsed_h:.2f}h  ETA={remaining_h:.1f}h",
                        flush=True,
                    )
            return s

    async def _amain() -> None:
        await asyncio.gather(*(_wrapped(r) for r in rows))

    asyncio.run(_amain())

    print("\n=== summary ===")
    print(f"  CIKs processed:   {completed_count}")
    print(f"  total patents:    {total_patents:,}")
    print(f"  nonzero entities: {nonzero_entities}/{completed_count}")
    print(f"  errors:           {error_count}")


if __name__ == "__main__":
    main()
