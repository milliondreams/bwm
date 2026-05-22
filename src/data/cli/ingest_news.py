"""Stage M4 driver: ingest GDELT 2.0 news events filtered to registry entities.

GDELT publishes a CSV of global events every 15 minutes at:
    http://data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMM}00.export.CSV.zip

We walk a date range in 15-min steps, fetch + unzip + parse each slot's
events, resolve Actor1Name/Actor2Name to registry CIKs via
case-insensitive name match, and write the matching events under the
focal CIK's canonical/news partition.

Run:
    # Apple-earnings-week smoke (small, fast)
    uv run python -m data.cli.ingest_news --start 2024-01-22 --end 2024-01-25

    # Year-long backfill (operational; minutes-to-hours depending on filter)
    uv run python -m data.cli.ingest_news --start 2020-01-01 --end 2025-01-01

Per the M4 design: focal entity is whichever actor resolves to a CIK
first (Actor1 preferred, Actor2 fallback). Unresolved events are dropped
when --registry-filter on (default) or kept under unresolved:{actor} when off.
"""
from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from collections import defaultdict
from datetime import date as date_cls, datetime, timedelta

import pandas as pd
import requests

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.news_event import NewsEvent
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.edgar.state import WatermarkStore
from data.sources.news.gdelt import parse_gdelt_csv
from data.storage import get_storage

WM_SOURCE = "gdelt:events2"
GDELT_URL_TPL = "http://data.gdeltproject.org/gdeltv2/{stamp}00.export.CSV.zip"
USER_AGENT = "norn-bwm research rohit@dragonscale.ai"


def _iter_slots(start: datetime, end: datetime):
    """Yield datetime objects at 15-minute intervals in [start, end)."""
    cur = start
    step = timedelta(minutes=15)
    while cur < end:
        yield cur
        cur += step


def _fetch_slot_csv(when: datetime, timeout_s: float = 30.0) -> bytes:
    """Download the GDELT 15-min export zip; return the unzipped TSV body."""
    stamp = when.strftime("%Y%m%d%H%M")
    url = GDELT_URL_TPL.format(stamp=stamp)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # GDELT export.CSV.zip contains exactly one .CSV member.
        names = zf.namelist()
        if not names:
            return b""
        return zf.read(names[0])


def _build_name_index(reg_df: pd.DataFrame) -> dict[str, str]:
    """Case-insensitive name → CIK lookup. SEC names are typically uppercase
    in tickers.json so the index keys are already uppercase; GDELT actor
    names are uppercased before lookup."""
    idx: dict[str, str] = {}
    for _, r in reg_df.iterrows():
        nm = str(r.get("name") or "").strip().upper()
        if nm and nm not in idx:
            idx[nm] = str(r["cik"])
    return idx


def _resolve_actor(name: str, name_idx: dict[str, str]) -> str | None:
    if not name:
        return None
    key = name.strip().upper()
    return name_idx.get(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2024-01-22")
    ap.add_argument("--end", type=str, default="2024-01-25")
    ap.add_argument("--registry-filter", choices=("on", "off"), default="on",
                    help="on (default): only persist events whose actor1/2 resolves to a CIK")
    ap.add_argument("--flush-every", type=int, default=10_000,
                    help="flush canonical rows when buffer exceeds this many entries")
    ap.add_argument("--max-events-per-slot", type=int, default=0,
                    help="cap events kept per 15-min slot AFTER filter (0 = no cap)")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    wm = WatermarkStore(storage)
    registry = EntityRegistry(storage)

    reg_df = registry.load()
    name_idx = _build_name_index(reg_df)
    print(f"=== news ingest: {args.start} → {args.end} ===")
    print(f"registry name-index size: {len(name_idx)}; filter={args.registry_filter}")

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    slots = list(_iter_slots(start_dt, end_dt))
    total_slots = len(slots)
    print(f"slots: {total_slots} (15-min cadence)\n")

    # Resume support: a single global watermark on (cik="global", source=gdelt:events2).
    last_wm = wm.get("global", WM_SOURCE)
    if last_wm and last_wm.last_filed_date:
        resume_from = datetime.combine(last_wm.last_filed_date, datetime.min.time())
        if resume_from > start_dt:
            print(f"  resuming from watermark {resume_from.isoformat()}")
            slots = [s for s in slots if s >= resume_from]

    buffer: dict[str, list[dict]] = defaultdict(list)
    buffered_rows = 0
    processed_slots = 0
    fetched_events = 0
    matched_events = 0
    fetch_errors = 0
    start_wall = time.monotonic()
    PROGRESS_EVERY = max(1, total_slots // 100) if total_slots > 100 else 4

    def _flush_buffer():
        nonlocal buffered_rows
        if not buffer:
            return
        for cik, rows in buffer.items():
            df = pd.DataFrame(rows)
            pit.write(Modality.NEWS, df, partition_keys=NewsEvent.PARTITION_KEYS)
        log_parse(storage, "global", WM_SOURCE, "news",
                  "ok_with_records", n_records=buffered_rows)
        print(f"  [flush] wrote {buffered_rows:,} rows across {len(buffer)} entities", flush=True)
        buffer.clear()
        buffered_rows = 0

    for slot in slots:
        try:
            body = _fetch_slot_csv(slot)
        except Exception as e:  # noqa: BLE001
            fetch_errors += 1
            log_parse(storage, "global", f"slot:{slot.isoformat()}", "news",
                      "fetch_failed", error_message=f"{type(e).__name__}: {e}")
            processed_slots += 1
            continue

        slot_matched = 0
        for ev in parse_gdelt_csv(body):
            fetched_events += 1
            cik = _resolve_actor(ev.actor1_name, name_idx) or _resolve_actor(ev.actor2_name, name_idx)
            if cik is None:
                if args.registry_filter == "on":
                    continue
                cik = f"unresolved:{(ev.actor1_name or ev.actor2_name or 'na')[:32]}"
                entity_id = cik  # synthetic
            else:
                entity_id = EntityRegistry.entity_id_from_cik(cik)

            buffer[cik].append({
                "entity_id": entity_id,
                "modality": Modality.NEWS.value,
                "effective_date": ev.event_date,
                "availability_date": ev.date_added,
                "restated_at": None,
                "source": "gdelt:events2",
                "source_ref": ev.global_event_id,
                "global_event_id": ev.global_event_id,
                "cameo_event_code": ev.cameo_event_code,
                "event_root_code": ev.event_root_code,
                "quad_class": ev.quad_class,
                "goldstein_scale": ev.goldstein_scale,
                "tone": ev.avg_tone,
                "source_url": ev.source_url,
                "actor1_name": ev.actor1_name,
                "actor2_name": ev.actor2_name,
                "n_mentions": ev.n_mentions,
                "avg_tone": ev.avg_tone,
            })
            slot_matched += 1
            buffered_rows += 1
            if args.max_events_per_slot > 0 and slot_matched >= args.max_events_per_slot:
                break

        matched_events += slot_matched
        processed_slots += 1

        if buffered_rows >= args.flush_every:
            _flush_buffer()
            # Persist watermark after each flush.
            import asyncio
            asyncio.run(wm.set_async("global", WM_SOURCE, None, slot.date()))

        if processed_slots % PROGRESS_EVERY == 0 or processed_slots == total_slots:
            elapsed_h = (time.monotonic() - start_wall) / 3600
            pace = processed_slots / max(elapsed_h * 3600, 1e-6)  # slots/sec
            remaining = (total_slots - processed_slots) / max(pace, 1e-6) / 3600
            print(
                f"[progress] slot {processed_slots}/{total_slots} ({100.0 * processed_slots / total_slots:.1f}%)  "
                f"events={fetched_events:,} matched={matched_events:,} "
                f"fetch_errs={fetch_errors} "
                f"elapsed={elapsed_h:.2f}h ETA={remaining:.2f}h",
                flush=True,
            )

    _flush_buffer()
    if slots:
        import asyncio
        asyncio.run(wm.set_async("global", WM_SOURCE, None, slots[-1].date()))

    print("\n=== summary ===")
    print(f"  slots processed:  {processed_slots}/{total_slots}")
    print(f"  fetch errors:     {fetch_errors}")
    print(f"  events parsed:    {fetched_events:,}")
    print(f"  events matched:   {matched_events:,}  "
          f"({100.0 * matched_events / max(fetched_events, 1):.2f}%)")


if __name__ == "__main__":
    main()
