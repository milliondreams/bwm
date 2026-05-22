"""Stage M1 batch driver: ingest daily market bars across the registry.

Run:
    # Whole registry, 2015→today, daily
    uv run python -m data.cli.ingest_market_all --all

    # Wider pilot (100 CIKs)
    uv run python -m data.cli.ingest_market_all --limit-ciks 100

    # Explicit CIK list
    uv run python -m data.cli.ingest_market_all --ciks 0000320193 0000789019

Mirrors `ingest_edgar_full` patterns: registry dedup, asyncio semaphore,
per-CIK [OK]/[ERR] lines, periodic [progress] line with pace + ETA,
WatermarkStore-gated incremental ingest.

Yfinance is synchronous; we wrap each per-CIK fetch in `asyncio.to_thread`
so the semaphore limits the number of concurrent web scrapes. Yahoo has no
documented rate limit but practical experience caps sustained throughput
around 5 req/s before IP bans — `--concurrency 4` (default) stays well
under that.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import date as date_cls, timedelta

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.market_bar import MarketBar
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.edgar.state import WatermarkStore
from data.sources.market.yahoo import fetch_history
from data.storage import get_storage

WM_SOURCE = "yahoo_finance"


@dataclass
class CikStats:
    cik: str
    ticker: str = ""
    bars: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def _bar_rows(
    cik: str, entity_id: str, ticker: str, bars: list, freq_label: str
) -> list[dict]:
    return [
        {
            "entity_id": entity_id,
            "modality": Modality.MARKET.value,
            "effective_date": b.bar_date,
            "availability_date": b.bar_date,
            "restated_at": None,
            "source": "yahoo_finance",
            "source_ref": f"{ticker}:{b.bar_date.isoformat()}",
            "frequency": freq_label,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "adj_close": b.adj_close,
            "volume": b.volume,
            "dividend": b.dividend,
            "split_ratio": b.split_ratio,
            "ticker": ticker,
        }
        for b in bars
    ]


def _resolve_start(args, wm: WatermarkStore, cik: str, default_start: date_cls) -> date_cls:
    """If a watermark exists, resume from the day after the last bar."""
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
    freq_label: str,
) -> CikStats:
    cik = registry_row["cik"]
    ticker = str(registry_row.get("ticker") or "").upper()
    s = CikStats(cik=cik, ticker=ticker)
    t0 = time.monotonic()
    if not ticker:
        s.errors.append("no ticker in registry")
        s.elapsed_s = time.monotonic() - t0
        return s

    start = _resolve_start(args, wm, cik, default_start)
    if start >= default_end:
        # Already up to date.
        s.elapsed_s = time.monotonic() - t0
        return s

    entity_id = EntityRegistry.entity_id_from_cik(cik)
    try:
        bars = await asyncio.to_thread(
            lambda: list(fetch_history(ticker, start=start, end=default_end, interval=args.interval))
        )
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        s.errors.append(msg)
        log_parse(storage, cik, f"{ticker}:{args.interval}", "market",
                  "fetch_failed", error_message=msg)
        s.elapsed_s = time.monotonic() - t0
        return s

    if not bars:
        log_parse(storage, cik, f"{ticker}:{args.interval}", "market", "ok_no_records")
        s.elapsed_s = time.monotonic() - t0
        return s

    if args.max_bars_per_cik > 0:
        bars = bars[-args.max_bars_per_cik:]  # keep most recent

    rows = _bar_rows(cik, entity_id, ticker, bars, freq_label)
    df = pd.DataFrame(rows)
    pit.write(Modality.MARKET, df, partition_keys=MarketBar.PARTITION_KEYS)
    log_parse(storage, cik, f"{ticker}:{args.interval}", "market",
              "ok_with_records", n_records=len(rows))
    s.bars = len(rows)

    latest = bars[-1].bar_date
    await wm.set_async(cik, WM_SOURCE, None, latest)
    s.elapsed_s = time.monotonic() - t0
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true")
    g.add_argument("--limit-ciks", type=int, default=0)
    g.add_argument("--ciks", nargs="*", default=[])

    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=date_cls.today().isoformat())
    ap.add_argument("--interval", default="1d", choices=("1d", "1wk", "1mo"))
    ap.add_argument("--concurrency", type=int, default=4,
                    help="number of CIKs in flight (yfinance has no documented rate limit; "
                         "stay ≤5 to avoid IP bans)")
    ap.add_argument("--max-bars-per-cik", type=int, default=0,
                    help="cap bars per CIK; 0 = no cap (keep most-recent N if >0)")
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

    # Dedup at dispatch (defense in depth).
    df = df.drop_duplicates(subset=["cik"]).reset_index(drop=True)
    rows = df.to_dict(orient="records")
    total = len(rows)
    if total == 0:
        print("no CIKs to process")
        return

    default_start = date_cls.fromisoformat(args.start)
    default_end = date_cls.fromisoformat(args.end)
    freq_label = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}[args.interval]

    print(f"=== market ingest: {total} CIKs, concurrency={args.concurrency} ===")
    print(f"start={args.start}  end={args.end}  interval={args.interval}\n")

    sem = asyncio.Semaphore(args.concurrency)
    progress_lock = asyncio.Lock()
    start_wall = time.monotonic()
    completed_count = 0
    error_count = 0
    total_bars = 0
    PROGRESS_EVERY = max(1, total // 200) if total > 200 else 10

    async def _wrapped(row: dict) -> CikStats:
        nonlocal completed_count, error_count, total_bars
        async with sem:
            s = await _process_cik(storage, pit, wm, row, args,
                                   default_start, default_end, freq_label)
            tag = "OK" if not s.errors else f"ERR×{len(s.errors)}"
            print(f"  [{tag}] cik={s.cik} ticker={s.ticker} bars={s.bars} in {s.elapsed_s:.1f}s",
                  flush=True)
            if s.errors:
                for e in s.errors[:3]:
                    print(f"      → {e}", flush=True)
            async with progress_lock:
                completed_count += 1
                total_bars += s.bars
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
                        f"errs={error_count} ({100.0 * error_count / max(n, 1):.1f}%)  "
                        f"bars={total_bars:,}  "
                        f"elapsed={elapsed_h:.2f}h  ETA={remaining_h:.1f}h",
                        flush=True,
                    )
            return s

    async def _amain() -> None:
        await asyncio.gather(*(_wrapped(r) for r in rows))

    asyncio.run(_amain())

    print("\n=== summary ===")
    print(f"  CIKs processed:   {completed_count}")
    print(f"  total bars written: {total_bars:,}")
    print(f"  errors:           {error_count}")


if __name__ == "__main__":
    main()
