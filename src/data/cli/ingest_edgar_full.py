"""Stage 5: full-universe EDGAR ingest driver.

For a set of CIKs (default: every entity in the registry), orchestrates the
Stage 2/3/4 ingest paths under the shared token-bucket and per-CIK watermarks
established in Stages 1 / 1.5:

  - Stage 2: per-filing XBRL (10-K/10-Q/20-F/40-F) → canonical/financials
  - Stage 3: Form 4 / Form 4-A                    → canonical/insider_trades
  - Stage 4: 10-K / 10-Q / 8-K HTML               → canonical/filings_text

Resumability: every per-CIK source carries its own watermark
(`sec_edgar:xbrl`, `sec_edgar:form4`, `sec_edgar:filing_html`). Stopping and
re-running picks up where we left off; idempotent at the canonical layer
because PITEngine.write dedups on partition keys.

Concurrency: we run up to `--concurrency` CIKs in flight at once. The
token bucket (9 req/s sustained, declared in client.py) serializes the
actual HTTP layer across all coroutines so SEC's rate limit is respected
regardless of concurrency setting.

Usage:
    # Smoke run: 10 CIKs, 5 filings of each type, no filings_text
    uv run python -m data.cli.ingest_edgar_full \
        --limit-ciks 10 --max-filings-per-cik 5 --skip-text

    # Full backfill: every CIK in registry, all filings, all modalities
    uv run python -m data.cli.ingest_edgar_full --all
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.filing_text import FilingTextChunk
from data.schemas.financial import FinancialFact
from data.schemas.insider_trade import InsiderTrade
from data.schemas.pit import Modality
from data.sources.edgar.archive import archive, log_parse
from data.sources.edgar.client import EdgarClient, edgar_client
from data.sources.edgar.filings_text import CHARS_PER_TOKEN, extract_chunks
from data.sources.edgar.form4 import parse_form4_filing
from data.sources.edgar.form4_supersede import supersede_prior_form4
from data.sources.edgar.state import WatermarkStore
from data.sources.edgar.submissions import (
    FORM4_FORMS,
    TEXT_FORMS,
    XBRL_FORMS,
    Filing,
    list_filings,
)
from data.sources.edgar.xbrl import parse_ixbrl
from data.storage import get_storage


@dataclass
class CikStats:
    cik: str
    financials_filings: int = 0
    financial_rows: int = 0
    form4_filings: int = 0
    form4_trades: int = 0
    text_filings: int = 0
    text_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


# ---------- per-modality processors -----------------------------------------


async def _process_financials(
    client: EdgarClient,
    storage,
    pit: PITEngine,
    wm: WatermarkStore,
    cik: str,
    entity_id: str,
    stats: CikStats,
    max_filings: int,
) -> None:
    filings = await list_filings(client, cik, forms=XBRL_FORMS)
    candidates = [(f.accession, f.filed) for f in filings]
    fresh_accns = set(a for a, _ in wm.filter_new(cik, "sec_edgar:xbrl", candidates))
    fresh = [f for f in filings if f.accession in fresh_accns]
    if max_filings > 0:
        fresh = fresh[:max_filings]

    for filing in fresh:
        try:
            body = await client.get_bytes(filing.primary_doc_url)
        except Exception as e:  # noqa: BLE001
            stats.errors.append(f"xbrl {filing.accession}: {type(e).__name__}")
            log_parse(storage, cik, filing.accession, "financials", "fetch_failed",
                      error_message=f"{type(e).__name__}: {e}")
            continue
        archive(storage, cik, "primary", body, accession=filing.accession)
        rows = _xbrl_rows(filing, entity_id, body)
        if rows.empty:
            log_parse(storage, cik, filing.accession, "financials", "ok_no_records")
            continue
        pit.write(
            Modality.FINANCIALS, rows, partition_keys=FinancialFact.PARTITION_KEYS
        )
        log_parse(storage, cik, filing.accession, "financials",
                  "ok_with_records", n_records=len(rows))
        stats.financials_filings += 1
        stats.financial_rows += len(rows)

    if fresh:
        latest = max(fresh, key=lambda f: f.filed)
        await wm.set_async(cik, "sec_edgar:xbrl", latest.accession, latest.filed)


def _xbrl_rows(filing: Filing, entity_id: str, body: bytes) -> pd.DataFrame:
    facts = list(parse_ixbrl(body))
    if not facts:
        return pd.DataFrame()
    fp = "FY" if filing.form.startswith("10-K") else "Q"
    rows = []
    for f in facts:
        if f.context.period_end != filing.report_date:
            continue
        rows.append({
            "entity_id": entity_id,
            "modality": Modality.FINANCIALS.value,
            "effective_date": f.context.period_end,
            "availability_date": filing.filed,
            "restated_at": None,
            "source": "sec_edgar:xbrl_instance",
            "source_ref": filing.accession,
            "concept": f.concept,
            "taxonomy": f.taxonomy,
            "period_start": f.context.period_start,
            "period_end": f.context.period_end,
            "fiscal_year": filing.report_date.year,
            "fiscal_period": fp,
            "unit": f.unit,
            "value": f.value,
            "form": filing.form,
            "accession": filing.accession,
            "context_id": f.context.context_id,
            "dimensions_json": f.context.dimensions_json,
        })
    return pd.DataFrame(rows)


async def _process_form4(
    client: EdgarClient,
    storage,
    pit: PITEngine,
    wm: WatermarkStore,
    cik: str,
    entity_id: str,
    stats: CikStats,
    max_filings: int,
) -> None:
    filings = await list_filings(client, cik, forms=FORM4_FORMS)
    candidates = [(f.accession, f.filed) for f in filings]
    fresh_accns = set(a for a, _ in wm.filter_new(cik, "sec_edgar:form4", candidates))
    fresh = [f for f in filings if f.accession in fresh_accns]
    if max_filings > 0:
        fresh = fresh[:max_filings]

    for filing in fresh:
        try:
            body = await client.get_bytes(filing.primary_doc_raw_url)
        except Exception as e:  # noqa: BLE001
            stats.errors.append(f"form4 {filing.accession}: {type(e).__name__}")
            log_parse(storage, cik, filing.accession, "insider_trades", "fetch_failed",
                      error_message=f"{type(e).__name__}: {e}")
            continue
        archive(storage, cik, "form4", body, accession=filing.accession)
        parsed = parse_form4_filing(body)
        if parsed is None:
            log_parse(storage, cik, filing.accession, "insider_trades", "parse_failed")
            continue
        if not parsed.trades:
            log_parse(storage, cik, filing.accession, "insider_trades", "ok_no_records")
            continue
        rows = []
        for idx, t in enumerate(parsed.trades):
            rows.append({
                "entity_id": entity_id,
                "modality": Modality.INSIDER_TRADES.value,
                "effective_date": t.transaction_date,
                "availability_date": filing.filed,
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
                "transaction_index": idx,
                "period_of_report": parsed.period_of_report,
                "form": filing.form,
                "accession": filing.accession,
            })
        df = pd.DataFrame(rows)
        pit.write(
            Modality.INSIDER_TRADES, df, partition_keys=InsiderTrade.PARTITION_KEYS
        )
        if filing.form.endswith("/A") and parsed.period_of_report is not None:
            supersede_prior_form4(
                storage=storage,
                entity_id=entity_id,
                insider_cik=parsed.trades[0].insider_cik,
                period_of_report=parsed.period_of_report,
                amendment_accession=filing.accession,
                amendment_filed=filing.filed,
            )
        log_parse(storage, cik, filing.accession, "insider_trades",
                  "ok_with_records", n_records=len(parsed.trades))
        stats.form4_filings += 1
        stats.form4_trades += len(rows)

    if fresh:
        latest = max(fresh, key=lambda f: f.filed)
        await wm.set_async(cik, "sec_edgar:form4", latest.accession, latest.filed)


async def _process_text(
    client: EdgarClient,
    storage,
    pit: PITEngine,
    wm: WatermarkStore,
    cik: str,
    entity_id: str,
    stats: CikStats,
    max_filings: int,
) -> None:
    filings = await list_filings(client, cik, forms=TEXT_FORMS)
    candidates = [(f.accession, f.filed) for f in filings]
    fresh_accns = set(
        a for a, _ in wm.filter_new(cik, "sec_edgar:filing_html", candidates)
    )
    fresh = [f for f in filings if f.accession in fresh_accns]
    if max_filings > 0:
        fresh = fresh[:max_filings]

    for filing in fresh:
        try:
            body = await client.get_bytes(filing.primary_doc_url)
        except Exception as e:  # noqa: BLE001
            stats.errors.append(f"text {filing.accession}: {type(e).__name__}")
            log_parse(storage, cik, filing.accession, "filings_text", "fetch_failed",
                      error_message=f"{type(e).__name__}: {e}")
            continue
        archive(storage, cik, "primary", body, accession=filing.accession)
        chunks = list(extract_chunks(body, filing.form))
        if not chunks:
            log_parse(storage, cik, filing.accession, "filings_text", "ok_no_records")
            continue
        rows = []
        for c in chunks:
            rows.append({
                "entity_id": entity_id,
                "modality": Modality.FILINGS_TEXT.value,
                "effective_date": filing.report_date,
                "availability_date": filing.filed,
                "restated_at": None,
                "source": "sec_edgar:filing_html",
                "source_ref": filing.accession,
                "form": filing.form,
                "accession": filing.accession,
                "item_id": c.item_id,
                "item_title": c.item_title,
                "chunk_idx": c.chunk_idx,
                "n_chunks_in_item": c.n_chunks_in_item,
                "text": c.text,
                "n_chars": len(c.text),
                "n_tokens_approx": len(c.text) // CHARS_PER_TOKEN,
            })
        pit.write(
            Modality.FILINGS_TEXT,
            pd.DataFrame(rows),
            partition_keys=FilingTextChunk.PARTITION_KEYS,
        )
        log_parse(storage, cik, filing.accession, "filings_text",
                  "ok_with_records", n_records=len(rows))
        stats.text_filings += 1
        stats.text_chunks += len(rows)

    if fresh:
        latest = max(fresh, key=lambda f: f.filed)
        await wm.set_async(cik, "sec_edgar:filing_html", latest.accession, latest.filed)


# ---------- driver -----------------------------------------------------------


async def _process_cik(
    client: EdgarClient,
    storage,
    pit: PITEngine,
    wm: WatermarkStore,
    cik: str,
    args: argparse.Namespace,
) -> CikStats:
    t0 = time.monotonic()
    stats = CikStats(cik=cik)
    entity_id = EntityRegistry.entity_id_from_cik(cik)

    if not args.skip_financials:
        try:
            await _process_financials(
                client, storage, pit, wm, cik, entity_id, stats, args.max_filings_per_cik
            )
        except Exception as e:  # noqa: BLE001
            stats.errors.append(f"financials: {type(e).__name__}: {e}")

    if not args.skip_form4:
        try:
            await _process_form4(
                client, storage, pit, wm, cik, entity_id, stats, args.max_filings_per_cik
            )
        except Exception as e:  # noqa: BLE001
            stats.errors.append(f"form4: {type(e).__name__}: {e}")

    if not args.skip_text:
        try:
            await _process_text(
                client, storage, pit, wm, cik, entity_id, stats, args.max_filings_per_cik
            )
        except Exception as e:  # noqa: BLE001
            stats.errors.append(f"text: {type(e).__name__}: {e}")

    stats.elapsed_s = time.monotonic() - t0
    return stats


async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="ingest every CIK in registry")
    ap.add_argument("--ciks", nargs="*", default=[], help="explicit CIKs (override --all)")
    ap.add_argument("--limit-ciks", type=int, default=0, help="cap registry slice (0 = all)")
    ap.add_argument("--max-filings-per-cik", type=int, default=0,
                    help="cap filings per modality per CIK (0 = no cap)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="number of CIKs processed in parallel (rate-limit is global)")
    ap.add_argument("--skip-financials", action="store_true")
    ap.add_argument("--skip-form4", action="store_true")
    ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    wm = WatermarkStore(storage)
    registry = EntityRegistry(storage)

    if args.ciks:
        ciks = [c.zfill(10) for c in args.ciks]
    elif args.all or args.limit_ciks > 0:
        df = registry.load()
        ciks = sorted(df["cik"].astype(str).tolist())
        if args.limit_ciks > 0:
            ciks = ciks[: args.limit_ciks]
    else:
        ap.error("specify --ciks <list> or --all or --limit-ciks N")

    print(f"=== EDGAR full ingest: {len(ciks)} CIKs, concurrency={args.concurrency} ===")
    print(f"modalities: "
          f"financials={'yes' if not args.skip_financials else 'no'}  "
          f"form4={'yes' if not args.skip_form4 else 'no'}  "
          f"text={'yes' if not args.skip_text else 'no'}")
    print(f"max filings per modality per CIK: "
          f"{args.max_filings_per_cik if args.max_filings_per_cik else 'unlimited'}\n")

    sem = asyncio.Semaphore(args.concurrency)

    async with edgar_client() as client:
        async def _one(cik: str) -> CikStats:
            async with sem:
                s = await _process_cik(client, storage, pit, wm, cik, args)
                tag = "OK" if not s.errors else f"ERR×{len(s.errors)}"
                print(
                    f"  [{tag}] cik={s.cik}  "
                    f"fin={s.financials_filings}/{s.financial_rows} "
                    f"f4={s.form4_filings}/{s.form4_trades} "
                    f"txt={s.text_filings}/{s.text_chunks} "
                    f"in {s.elapsed_s:.1f}s"
                )
                if s.errors:
                    for e in s.errors[:3]:
                        print(f"      → {e}")
                return s

        all_stats = await asyncio.gather(*(_one(c) for c in ciks))

    # Summary
    total_fin_filings = sum(s.financials_filings for s in all_stats)
    total_fin_rows = sum(s.financial_rows for s in all_stats)
    total_f4_filings = sum(s.form4_filings for s in all_stats)
    total_f4_trades = sum(s.form4_trades for s in all_stats)
    total_text_filings = sum(s.text_filings for s in all_stats)
    total_text_chunks = sum(s.text_chunks for s in all_stats)
    total_errors = sum(len(s.errors) for s in all_stats)

    print("\n=== summary ===")
    print(f"  CIKs processed:          {len(all_stats)}")
    print(f"  financials filings/rows: {total_fin_filings:,} / {total_fin_rows:,}")
    print(f"  form4 filings/trades:    {total_f4_filings:,} / {total_f4_trades:,}")
    print(f"  text filings/chunks:     {total_text_filings:,} / {total_text_chunks:,}")
    print(f"  errors:                  {total_errors}")


if __name__ == "__main__":
    asyncio.run(amain())
