"""Stage 3 driver: ingest Form 4 filings for one CIK.

Lists Form 4 / Form 4/A filings via the submissions API, fetches each
ownership XML, parses transactions, writes to canonical insider_trades.

Run:
    uv run python -m data.cli.ingest_form4_one --cik 0000320193 --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import os
from datetime import date as date_cls

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.insider_trade import InsiderTrade
from data.schemas.pit import Modality
from data.sources.edgar.archive import archive, log_parse
from data.sources.edgar.client import edgar_client
from data.sources.edgar.form4 import parse_form4_filing
from data.sources.edgar.form4_supersede import supersede_prior_form4
from data.sources.edgar.state import WatermarkStore
from data.sources.edgar.submissions import FORM4_FORMS, Filing, list_filings


def _to_pit_rows(filing: Filing, entity_id: str, parsed) -> pd.DataFrame:
    rows = []
    for idx, t in enumerate(parsed.trades):
        rows.append({
            "transaction_index": idx,
            "period_of_report": parsed.period_of_report,
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
            "form": filing.form,
            "accession": filing.accession,
        })
    return pd.DataFrame(rows)


async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cik", required=True)
    ap.add_argument("--limit", type=int, default=0, help="ingest at most N filings (0 = all)")
    ap.add_argument("--since", type=str, default="", help="ISO date; only ingest filings on/after")
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    from data.storage import get_storage
    storage = get_storage()
    pit = PITEngine(storage)
    wm = WatermarkStore(storage)
    entity_id = EntityRegistry.entity_id_from_cik(args.cik)
    since = date_cls.fromisoformat(args.since) if args.since else None

    async with edgar_client() as client:
        print(f"listing Form 4 filings for CIK {args.cik}…")
        filings = await list_filings(client, args.cik, forms=FORM4_FORMS)
        print(f"  {len(filings)} Form 4 filings in recent index")

        # Watermark filter — skip already-seen
        candidates = [(f.accession, f.filed) for f in filings]
        fresh_keys = set(
            a for a, _ in wm.filter_new(args.cik, "sec_edgar:form4", candidates)
        )
        fresh = [f for f in filings if f.accession in fresh_keys]
        if since is not None:
            fresh = [f for f in fresh if f.filed >= since]
        if args.limit > 0:
            fresh = fresh[: args.limit]
        print(f"  {len(fresh)} new (after watermark / since-filter / limit)")

        total_trades = 0
        for filing in fresh:
            cik10 = args.cik.zfill(10)
            try:
                body = await client.get_bytes(filing.primary_doc_raw_url)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {filing.accession}: {type(e).__name__}: {e}")
                log_parse(storage, cik10, filing.accession, "insider_trades", "fetch_failed",
                          error_message=f"{type(e).__name__}: {e}")
                continue
            archive(storage, cik10, "form4", body, accession=filing.accession)
            parsed = parse_form4_filing(body)
            if parsed is None:
                log_parse(storage, cik10, filing.accession, "insider_trades", "parse_failed")
                continue
            if not parsed.trades:
                log_parse(storage, cik10, filing.accession, "insider_trades", "ok_no_records")
                continue
            df = _to_pit_rows(filing, entity_id, parsed)
            if df.empty:
                log_parse(storage, cik10, filing.accession, "insider_trades", "ok_no_records")
                continue
            pit.write(
                Modality.INSIDER_TRADES,
                df,
                partition_keys=InsiderTrade.PARTITION_KEYS,
            )
            # Amendment supersession: a Form 4/A marks all prior rows from
            # the same (insider, period_of_report) as restated.
            if filing.form.endswith("/A") and parsed.period_of_report is not None:
                n_superseded = supersede_prior_form4(
                    storage=storage,
                    entity_id=entity_id,
                    insider_cik=parsed.trades[0].insider_cik,
                    period_of_report=parsed.period_of_report,
                    amendment_accession=filing.accession,
                    amendment_filed=filing.filed,
                )
                if n_superseded:
                    print(f"    (superseded {n_superseded} prior rows for this period)")
            total_trades += len(parsed.trades)
            log_parse(storage, cik10, filing.accession, "insider_trades",
                      "ok_with_records", n_records=len(parsed.trades))
            print(
                f"  + {filing.accession}: {len(parsed.trades)} trade(s)  "
                f"insider={parsed.trades[0].insider_name}  filed={filing.filed}"
            )

        # Bump watermark to the most-recent filed date we actually processed.
        if fresh:
            latest = max(fresh, key=lambda f: f.filed)
            await wm.set_async(
                args.cik, "sec_edgar:form4", latest.accession, latest.filed
            )

        print(f"\nwrote {total_trades} insider trades to canonical")


if __name__ == "__main__":
    asyncio.run(amain())
