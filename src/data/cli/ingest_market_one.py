"""Stage M1 driver: ingest daily market bars for one CIK via yfinance.

Run:
    uv run python -m data.cli.ingest_market_one --cik 0000320193 --start 2020-01-01
"""
from __future__ import annotations

import argparse
import os
from datetime import date as date_cls

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.market_bar import MarketBar
from data.schemas.pit import Modality
from data.sources.edgar.archive import log_parse
from data.sources.market.yahoo import fetch_history
from data.storage import get_storage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cik", required=True)
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=date_cls.today().isoformat())
    ap.add_argument("--interval", default="1d", choices=("1d", "1wk", "1mo"))
    args = ap.parse_args()

    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    registry = EntityRegistry(storage)

    # Resolve CIK → ticker via registry
    reg = registry.load()
    row = reg[reg["cik"] == args.cik.zfill(10)]
    if row.empty:
        raise SystemExit(f"CIK {args.cik} not in registry")
    ticker = str(row.iloc[0]["ticker"] or "").upper()
    if not ticker:
        raise SystemExit(f"CIK {args.cik} has no ticker (registry must be seeded)")
    entity_id = EntityRegistry.entity_id_from_cik(args.cik)
    print(f"ingesting market bars for {ticker} (CIK {args.cik}, entity_id={entity_id})")

    bars = list(fetch_history(
        ticker,
        start=date_cls.fromisoformat(args.start),
        end=date_cls.fromisoformat(args.end),
        interval=args.interval,
    ))
    if not bars:
        log_parse(storage, args.cik.zfill(10), f"{ticker}:{args.interval}",
                  "market", "ok_no_records")
        print(f"  no bars returned — ticker may be delisted or invalid")
        return

    freq_label = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}[args.interval]
    rows = [
        {
            "entity_id": entity_id,
            "modality": Modality.MARKET.value,
            "effective_date": b.bar_date,
            "availability_date": b.bar_date,  # end-of-day prices knowable same day
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
    df = pd.DataFrame(rows)
    pit.write(Modality.MARKET, df, partition_keys=MarketBar.PARTITION_KEYS)
    log_parse(storage, args.cik.zfill(10), f"{ticker}:{args.interval}",
              "market", "ok_with_records", n_records=len(rows))
    print(f"  wrote {len(rows):,} {freq_label} bars; "
          f"date range {bars[0].bar_date} to {bars[-1].bar_date}")


if __name__ == "__main__":
    main()
