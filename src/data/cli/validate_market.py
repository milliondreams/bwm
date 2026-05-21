"""Stage M1 validation: market data ingest is correct.

Asserts:
  - AAPL has bars for 2020 with sensible OHLCV
  - PIT contract: as-of before the bar date returns nothing for that date;
    as-of on the bar date includes it
  - Adjusted close on a known historical date is in the expected range
    (Apple's adjusted close on 2020-03-23 — the COVID-19 low — was ~$56-57
    after the 2020 4-for-1 split adjustment)
"""
from __future__ import annotations

import os
from datetime import date

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.schemas.market_bar import MarketBar
from data.schemas.pit import Modality
from data.storage import get_storage


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)
    eid = EntityRegistry.entity_id_from_cik("0000320193")

    print("=== Stage M1 validation: market data ===\n")

    full = pit.snapshot(eid, as_of=date(2030, 1, 1))
    assert not full.market.empty, "no market bars ingested for AAPL"
    n = len(full.market)
    print(f"AAPL market bars in canonical: {n:,}")

    # Sanity on OHLC relationships
    bad_ohlc = (
        (full.market["high"] < full.market["low"])
        | (full.market["close"] > full.market["high"])
        | (full.market["close"] < full.market["low"])
    ).sum()
    assert bad_ohlc == 0, f"{bad_ohlc} bars violate OHLC ordering"
    print("  [OK] OHLC ordering valid on every bar")

    # PIT contract: bar at 2020-03-23 visible by snapshot at 2020-03-23, not 2020-03-22
    pre_snap = pit.snapshot(eid, as_of=date(2020, 3, 22))
    has_2020_03_23_pre = (
        not pre_snap.market.empty
        and (pre_snap.market["effective_date"] == date(2020, 3, 23)).any()
    )
    assert not has_2020_03_23_pre, "2020-03-23 bar visible in 2020-03-22 snapshot"

    on_snap = pit.snapshot(eid, as_of=date(2020, 3, 23))
    on_bar = on_snap.market[on_snap.market["effective_date"] == date(2020, 3, 23)]
    assert not on_bar.empty, "2020-03-23 bar missing from same-day snapshot"
    adj_close = float(on_bar.iloc[0]["adj_close"])
    print(f"  [OK] PIT: 2020-03-23 bar excluded on 2020-03-22, included on 2020-03-23")
    print(f"  AAPL 2020-03-23 adj_close = ${adj_close:.2f} (COVID-19 low)")
    # Apple's adj_close on 2020-03-23 should be in the $55-60 range
    # (after 4-for-1 split that happened later in 2020). yfinance backdates the
    # split adjustment so historical pre-split prices show as ~$56 post-adjustment.
    assert 50 <= adj_close <= 65, (
        f"AAPL 2020-03-23 adj_close ${adj_close:.2f} outside expected $50-65 range"
    )

    print("\n=== Stage M1 validation: PASS ===")


if __name__ == "__main__":
    main()
