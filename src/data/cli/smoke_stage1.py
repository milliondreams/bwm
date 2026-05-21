"""Stage 1 end-to-end smoke test.

Hits SEC's real submissions endpoint for one company, archives the body
content-addressed, writes a watermark, and verifies idempotency on re-run.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime

from data.sources.edgar.archive import archive
from data.sources.edgar.client import edgar_client
from data.sources.edgar.state import WatermarkStore
from data.storage import get_storage

APPLE_CIK = "0000320193"
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{APPLE_CIK}.json"


async def amain() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    wmstore = WatermarkStore(storage)

    print("=== Stage 1 smoke: client + archive + watermarks ===")

    async with edgar_client() as client:
        t0 = time.monotonic()
        body = await client.get_bytes(SUBMISSIONS_URL)
        t1 = time.monotonic()
    print(f"  fetched {len(body):,} bytes in {(t1 - t0)*1000:.0f} ms")

    payload = json.loads(body)
    filings = payload["filings"]["recent"]
    n = len(filings["accessionNumber"])
    print(f"  parsed {n} recent filings (most recent: {filings['form'][0]} on {filings['filingDate'][0]})")

    path1, new1 = archive(storage, APPLE_CIK, "submissions", body)
    print(f"  archive write #1: {path1}  (new={new1})")

    path2, new2 = archive(storage, APPLE_CIK, "submissions", body)
    print(f"  archive write #2: {path2}  (new={new2})  ← should be False")
    assert path1 == path2
    assert new2 is False, "second archive write should be a no-op (idempotent)"

    # Watermark roundtrip: set, get, filter_new
    last_accn = filings["accessionNumber"][0]
    last_filed = datetime.strptime(filings["filingDate"][0], "%Y-%m-%d").date()
    wmstore.set(APPLE_CIK, "sec_edgar:submissions", last_accn, last_filed)
    wm = wmstore.get(APPLE_CIK, "sec_edgar:submissions")
    print(f"  watermark stored: cik={wm.cik}  accn={wm.last_accn}  filed={wm.last_filed_date}")
    assert wm is not None and wm.last_accn == last_accn

    # filter_new: pretend we have the same set of candidates. None should be "new".
    candidates = [
        (filings["accessionNumber"][i],
         datetime.strptime(filings["filingDate"][i], "%Y-%m-%d").date())
        for i in range(min(5, n))
    ]
    fresh = wmstore.filter_new(APPLE_CIK, "sec_edgar:submissions", candidates)
    print(f"  filter_new on already-seen: {len(fresh)} new (expected 0)")
    assert len(fresh) == 0

    # Synthetic future filing → should be flagged new
    from datetime import date, timedelta
    future = [("0000000000-99-999999", date.today() + timedelta(days=365))]
    fresh = wmstore.filter_new(APPLE_CIK, "sec_edgar:submissions", future)
    print(f"  filter_new on synthetic future filing: {len(fresh)} new (expected 1)")
    assert len(fresh) == 1

    print("\n=== Stage 1 PASS ===")


if __name__ == "__main__":
    asyncio.run(amain())
