"""Watermark store for incremental EDGAR ingest.

Each (cik, source) pair carries a watermark: the latest filing we have
already processed. On the next ingest pass we ask SEC for the submissions
index and skip anything at or below the watermark.

The watermark is the *filing date* (not effective date) — we only re-process
a filing if a strictly newer one has appeared since last run. Re-ingesting
the same filing yields the same canonical rows anyway (PITEngine.write dedups),
but skipping it saves rate-limit budget on the full-universe path.

Sources we track separately because each has its own concept of "newer":
    sec_edgar:submissions   — last filing seen in the index
    sec_edgar:xbrl          — last XBRL instance ingested
    sec_edgar:form4         — last Form 4 ingested
    sec_edgar:filing_html   — last primary HTML ingested

### Storage layout

One parquet per CIK at `state/edgar/{cik10}.parquet`, each holding a small
table indexed by `source`. This eliminates the read-modify-write race that
existed when all CIKs shared one parquet file: concurrent writers for
different CIKs touch different files. Writers for the *same* CIK across
sources serialize through `_locks[cik]`.

`list_all()` is for orchestration: glob the directory and concatenate. At
~10K CIK files of ~1KB each, this is well within parquet read overhead.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable, Optional

import pandas as pd

from data.storage import Storage

STATE_DIR = "state/edgar"
_COLS = ["cik", "source", "last_accn", "last_filed_date", "last_ingest_at"]

# A watermark file is `state/edgar/{cik10}.parquet` — a 10-digit name. Other
# parquets in the same directory (archive logs) have suffixes after the CIK
# and must NOT be picked up by list_all().
import re

_WATERMARK_FILE_RE = re.compile(r"^\d{10}\.parquet$")


def _cik_path(cik: str) -> str:
    return f"{STATE_DIR}/{cik.zfill(10)}.parquet"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Watermark:
    cik: str
    source: str
    last_accn: Optional[str]
    last_filed_date: Optional[date]
    last_ingest_at: datetime


class WatermarkStore:
    """Per-CIK watermark store. One file per entity, asyncio-locked per CIK.

    The lock prevents intra-process races when multiple modalities ingest the
    same CIK concurrently (e.g. Stage 5 dispatches financials + Form 4 for
    AAPL in parallel). Cross-process safety requires fcntl/flock and is out
    of scope for the current single-process design.
    """

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self._locks: dict[str, asyncio.Lock] = {}

    # ----- internal helpers -----------------------------------------------

    def _lock_for(self, cik: str) -> asyncio.Lock:
        cik10 = cik.zfill(10)
        if cik10 not in self._locks:
            self._locks[cik10] = asyncio.Lock()
        return self._locks[cik10]

    def _read_one(self, cik: str) -> pd.DataFrame:
        path = _cik_path(cik)
        if not self.storage.exists(path):
            return pd.DataFrame(columns=_COLS)
        df = self.storage.read_parquet(path)
        df["last_filed_date"] = pd.to_datetime(
            df["last_filed_date"], errors="coerce"
        ).dt.date
        return df

    def _write_one(self, cik: str, df: pd.DataFrame) -> None:
        self.storage.write_parquet(_cik_path(cik), df)

    # ----- public API (sync) ----------------------------------------------

    def get(self, cik: str, source: str) -> Optional[Watermark]:
        df = self._read_one(cik)
        if df.empty:
            return None
        m = df["source"] == source
        if not m.any():
            return None
        r = df[m].iloc[-1]
        return Watermark(
            cik=r["cik"],
            source=r["source"],
            last_accn=r["last_accn"] or None,
            last_filed_date=(r["last_filed_date"] if pd.notna(r["last_filed_date"]) else None),
            last_ingest_at=pd.to_datetime(r["last_ingest_at"]).to_pydatetime(),
        )

    def set(
        self,
        cik: str,
        source: str,
        last_accn: Optional[str],
        last_filed_date: Optional[date],
    ) -> None:
        """Synchronous setter. Use `set_async` from coroutines that share the
        CIK across multiple sources."""
        self._merge_and_write(cik, [(source, last_accn, last_filed_date)])

    async def set_async(
        self,
        cik: str,
        source: str,
        last_accn: Optional[str],
        last_filed_date: Optional[date],
    ) -> None:
        async with self._lock_for(cik):
            # Sync write inside the lock — pandas/parquet are blocking and
            # fast enough at this size that releasing the loop is unnecessary.
            self._merge_and_write(cik, [(source, last_accn, last_filed_date)])

    def _merge_and_write(
        self,
        cik: str,
        rows: list[tuple[str, Optional[str], Optional[date]]],
    ) -> None:
        df = self._read_one(cik)
        now = _now_utc()
        updates = pd.DataFrame(
            [
                {
                    "cik": cik.zfill(10),
                    "source": source,
                    "last_accn": last_accn,
                    "last_filed_date": last_filed_date,
                    "last_ingest_at": now,
                }
                for source, last_accn, last_filed_date in rows
            ],
            columns=_COLS,
        )
        combined = updates if df.empty else pd.concat([df, updates], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["cik", "source"], keep="last"
        ).reset_index(drop=True)
        self._write_one(cik, combined)

    def filter_new(
        self,
        cik: str,
        source: str,
        candidates: list[tuple[str, date]],
    ) -> list[tuple[str, date]]:
        """Given (accn, filed_date) pairs, return those strictly newer than
        the current watermark. Preserves input order.

        Filing dates are the right key because SEC's accession-number
        ordering can be ambiguous across filers / quarters; filed_date is
        monotone for a given CIK by construction.
        """
        wm = self.get(cik, source)
        if wm is None or wm.last_filed_date is None:
            return candidates
        return [(a, d) for a, d in candidates if d > wm.last_filed_date]

    def list_all(self) -> pd.DataFrame:
        """Return every watermark in the store, one row per (cik, source).

        Used by orchestration code that wants a global view (e.g. "find the
        100 oldest watermarks for prioritized re-ingest"). Reads every per-CIK
        file; at 10K CIKs × ~1KB each this is sub-second.
        """
        frames: list[pd.DataFrame] = []
        for p in self.storage.list(STATE_DIR):
            fname = p.rsplit("/", 1)[-1]
            if _WATERMARK_FILE_RE.match(fname):
                frames.append(self.storage.read_parquet(p))
        if not frames:
            return pd.DataFrame(columns=_COLS)
        return pd.concat(frames, ignore_index=True)
