"""Form 4 amendment supersession.

When a Form 4/A is ingested, all canonical insider_trades rows from the same
(entity, insider_cik, period_of_report) filed *earlier* (lower accession or
earlier availability_date) should be marked as `restated_at = amendment.filed`
so PIT queries after the amendment ignore them.

Why this is needed: a Form 4/A can change the number, indices, or contents
of transactions relative to the original. Partition-key dedup alone (which
keeps the last-filed row per partition key) won't supersede a transaction
that was removed entirely in the amendment — that row has no successor with
the same partition key, so it lingers in canonical visible forever after.

Idempotent: re-running over the same amendment is a no-op (rows already
restated stay restated; their restated_at doesn't move backward).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data.pit.engine import _entity_path
from data.schemas.pit import Modality
from data.storage import Storage


def supersede_prior_form4(
    storage: Storage,
    entity_id: str,
    insider_cik: str,
    period_of_report: date,
    amendment_accession: str,
    amendment_filed: date,
) -> int:
    """Mark prior rows as restated. Returns number of rows updated.

    No-op if there's no canonical file yet, or if no rows match the
    (insider_cik, period_of_report) selector with an earlier accession.
    """
    path = _entity_path(Modality.INSIDER_TRADES, entity_id)
    if not storage.exists(path):
        return 0

    df = storage.read_parquet(path)
    if df.empty:
        return 0

    # The supersession candidates: same insider + period, NOT the amendment
    # itself (different accession). Earlier filings only — `availability_date
    # < amendment_filed` is the strict ordering guard. Use Timestamps for
    # comparisons so date-vs-datetime64 mismatches don't bite.
    period_ts = pd.Timestamp(period_of_report)
    amendment_ts = pd.Timestamp(amendment_filed)
    avail = pd.to_datetime(df["availability_date"], errors="coerce")
    period_col = pd.to_datetime(df["period_of_report"], errors="coerce")
    sel = (
        (df["insider_cik"] == insider_cik)
        & (period_col == period_ts)
        & (df["accession"] != amendment_accession)
        & (avail < amendment_ts)
    )

    n = int(sel.sum())
    if n == 0:
        return 0

    # Only set restated_at if it wasn't already set to an earlier (or equal)
    # date — re-running shouldn't move it backward.
    current_rs = pd.to_datetime(df["restated_at"], errors="coerce")
    needs_update = sel & (current_rs.isna() | (current_rs > amendment_ts))
    n_updated = int(needs_update.sum())
    if n_updated == 0:
        return 0

    df.loc[needs_update, "restated_at"] = amendment_filed
    storage.write_parquet(path, df)
    return n_updated
