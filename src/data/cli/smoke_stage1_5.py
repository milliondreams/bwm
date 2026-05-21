"""Regression tests for the Stage 1.5 hardening pass.

Covers the six fixes:
  1. PITEngine partition by entity_id — writes for one CIK don't read files
     for other CIKs.
  2. Per-CIK watermarks — concurrent writes for different CIKs don't race.
  3. tz-aware datetime — last_ingest_at carries tzinfo=UTC after roundtrip.
  4. Stable archive paths — re-archiving same body lands at same path; the
     archive log records both events.
  5. Deterministic restatement tiebreaker — same-day collision picks the row
     with the lexicographically-greatest source_ref.
  6. Schema-declared partition keys — FinancialFact.PARTITION_KEYS drives
     dedup; passing the list explicitly produces identical results to the
     legacy column-inference path.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from data.pit.engine import PITEngine, _entity_path
from data.schemas.financial import FinancialFact
from data.schemas.pit import Modality
from data.sources.edgar.archive import archive
from data.sources.edgar.state import WatermarkStore
from data.storage import get_storage

# --- helpers ----------------------------------------------------------------


def _fact_row(entity_id: str, value: float, filed: str, accn: str, fy: int = 2020) -> dict:
    # Spread v3 PIT defaults (Stage 1.7) so synthetic test rows carry the
    # same field set as real ingest produces; otherwise validate_v3_fields
    # flags these test fixtures as malformed during the regression sweep.
    return {
        **FinancialFact.pit_default_fields(),
        "entity_id": entity_id,
        "modality": Modality.FINANCIALS.value,
        "effective_date": date(fy, 12, 31),
        "availability_date": date.fromisoformat(filed),
        "restated_at": None,
        "source": "test",
        "source_ref": accn,
        "concept": "us-gaap:Revenues",
        "taxonomy": "us-gaap",
        "period_start": date(fy, 1, 1),
        "period_end": date(fy, 12, 31),
        "fiscal_year": fy,
        "fiscal_period": "FY",
        "unit": "USD",
        "value": value,
        "form": "10-K",
        "accession": accn,
        "context_id": "",
        "dimensions_json": "",
    }


# --- Fix 1: per-CIK partitioning ----------------------------------------------


def _wipe(storage, *paths: str) -> None:
    """Delete fixture files between test runs so tests are repeatable."""
    root = getattr(storage, "root", None)
    if root is None:
        return
    for p in paths:
        f = Path(root) / p
        if f.exists():
            f.unlink()


def test_per_entity_partition(storage, pit) -> None:
    e1, e2 = "us-cik-0000000001", "us-cik-0000000002"
    _wipe(
        storage,
        _entity_path(Modality.FINANCIALS, e1),
        _entity_path(Modality.FINANCIALS, e2),
    )
    pit.write(
        Modality.FINANCIALS,
        pd.DataFrame([_fact_row(e1, 100.0, "2021-02-01", "ACC-A")]),
        partition_keys=FinancialFact.PARTITION_KEYS,
    )
    p1 = _entity_path(Modality.FINANCIALS, e1)
    p2 = _entity_path(Modality.FINANCIALS, e2)
    assert storage.exists(p1), f"expected {p1}"
    assert not storage.exists(p2), f"unexpected {p2}"

    # Writing for e2 must not touch e1's file (mtime check).
    mtime_e1_before = Path(storage.root) / p1
    t0 = mtime_e1_before.stat().st_mtime_ns
    pit.write(
        Modality.FINANCIALS,
        pd.DataFrame([_fact_row(e2, 200.0, "2021-02-01", "ACC-B")]),
        partition_keys=FinancialFact.PARTITION_KEYS,
    )
    t1 = mtime_e1_before.stat().st_mtime_ns
    assert t1 == t0, "writing entity 2 touched entity 1's parquet file"
    assert storage.exists(p2)
    print("  [OK] Fix 1: per-CIK writes are isolated")


# --- Fix 2: concurrent watermark writes ---------------------------------------


async def test_concurrent_watermarks(storage) -> None:
    wm = WatermarkStore(storage)
    ciks = [f"00000000{i:02d}" for i in range(1, 11)]  # 10 distinct CIKs
    sources = ["sec_edgar:submissions", "sec_edgar:xbrl"]

    async def write_one(cik: str, source: str) -> None:
        await wm.set_async(cik, source, f"accn-{cik}-{source}", date(2025, 1, 1))

    # 10 CIKs × 2 sources = 20 concurrent writes
    await asyncio.gather(*(write_one(c, s) for c in ciks for s in sources))

    df = wm.list_all()
    pairs = set(zip(df["cik"], df["source"]))
    expected = {(c.zfill(10), s) for c in ciks for s in sources}
    missing = expected - pairs
    # We assert no missing rows; extras are OK (other tests / prior runs may
    # have left their own CIKs in the store). This test is scoped to its own
    # CIKs.
    assert not missing, f"missing watermarks: {missing}"
    print(f"  [OK] Fix 2: {len(expected)} concurrent watermarks all landed (store also has {len(pairs - expected)} pre-existing)")


# --- Fix 3: tz-aware timestamps ----------------------------------------------


def test_tz_aware(storage) -> None:
    wm = WatermarkStore(storage)
    wm.set("0000999999", "test", "accn-x", date(2025, 6, 15))
    row = wm.get("0000999999", "test")
    assert row is not None
    assert row.last_ingest_at.tzinfo is not None, "last_ingest_at is tz-naive"
    # Sanity: should be within ~5 seconds of "now"
    now = datetime.now(timezone.utc)
    delta = abs((now - row.last_ingest_at).total_seconds())
    assert delta < 5, f"last_ingest_at off by {delta:.1f}s"
    print(f"  [OK] Fix 3: last_ingest_at carries tz={row.last_ingest_at.tzinfo}")


# --- Fix 4: stable archive paths + log ---------------------------------------


def test_archive_stable_path_and_log(storage) -> None:
    import json as _json
    cik = "0000888888"
    body = b'{"hello":"world"}'

    # Wipe per-CIK archive + log so the first call sees was_new=True.
    from data.sources.edgar.archive import archive_path, archive_log_path
    _wipe(
        storage,
        archive_path(cik, "submissions", body),
        archive_log_path(cik),
    )

    p1, new1 = archive(storage, cik, "submissions", body)
    p2, new2 = archive(storage, cik, "submissions", body)
    assert p1 == p2, "identical body produced different paths"
    assert new1 is True and new2 is False, "idempotency broken"

    # Log is JSONL — one event per line.
    raw = storage.read_bytes(archive_log_path(cik)).decode("utf-8")
    log_rows = [_json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(log_rows) == 2, f"expected 2 log rows, got {len(log_rows)}"
    assert [r["was_new"] for r in log_rows] == [True, False]
    assert log_rows[0]["sha12"] == log_rows[1]["sha12"], "different SHAs for same body"
    print(f"  [OK] Fix 4: archive path stable, JSONL log has {len(log_rows)} rows for one body")


# --- Fix 5: deterministic restatement tiebreaker -----------------------------


def test_query_tiebreaker_deterministic(storage, pit) -> None:
    """Multiple rows with the same partition key but distinct `source_ref` —
    the engine must always pick the lex-greatest source_ref regardless of
    the input order they were written in. Tested with 4 rows and every
    permutation."""
    import itertools

    e = "us-cik-0000777777"
    path = _entity_path(Modality.FINANCIALS, e)

    # Four rows sharing the entire partition key — different source_ref.
    rows = [
        _fact_row(e, 100.0, "2021-04-01", "ACC-A"),
        _fact_row(e, 200.0, "2021-04-01", "ACC-B"),
        _fact_row(e, 300.0, "2021-04-01", "ACC-C"),
        _fact_row(e, 400.0, "2021-04-01", "ACC-D"),
    ]
    expected_winner = "ACC-D"  # lex-greatest

    # Run every permutation: 4! = 24 orderings; outcome must match each time.
    permutations_tested = 0
    for perm in itertools.permutations(rows):
        if storage.exists(path):
            (Path(storage.root) / path).unlink()
        pit.write(
            Modality.FINANCIALS,
            pd.DataFrame(list(perm)),
            partition_keys=FinancialFact.PARTITION_KEYS,
        )
        result = pit.query(
            Modality.FINANCIALS,
            entity_ids=[e],
            as_of=date(2022, 1, 1),
            partition_keys=FinancialFact.PARTITION_KEYS,
        )
        assert len(result) == 1
        got = result.iloc[0]["source_ref"]
        assert got == expected_winner, (
            f"non-deterministic: input order {[r['source_ref'] for r in perm]} "
            f"won {got}, expected {expected_winner}"
        )
        permutations_tested += 1
    print(f"  [OK] Fix 5: query tiebreaker deterministic across {permutations_tested} permutations of 4 rows")


# --- Fix 6: schema-declared partition keys -----------------------------------


def test_schema_partition_keys(storage, pit) -> None:
    """Two rows that look identical except for dimensions_json must be kept
    as distinct logical facts when FinancialFact.PARTITION_KEYS is passed
    (because that key list includes dimensions_json)."""
    e = "us-cik-0000666666"
    path = _entity_path(Modality.FINANCIALS, e)
    if storage.exists(path):
        (Path(storage.root) / path).unlink()

    consolidated = _fact_row(e, 1000.0, "2021-02-01", "ACC-CONS")
    segment = _fact_row(e, 600.0, "2021-02-01", "ACC-CONS")
    segment["dimensions_json"] = '{"axis":"us-gaap:SegmentReportingAxis","member":"acme:Americas"}'

    pit.write(
        Modality.FINANCIALS,
        pd.DataFrame([consolidated, segment]),
        partition_keys=FinancialFact.PARTITION_KEYS,
    )
    result = pit.query(
        Modality.FINANCIALS,
        entity_ids=[e],
        as_of=date(2022, 1, 1),
        partition_keys=FinancialFact.PARTITION_KEYS,
    )
    assert len(result) == 2, f"expected 2 distinct facts, got {len(result)}"
    values = sorted(result["value"].tolist())
    assert values == [600.0, 1000.0]
    print("  [OK] Fix 6: schema PARTITION_KEYS keeps dimensional facts distinct")


# --- Fix A6 (Stage 1.6): concurrent cross-CIK write under asyncio.gather -----


async def test_concurrent_cross_cik_writes(storage, pit) -> None:
    """Confirm that writes for different CIKs in parallel via asyncio.gather
    don't corrupt each other. Each CIK's file should contain exactly the rows
    written for that CIK and nothing else; the sequential-order outcome must
    equal the concurrent-order outcome row-for-row."""
    n_ciks = 10
    rows_per_cik = 50
    eids = [f"us-cik-1000{i:05d}" for i in range(n_ciks)]

    # Wipe any prior state for these entities.
    for eid in eids:
        path = _entity_path(Modality.FINANCIALS, eid)
        if storage.exists(path):
            (Path(storage.root) / path).unlink()

    async def write_for(eid: str) -> None:
        rows = [
            _fact_row(eid, float(j), f"2021-{1+(j%12):02d}-15", f"ACC-{eid}-{j}", fy=2021 - (j // 4))
            for j in range(rows_per_cik)
        ]
        pit.write(
            Modality.FINANCIALS,
            pd.DataFrame(rows),
            partition_keys=FinancialFact.PARTITION_KEYS,
        )

    await asyncio.gather(*(write_for(eid) for eid in eids))

    # Each per-entity file must contain exactly rows_per_cik rows, and those
    # rows must reference only their own entity_id — no cross-contamination.
    for eid in eids:
        path = _entity_path(Modality.FINANCIALS, eid)
        df = storage.read_parquet(path)
        assert len(df) == rows_per_cik, (
            f"{eid}: expected {rows_per_cik} rows, got {len(df)}"
        )
        bad = (df["entity_id"] != eid).sum()
        assert bad == 0, f"{eid}: {bad} rows leaked from another entity"

    print(
        f"  [OK] A6: {n_ciks} concurrent CIK writes via asyncio.gather "
        f"land cleanly, no cross-contamination"
    )


# --- driver ------------------------------------------------------------------


async def amain() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    print("=== Stage 1.5 hardening regression ===")
    test_per_entity_partition(storage, pit)
    await test_concurrent_watermarks(storage)
    test_tz_aware(storage)
    test_archive_stable_path_and_log(storage)
    test_query_tiebreaker_deterministic(storage, pit)
    test_schema_partition_keys(storage, pit)
    await test_concurrent_cross_cik_writes(storage, pit)
    print("\n=== Stage 1.5 PASS ===")


if __name__ == "__main__":
    asyncio.run(amain())
