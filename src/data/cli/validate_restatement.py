"""Stress-test the PIT engine against a real, well-documented restatement.

Target: General Electric (CIK 0000040545). In Feb 2018 GE refiled prior-year
financials following the long-term care insurance reserve issue, restating
2016 and 2017 figures. We expect to see, in the raw EDGAR companyfacts data,
multiple rows for the same (concept, fiscal_year, fiscal_period) with different
`filed` dates — and the PIT engine should return *different* values depending
on the as-of date.

This is the test that proves FR-2 (point-in-time integrity) actually works,
not just that it compiles. A naive backtester that does `WHERE filed <= as_of`
without restatement handling will silently use restated numbers for dates
where only the original numbers were knowable, and every accuracy figure
that comes out of such a backtest is fictional.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit import PITEngine
from data.schemas import Modality
from data.sources.edgar import ingest_company
from data.storage import get_storage

GE_CIK = "0000040545"
USER_AGENT = "norn-bwm research rohit@dragonscale.ai"


def find_restated_concepts(
    df: pd.DataFrame, top_n: int = 10, min_days_between: int = 90
) -> pd.DataFrame:
    """Find true cross-filing restatements.

    A real restatement is a *later filing* that supersedes an earlier one for
    the same (concept, fiscal_year, fiscal_period). Same-filing collisions
    (multiple XBRL contexts collapsing into one logical row in our flattened
    schema) appear as multiple rows with the same `filed` date — we exclude
    those by requiring ≥ min_days_between days between the first and last
    filings.
    """
    key = ["entity_id", "concept", "fiscal_year", "fiscal_period", "unit"]
    # Within a group, collapse same-day-multi-context rows to one (their max
    # absolute value — biggest reported magnitude). This isn't a fix for the
    # connector bug, but it stops same-day artifacts from masquerading here.
    df = (
        df.sort_values(key + ["availability_date", "value"])
          .drop_duplicates(subset=key + ["availability_date"], keep="last")
    )
    grouped = df.groupby(key)
    summary_rows = []
    for keys, g in grouped:
        if len(g) < 2:
            continue
        first = g.iloc[0]
        last = g.iloc[-1]
        days_between = (last["availability_date"] - first["availability_date"]).days
        if days_between < min_days_between:
            continue
        if first["value"] == last["value"]:
            continue
        summary_rows.append({
            "days_between": days_between,
            "concept": keys[1],
            "fiscal_year": keys[2],
            "fiscal_period": keys[3],
            "unit": keys[4],
            "n_filings": len(g),
            "orig_filed": first["availability_date"],
            "orig_value": first["value"],
            "latest_filed": last["availability_date"],
            "latest_value": last["value"],
            "delta": last["value"] - first["value"],
            "abs_delta": abs(last["value"] - first["value"]),
        })
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows).sort_values("abs_delta", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    registry = EntityRegistry(storage)
    pit = PITEngine(storage)

    # Ensure GE is ingested. ingest_company is idempotent because PITEngine.write
    # dedupes on (entity_id, effective_date, source_ref, concept, …) — re-running
    # just no-ops if the data is already present.
    print(f"Ingesting GE (CIK {GE_CIK}) — this is idempotent…")
    n = ingest_company(GE_CIK, USER_AGENT, pit, registry)
    print(f"  wrote {n} facts")

    entity_id = EntityRegistry.entity_id_from_cik(GE_CIK)
    from data.pit.engine import _entity_path
    raw = storage.read_parquet(_entity_path(Modality.FINANCIALS, entity_id))
    for col in ("effective_date", "availability_date", "restated_at"):
        raw[col] = pd.to_datetime(raw[col], errors="coerce").dt.date
    print(f"\nGE raw rows in store: {len(raw):,}")

    print("\n=== Top 10 restated (concept, fy, fp) groups by absolute change ===")
    restated = find_restated_concepts(raw, top_n=10)
    if restated.empty:
        print("  (no restatements found — unexpected for GE)")
        return
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(restated.to_string(index=False))

    # Pick the most material restatement and run the PIT contract against it.
    target = restated.iloc[0]
    concept = target["concept"]
    fy = int(target["fiscal_year"])
    fp = target["fiscal_period"]
    orig_filed = target["orig_filed"]
    latest_filed = target["latest_filed"]
    print(f"\n=== Stress test: {concept} FY{fy} {fp} ===")
    print(f"  original filed:  {orig_filed}  value = {target['orig_value']:,.0f}")
    print(f"  latest filed:    {latest_filed}  value = {target['latest_value']:,.0f}")
    print(f"  delta:           {target['delta']:+,.0f}  ({target['unit']})")

    # Three as-of probes
    from datetime import timedelta

    probes = [
        ("before original filing", orig_filed - timedelta(days=1), None),
        ("just after original filing", orig_filed + timedelta(days=1), target["orig_value"]),
        ("day before restatement filing", latest_filed - timedelta(days=1), target["orig_value"]),
        ("just after restatement filing", latest_filed + timedelta(days=1), target["latest_value"]),
        ("years after restatement", date(latest_filed.year + 3, 1, 1), target["latest_value"]),
    ]

    print("\n  PIT engine returns:")
    all_pass = True
    for label, as_of, expected in probes:
        if as_of is None:
            continue
        result = pit.query(
            Modality.FINANCIALS,
            entity_ids=[entity_id],
            as_of=as_of,
            extra_filters={
                "concept": concept,
                "fiscal_year": fy,
                "fiscal_period": fp,
                "unit": target["unit"],
            },
        )
        if result.empty:
            actual = None
            ok = (expected is None)
        else:
            actual = float(result.iloc[0]["value"])
            ok = (expected is not None and abs(actual - expected) < 1.0)
        status = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        actual_s = f"{actual:,.0f}" if actual is not None else "<not knowable>"
        expected_s = f"{expected:,.0f}" if expected is not None else "<not knowable>"
        print(f"  [{status}] as_of={as_of}  ({label:35s})  actual={actual_s:>20s}  expected={expected_s}")

    print(f"\n=== RESULT: {'PASS — PIT contract holds against real restatement' if all_pass else 'FAIL — engine returned wrong as-of value'} ===")

    # Stage 1.6 A7: multi-restatement timeline test.
    test_multi_restatement_timeline()


def test_multi_restatement_timeline() -> None:
    """A→B→C: three rows for the same logical fact filed at T1 < T2 < T3.
    Query the engine at T1.5, T2.5, T3.5 — must return A, B, C respectively.

    Synthesizes a fresh entity (us-cik-99...) and writes directly to canonical
    so this test isn't coupled to any real-world filing pattern. Confirms
    the engine handles cascading restatements correctly (a Phase A review
    reviewer flagged this as a potential bug; the test proves it isn't one)."""
    from datetime import date as date_cls
    from pathlib import Path
    import pandas as pd
    from data.pit.engine import PITEngine, _entity_path
    from data.schemas.financial import FinancialFact
    from data.schemas.pit import Modality
    from data.storage import get_storage

    print("\n=== Stage 1.6 A7: multi-restatement (A→B→C) timeline ===")
    storage = get_storage()
    pit = PITEngine(storage)
    eid = "us-cik-9999999991"
    path = _entity_path(Modality.FINANCIALS, eid)
    if storage.exists(path):
        (Path(storage.root) / path).unlink()

    base = {
        "entity_id": eid,
        "modality": Modality.FINANCIALS.value,
        "effective_date": date_cls(2020, 12, 31),
        "restated_at": None,
        "source": "test:multi-restatement",
        "concept": "us-gaap:Revenues",
        "taxonomy": "us-gaap",
        "period_start": date_cls(2020, 1, 1),
        "period_end": date_cls(2020, 12, 31),
        "fiscal_year": 2020,
        "fiscal_period": "FY",
        "unit": "USD",
        "form": "10-K",
        "context_id": "",
        "dimensions_json": "",
    }
    rows = pd.DataFrame([
        {**base, "availability_date": date_cls(2021, 2, 1), "value": 100.0, "source_ref": "ACC-A", "accession": "ACC-A"},
        {**base, "availability_date": date_cls(2021, 5, 1), "value": 110.0, "source_ref": "ACC-B", "accession": "ACC-B"},
        {**base, "availability_date": date_cls(2021, 8, 1), "value": 120.0, "source_ref": "ACC-C", "accession": "ACC-C"},
    ])
    pit.write(Modality.FINANCIALS, rows, partition_keys=FinancialFact.PARTITION_KEYS)

    probes = [
        ("between A and B", date_cls(2021, 3, 15), 100.0, "ACC-A"),
        ("between B and C", date_cls(2021, 6, 15), 110.0, "ACC-B"),
        ("after C", date_cls(2022, 1, 1), 120.0, "ACC-C"),
    ]
    all_pass = True
    for label, as_of, expected_value, expected_source in probes:
        result = pit.query(
            Modality.FINANCIALS,
            entity_ids=[eid],
            as_of=as_of,
            partition_keys=FinancialFact.PARTITION_KEYS,
        )
        assert len(result) == 1, f"{label}: expected 1 row, got {len(result)}"
        got_value = float(result.iloc[0]["value"])
        got_source = result.iloc[0]["source_ref"]
        ok = got_value == expected_value and got_source == expected_source
        print(
            f"  [{'OK' if ok else 'FAIL'}] as_of={as_of} ({label:>20s}): "
            f"value={got_value:>8.1f} src={got_source}  "
            f"expected={expected_value:>8.1f} src={expected_source}"
        )
        all_pass = all_pass and ok

    assert all_pass, "multi-restatement timeline contract broken"
    print("\n=== A7 multi-restatement timeline: PASS ===")


if __name__ == "__main__":
    main()
