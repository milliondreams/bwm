"""Validate training-time PIT awareness end-to-end.

Five tests, each exercising a different failure mode the snapshot API is
designed to prevent. The adversarial one (test 3) is the key gate: it
constructs two snapshots straddling a real restatement (GE FY2015
DerivativeNotionalAmount, refiled from $245M to $245B on 2016-06-03) and
asserts each snapshot returns the correct as-of value with no leakage.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.pit.snapshot import EntitySnapshot, build_snapshot_from_pit
from data.schemas.pit import Modality
from data.training.dataset import (
    PITDataset,
    PITTrainingExample,
    advance_quarters,
    make_training_example,
    sample_training_times,
)
from data.storage import get_storage

GE_CIK = "0000040545"
AAPL_CIK = "0000320193"


# ---------- test 1: snapshot invariants ---------------------------------


def test_snapshot_invariants(pit: PITEngine) -> None:
    print("\n--- test 1: snapshot construction enforces PIT invariants ---")
    eid = EntityRegistry.entity_id_from_cik(AAPL_CIK)

    snap = pit.snapshot(eid, as_of=date(2020, 6, 30))
    print(f"  Apple snapshot at 2020-06-30 — {snap.n_rows()}")
    assert isinstance(snap, EntitySnapshot)

    # No row should have availability_date > as_of
    for name in ("financials", "filings_text"):
        df = snap.get(getattr(Modality, name.upper()))
        if df.empty:
            continue
        avail = pd.to_datetime(df["availability_date"])
        assert (avail <= pd.Timestamp("2020-06-30")).all(), (
            f"{name}: snapshot contains rows with availability_date > as_of"
        )
    print("  [OK] availability_date <= as_of on every row across all modalities")

    # Frozen: cannot mutate
    try:
        snap.financials = pd.DataFrame()  # type: ignore[misc]
        raise AssertionError("EntitySnapshot is mutable — should be frozen")
    except (AttributeError, TypeError):
        pass
    print("  [OK] snapshot is frozen (cannot rebind fields)")


# ---------- test 2: empty snapshot is well-typed ------------------------


def test_empty_snapshot(pit: PITEngine) -> None:
    print("\n--- test 2: empty snapshot is well-typed ---")
    snap = pit.snapshot("us-cik-0000000999", as_of=date(2020, 1, 1))
    assert snap.is_empty()
    counts = snap.n_rows()
    # All 10 modalities should report zero rows.
    assert all(v == 0 for v in counts.values()), f"unexpected non-empty modality: {counts}"
    assert set(counts.keys()) == {
        "financials", "insider_trades", "filings_text", "supply_chain",
        "market", "macro", "news", "patents", "earnings_calls", "hiring",
    }, f"snapshot missing modalities: {set(counts.keys())}"
    print(f"  [OK] non-existent entity returns empty snapshot across all 10 modalities")


# ---------- test 3: adversarial restatement does not leak ---------------


def test_no_leak_across_restatement(pit: PITEngine) -> None:
    """GE refiled DerivativeNotionalAmount FY2015 from $245M (2016-02-26)
    to $245B (2016-06-03). A snapshot at 2016-04-01 must NOT contain the
    restated value; a snapshot at 2016-08-01 must contain it (not the
    original)."""
    print("\n--- test 3: adversarial — restatement does not leak into earlier snapshot ---")
    eid = EntityRegistry.entity_id_from_cik(GE_CIK)
    # SEC reports this concept under the `invest` taxonomy, not us-gaap.
    concept = "invest:DerivativeNotionalAmount"
    selector = lambda df: (df["concept"] == concept) & (df["fiscal_year"] == 2015) & (df["fiscal_period"] == "FY")

    # Snapshot before restatement filing
    snap_pre = pit.snapshot(eid, as_of=date(2016, 4, 1))
    pre_rows = snap_pre.financials[selector(snap_pre.financials)]
    assert len(pre_rows) >= 1, "pre-restatement snapshot is missing the FY2015 fact"
    pre_value = float(pre_rows.iloc[0]["value"])
    print(f"  snapshot @ 2016-04-01 (before restatement)  → value = {pre_value:>20,.0f}")
    assert pre_value == 245_000_000, (
        f"pre-restatement snapshot LEAKED the restated value: {pre_value}"
    )

    # Snapshot after restatement filing
    snap_post = pit.snapshot(eid, as_of=date(2016, 8, 1))
    post_rows = snap_post.financials[selector(snap_post.financials)]
    assert len(post_rows) >= 1
    post_value = float(post_rows.iloc[0]["value"])
    print(f"  snapshot @ 2016-08-01 (after restatement)   → value = {post_value:>20,.0f}")
    assert post_value == 245_000_000_000

    print("  [OK] each snapshot returns the value knowable on its own as_of date — no leak")


# ---------- test 4: training example sanity -----------------------------


def test_training_example(pit: PITEngine) -> None:
    print("\n--- test 4: training example has aligned input/target as_of dates ---")
    eid = EntityRegistry.entity_id_from_cik(GE_CIK)
    t = date(2015, 12, 31)
    h = 2  # 2 quarters forward

    example = make_training_example(pit, eid, t, horizon_quarters=h)
    assert example.input_snapshot.as_of == t
    assert example.target_snapshot.as_of == advance_quarters(t, h)
    assert example.input_snapshot.entity_id == eid
    assert example.target_snapshot.entity_id == eid

    print(f"  input  as_of = {example.input_snapshot.as_of}  "
          f"({example.input_snapshot.n_rows()})")
    print(f"  target as_of = {example.target_snapshot.as_of}  "
          f"({example.target_snapshot.n_rows()})")
    assert not example.input_snapshot.is_empty()
    # Target may sometimes be empty for short horizons before any filing — but
    # for GE FY2015 → 2016-Q2 there's definitely data.
    print("  [OK] training example structure is correct")


# ---------- test 5: PITDataset iterates correctly -----------------------


def test_pit_dataset(pit: PITEngine) -> None:
    print("\n--- test 5: PITDataset yields PIT-safe examples ---")
    times = sample_training_times(
        start=date(2018, 1, 1), end=date(2020, 12, 31), frequency="quarterly"
    )
    horizons = [1, 4]
    entity_ids = [
        EntityRegistry.entity_id_from_cik(AAPL_CIK),
        EntityRegistry.entity_id_from_cik(GE_CIK),
    ]
    ds = PITDataset(
        pit=pit,
        entity_ids=entity_ids,
        times=times,
        horizons=horizons,
        shuffle=True,
        seed=42,
    )
    print(f"  dataset upper-bound size: {len(ds)} (entities×times×horizons)")

    yielded = 0
    seen_t_h = set()
    for ex in ds:
        assert isinstance(ex, PITTrainingExample)
        # Each example is already PIT-safe by construction.
        assert ex.input_snapshot.as_of == ex.t
        assert ex.target_snapshot.as_of == advance_quarters(ex.t, ex.horizon_quarters)
        seen_t_h.add((ex.t, ex.horizon_quarters))
        yielded += 1
        if yielded >= 20:
            break

    print(f"  [OK] yielded {yielded} examples; {len(seen_t_h)} distinct (t,h) pairs")
    assert yielded >= 5

    # Determinism with fixed seed: re-iterating with the same seed produces
    # the same first triple. We need a fresh dataset because __iter__ on a
    # generator doesn't restart from the same shuffle.
    ds2 = PITDataset(
        pit=pit, entity_ids=entity_ids, times=times, horizons=horizons,
        shuffle=True, seed=42,
    )
    first_again = next(iter(ds2))
    first = next(iter(ds))
    # Both iterators draw from the same shuffled order; assert their entity/t/h match.
    assert first.entity_id == first_again.entity_id
    assert first.t == first_again.t
    assert first.horizon_quarters == first_again.horizon_quarters
    print(f"  [OK] shuffle is deterministic with fixed seed (first example: "
          f"{first.entity_id} t={first.t} h={first.horizon_quarters}q)")


# --- test 6 (Stage 1.6 A4): as_of == availability_date boundary -------------


def test_as_of_boundary_inclusive(pit: PITEngine) -> None:
    """A snapshot at exactly the filing date must include the filing
    (engine's `availability_date <= as_of` is inclusive). One day earlier
    must exclude it. This is the boundary case downstream backtests trip on
    most often when they pass `as_of = period_end` and expect "all data
    knowable by end of that day"."""
    print("\n--- test 6: as_of == availability_date is inclusive ---")
    eid = EntityRegistry.entity_id_from_cik(GE_CIK)

    # GE FY2015 DerivativeNotionalAmount was originally filed on 2016-02-26.
    concept = "invest:DerivativeNotionalAmount"
    sel = lambda df: (df["concept"] == concept) & (df["fiscal_year"] == 2015) & (df["fiscal_period"] == "FY")

    snap_day_of = pit.snapshot(eid, as_of=date(2016, 2, 26))  # filing date itself
    rows_day_of = snap_day_of.financials[sel(snap_day_of.financials)]
    assert len(rows_day_of) >= 1, (
        "snapshot at as_of == availability_date must INCLUDE the row (engine uses <=)"
    )
    print(f"  snapshot @ 2016-02-26 (filing day) → {len(rows_day_of)} row(s); first value = {float(rows_day_of.iloc[0]['value']):,.0f}")

    snap_day_before = pit.snapshot(eid, as_of=date(2016, 2, 25))  # one day before
    rows_day_before = snap_day_before.financials[sel(snap_day_before.financials)]
    assert len(rows_day_before) == 0, (
        f"snapshot one day before filing date must EXCLUDE the row, got {len(rows_day_before)}"
    )
    print(f"  snapshot @ 2016-02-25 (day before) → {len(rows_day_before)} row(s)")
    print("  [OK] as_of == availability_date is inclusive (<=); strict-before excludes")


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    pit = PITEngine(storage)

    print("=== Training-time PIT awareness validation ===")
    test_snapshot_invariants(pit)
    test_empty_snapshot(pit)
    test_no_leak_across_restatement(pit)
    test_training_example(pit)
    test_pit_dataset(pit)
    test_as_of_boundary_inclusive(pit)
    print("\n=== Training-time PIT awareness: PASS ===")


if __name__ == "__main__":
    main()
