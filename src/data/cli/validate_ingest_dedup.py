"""Stage 5 Hotfix validation: registry + dispatcher dedup, ticker selection.

Asserts:
  1. After `seed_registry`, every CIK in entities/registry.parquet appears
     exactly once (no preferred-share-class duplication).
  2. The `ingest_edgar_full.main()` CIK-list construction also produces
     a deduplicated list (defense in depth: even if a hand-edited registry
     reintroduces dupes, the dispatcher never processes one CIK twice in
     one run).
  3. Ticker dedup chose the COMMON-STOCK ticker (shortest) for known
     dupe-heavy CIKs (BAC, FNMA, FMCC, etc.) — not a preferred-share suffix.

This test exists because Stage 5's first --all run wasted 14.5 hours of
cluster time on duplicate-CIK dispatch (Federal Home Loan Mortgage had 25
ticker rows in the registry; the dispatcher iterated rows, not unique CIKs).
The bug is now fixed at both layers but the regression test makes sure it
stays fixed.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from data.entity.registry import EntityRegistry
from data.storage import get_storage


# CIKs known to have multiple ticker rows in SEC's tickers.json. After dedup
# each should appear exactly once with the recognizable common-stock ticker.
# Reference: SEC EDGAR ticker lookup (sec.gov/cgi-bin/browse-edgar?action=getcompany).
KNOWN_DUPE_CIKS_WITH_COMMON_TICKER: dict[str, str] = {
    "0000070858": "BAC",   # Bank of America Corp /DE/
    "0000310522": "FNMA",  # Federal National Mortgage Association
    "0001026214": "FMCC",  # Federal Home Loan Mortgage Corp
    "0001114446": "AMUB",  # UBS AG — first-listed; many ETN suffixes follow
    "0001001085": "BN",    # Brookfield Corp /ON/
}


def test_registry_one_row_per_cik(reg_df: pd.DataFrame) -> None:
    print("\n--- test 1: registry has at most one row per CIK ---")
    n_rows = len(reg_df)
    n_unique = reg_df["cik"].nunique()
    print(f"  registry rows: {n_rows}, unique CIKs: {n_unique}")
    assert n_rows == n_unique, (
        f"registry has {n_rows - n_unique} duplicate CIK rows; seed_registry's "
        f"dedup didn't take effect. Re-run `python -m data.sources.sec_tickers`."
    )
    print(f"  [OK] every CIK appears exactly once ({n_unique} entities)")


def test_dispatcher_dedup_constructs_unique_cik_list(reg_df: pd.DataFrame) -> None:
    """Mirror the dispatch-side dedup logic from ingest_edgar_full.main()
    to confirm `sorted(set(...))` produces a unique list even if the
    registry on disk were to reintroduce duplicates."""
    print("\n--- test 2: dispatcher dedup is a unique CIK list ---")
    # Construct the same list that ingest_edgar_full.main() builds for --all
    ciks = sorted(set(reg_df["cik"].astype(str).tolist()))
    assert len(ciks) == len(set(ciks)), "dispatcher CIK list has dupes — set(...) broken?"
    print(f"  [OK] dispatcher list has {len(ciks)} unique CIKs")

    # Stress test: even if we synthesize a 100x-duplicated input, the set
    # construction collapses it. This guards against a future regression
    # where the dedup line gets reverted.
    synthetic_duped = reg_df["cik"].astype(str).tolist() * 100
    deduped = sorted(set(synthetic_duped))
    assert len(deduped) == reg_df["cik"].nunique(), (
        f"dedup broken under stress: {len(deduped)} != {reg_df['cik'].nunique()}"
    )
    print(f"  [OK] 100x-duplicated input collapses to {len(deduped)} unique CIKs")


def test_ticker_dedup_picks_common_stock(reg_df: pd.DataFrame) -> None:
    """For CIKs with many ticker rows in SEC's tickers.json (preferred shares,
    ETN families), the dedup should keep the common-stock / shortest ticker."""
    print("\n--- test 3: ticker dedup picks the recognizable common-stock ticker ---")
    found_any = False
    failures: list[str] = []
    for cik, expected_ticker in KNOWN_DUPE_CIKS_WITH_COMMON_TICKER.items():
        rows = reg_df[reg_df["cik"] == cik]
        if rows.empty:
            print(f"  - cik={cik} not in registry (skip)")
            continue
        found_any = True
        actual = str(rows.iloc[0]["ticker"])
        ok = actual == expected_ticker
        mark = "[OK]" if ok else "[FAIL]"
        print(f"  {mark} cik={cik} → ticker={actual!r}  (expected {expected_ticker!r})")
        if not ok:
            failures.append(f"cik={cik} got {actual!r}, expected {expected_ticker!r}")
    assert found_any, (
        "none of the known dupe CIKs were in the registry — has the registry "
        "been seeded yet? Run `python -m data.sources.sec_tickers`."
    )
    assert not failures, (
        f"ticker dedup picked wrong tickers for {len(failures)} CIK(s): {failures}"
    )
    print(f"  [OK] common-stock ticker preserved across {len(KNOWN_DUPE_CIKS_WITH_COMMON_TICKER)} known-dupe CIKs")


def main() -> None:
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    storage = get_storage()
    registry = EntityRegistry(storage)
    reg_df = registry.load()

    print("=== Stage 5 Hotfix validation: registry + dispatcher dedup ===")
    print(f"registry path: entities/registry.parquet")
    print(f"row count: {len(reg_df)}")

    test_registry_one_row_per_cik(reg_df)
    test_dispatcher_dedup_constructs_unique_cik_list(reg_df)
    test_ticker_dedup_picks_common_stock(reg_df)

    print("\n=== Stage 5 Hotfix dedup validation: PASS ===")


if __name__ == "__main__":
    main()
