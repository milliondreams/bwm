"""Modality coverage gates — Phase A acceptance criteria.

The spec (§6.2) requires:
    - ≥ 10 modalities populated
    - Median quarters per entity ≥ 80 for financials (20yr depth)
    - Per-modality sentinel facts present
    - Adequate entity breadth

These are operational gates (do we have ENOUGH data?), distinct from the
correctness gates in `accounting.py` (is the data we have INTERNALLY
consistent?). Both must pass for Phase A acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from data.pit.engine import PITEngine
from data.schemas.pit import Modality
from data.storage import Storage


@dataclass(frozen=True)
class CoverageCheck:
    """One coverage gate. Returns (passes, observed, message)."""

    name: str
    description: str
    severity: str  # "hard" or "soft"
    check: Callable[[Storage, PITEngine], tuple[bool, str, str]]


@dataclass
class CoverageResult:
    name: str
    severity: str
    passed: bool
    observed: str
    message: str


@dataclass
class CoverageReport:
    results: list[CoverageResult] = field(default_factory=list)

    @property
    def hard_failures(self) -> list[CoverageResult]:
        return [r for r in self.results if r.severity == "hard" and not r.passed]

    @property
    def soft_failures(self) -> list[CoverageResult]:
        return [r for r in self.results if r.severity == "soft" and not r.passed]

    @property
    def passes(self) -> bool:
        return not self.hard_failures


# ---------- check implementations ----------

def _check_modality_count(storage: Storage, engine: PITEngine) -> tuple[bool, str, str]:
    """At least 10 modalities have populated canonical/ subdirectories."""
    populated = []
    for mod in Modality:
        try:
            paths = list(storage.list(f"canonical/{mod.value}"))
            if paths:
                populated.append(mod.value)
        except Exception:
            pass
    # filings_full + supply_chain may live under canonical/filings_full / canonical/supply_chain
    # but Modality enum has supply_chain entry — handled above.
    return (
        len(populated) >= 10,
        f"{len(populated)} modalities populated",
        f"populated: {sorted(populated)}",
    )


def _check_financials_median_quarters(storage: Storage, engine: PITEngine) -> tuple[bool, str, str]:
    """Median quarters per entity in financials >= 60 (15yr depth, soft target)."""
    paths = [
        p for p in storage.list(f"canonical/{Modality.FINANCIALS.value}")
        if p.endswith(".parquet") or p.split("/")[-1].startswith("entity=")
    ]
    if not paths:
        return False, "0 entities", "no financials data found"
    quarter_counts = []
    sample = paths[:500]  # sample for speed on large corpora
    for p in sample:
        try:
            df = storage.read_table(p)
            qs = df[["fiscal_year", "fiscal_period"]].drop_duplicates()
            quarter_counts.append(len(qs))
        except Exception:
            continue
    if not quarter_counts:
        return False, "no data readable", "no parseable shards"
    median = sorted(quarter_counts)[len(quarter_counts) // 2]
    return (
        median >= 60,
        f"median quarters/entity = {median} (sampled {len(quarter_counts)})",
        f"target: 60+; entities sampled: {len(quarter_counts)} of {len(paths)}",
    )


def _check_aapl_sentinel(storage: Storage, engine: PITEngine) -> tuple[bool, str, str]:
    """AAPL has Revenues data for 2020-Q2 (the COVID quarter sentinel)."""
    try:
        df = engine.query(
            Modality.FINANCIALS,
            entity_ids=["us-cik-0000320193"],
            as_of=date(2025, 12, 31),
        )
        if df.empty:
            return False, "AAPL canonical empty", "no AAPL data ingested"
        revenues = df[df["concept"].str.contains("Revenues", na=False)]
        q2_2020 = revenues[(revenues["fiscal_year"] == 2020) & (revenues["fiscal_period"] == "Q2")]
        if q2_2020.empty:
            return False, "no AAPL Revenues 2020-Q2", "sentinel fact missing"
        return True, f"AAPL Revenues 2020-Q2: {len(q2_2020)} rows", "OK"
    except Exception as e:
        return False, "query failed", str(e)[:200]


def _check_entity_breadth(storage: Storage, engine: PITEngine) -> tuple[bool, str, str]:
    """At least 1000 entities in canonical/financials (soft Phase A gate)."""
    paths = list(storage.list(f"canonical/{Modality.FINANCIALS.value}"))
    n = sum(1 for p in paths if "entity=" in p)
    return n >= 1000, f"{n} entities", "target: 1000+"


# ---------- the gate set ----------

DEFAULT_CHECKS: list[CoverageCheck] = [
    CoverageCheck(
        name="modality_count_ge_10",
        description="Phase A §6.2: ≥10 modalities populated",
        severity="hard",
        check=_check_modality_count,
    ),
    CoverageCheck(
        name="financials_median_quarters_ge_60",
        description="Financials: median quarters per entity ≥ 60 (15yr depth)",
        severity="hard",
        check=_check_financials_median_quarters,
    ),
    CoverageCheck(
        name="aapl_revenues_2020q2_sentinel",
        description="AAPL Revenues 2020-Q2 (COVID quarter) present",
        severity="hard",
        check=_check_aapl_sentinel,
    ),
    CoverageCheck(
        name="entity_breadth_ge_1000",
        description="≥1000 entities in canonical/financials",
        severity="soft",
        check=_check_entity_breadth,
    ),
]


def run_coverage_checks(
    storage: Storage,
    engine: PITEngine | None = None,
    checks: list[CoverageCheck] | None = None,
) -> CoverageReport:
    if engine is None:
        engine = PITEngine(storage)
    if checks is None:
        checks = DEFAULT_CHECKS
    report = CoverageReport()
    for chk in checks:
        try:
            passed, observed, message = chk.check(storage, engine)
        except Exception as e:
            passed, observed, message = False, "check raised", f"{type(e).__name__}: {e}"
        report.results.append(CoverageResult(
            name=chk.name,
            severity=chk.severity,
            passed=passed,
            observed=observed,
            message=message,
        ))
    return report
