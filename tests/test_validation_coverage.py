"""Coverage gate tests.

Verifies the modality-coverage validator correctly:
 - PASSES with sentinel data populated
 - FAILS with stub / empty data
 - Each individual check works in isolation
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest


def _build_aapl_financials(storage, n_quarters: int = 80) -> None:
    """Seed canonical/financials/cik=...0000320193 with N distinct quarters
    of Revenues, including 2020-Q2 (the COVID sentinel).
    """
    from data.pit.engine import PITEngine
    from data.schemas.pit import Modality
    from data.schemas.financial import FinancialFact

    rows = []
    # Build n_quarters of data, anchored so 2020-Q2 falls inside the range
    start_year = 2025 - (n_quarters // 4) + 1
    qid = 0
    for year in range(start_year, 2026):
        for q in (1, 2, 3, 4):
            qid += 1
            if qid > n_quarters:
                break
            rows.append({
                "entity_id": "us-cik-0000320193",
                "modality": "financials",
                "effective_date": date(year, 3 * q, 28),
                "availability_date": date(year, 3 * q + 1 if 3 * q + 1 <= 12 else 12, 15),
                "restated_at": None,
                "source": "test", "source_ref": f"acc-{year}-Q{q}",
                "concept": "us-gaap:Revenues", "taxonomy": "us-gaap",
                "fiscal_year": year, "fiscal_period": f"Q{q}", "unit": "USD",
                "context_id": f"c{year}{q}", "dimensions_json": "",
                "value": 100.0 + qid,
            })
    df = pd.DataFrame(rows)
    PITEngine(storage).write(
        Modality.FINANCIALS, df, partition_keys=FinancialFact.PARTITION_KEYS,
    )


class TestEmptyStorageFails:
    def test_all_checks_fail_on_empty(self, any_storage):
        from data.validation.coverage import run_coverage_checks
        report = run_coverage_checks(any_storage)
        # No data of any kind → every modality check fails
        assert not report.passes
        assert len(report.hard_failures) >= 1


class TestSentinelFactCheck:
    def test_sentinel_passes_with_aapl_2020q2(self, any_storage):
        from data.validation.coverage import (
            CoverageCheck, run_coverage_checks, _check_aapl_sentinel,
        )
        _build_aapl_financials(any_storage, n_quarters=80)
        chk = CoverageCheck(
            name="aapl",
            description="",
            severity="hard",
            check=_check_aapl_sentinel,
        )
        report = run_coverage_checks(any_storage, checks=[chk])
        assert report.passes, f"sentinel should pass; got: {report.results}"

    def test_sentinel_fails_without_2020q2(self, any_storage):
        from data.validation.coverage import (
            CoverageCheck, run_coverage_checks, _check_aapl_sentinel,
        )
        # Only build 4 quarters (won't include 2020-Q2)
        _build_aapl_financials(any_storage, n_quarters=4)
        chk = CoverageCheck(
            name="aapl",
            description="",
            severity="hard",
            check=_check_aapl_sentinel,
        )
        report = run_coverage_checks(any_storage, checks=[chk])
        assert not report.passes


class TestMedianQuartersCheck:
    def test_median_fails_when_below_threshold(self, any_storage):
        from data.validation.coverage import (
            CoverageCheck, run_coverage_checks,
            _check_financials_median_quarters,
        )
        _build_aapl_financials(any_storage, n_quarters=10)  # < 60 threshold
        chk = CoverageCheck(
            name="median", description="", severity="hard",
            check=_check_financials_median_quarters,
        )
        report = run_coverage_checks(any_storage, checks=[chk])
        assert not report.passes

    def test_median_passes_when_above_threshold(self, any_storage):
        from data.validation.coverage import (
            CoverageCheck, run_coverage_checks,
            _check_financials_median_quarters,
        )
        _build_aapl_financials(any_storage, n_quarters=80)
        chk = CoverageCheck(
            name="median", description="", severity="hard",
            check=_check_financials_median_quarters,
        )
        report = run_coverage_checks(any_storage, checks=[chk])
        assert report.passes


class TestReportSerialization:
    def test_results_carry_severity_passed_observed(self, any_storage):
        from data.validation.coverage import run_coverage_checks
        report = run_coverage_checks(any_storage)
        for r in report.results:
            assert r.severity in {"hard", "soft"}
            assert isinstance(r.passed, bool)
            assert isinstance(r.observed, str)

    def test_passes_property_only_considers_hard(self, any_storage):
        """A failing SOFT check should not flip the gate from PASS to FAIL."""
        from data.validation.coverage import (
            CoverageCheck, CoverageReport, CoverageResult,
        )
        report = CoverageReport(results=[
            CoverageResult("a", "hard", True, "ok", "ok"),
            CoverageResult("b", "soft", False, "warn", "warn"),
        ])
        assert report.passes  # hard passes; soft warns are ignored for gate
