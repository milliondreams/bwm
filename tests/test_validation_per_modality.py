"""Per-modality CVR breakdown tests for ConstraintReport.

The per_modality view aggregates hard-rule checks/violations by the
`modality` field that every AccountingRule now carries. Soft rules are
excluded (matching the aggregate gate's semantics).
"""
from __future__ import annotations

from data.validation.accounting import RULES as ACCOUNTING_RULES
from data.validation.regulation import RULES as REGULATION_RULES
from data.validation.rules import ConstraintReport, RuleResult


def _build_report() -> ConstraintReport:
    report = ConstraintReport()
    for rule in ACCOUNTING_RULES + REGULATION_RULES:
        report.per_rule[rule.name] = RuleResult(
            rule_name=rule.name, module=rule.module,
            severity=rule.severity, modality=rule.modality,
        )
    return report


class TestModalityField:
    def test_every_accounting_rule_has_modality(self):
        for rule in ACCOUNTING_RULES:
            assert rule.modality, f"{rule.name} missing modality"

    def test_regulation_10k_rule_is_filings_modality(self):
        by_name = {r.name: r for r in REGULATION_RULES}
        assert by_name["tenk_filing_window"].modality == "filings"

    def test_accounting_rules_default_to_financials(self):
        for rule in ACCOUNTING_RULES:
            assert rule.modality == "financials"


class TestPerModalityAggregation:
    def test_empty_report_has_empty_per_modality(self):
        report = ConstraintReport()
        assert report.per_modality == {}

    def test_aggregation_matches_total_hard_checks(self):
        report = _build_report()
        # Synthesize checks on the first hard rule of each modality
        for name, res in report.per_rule.items():
            if res.severity == "hard" and res.modality == "financials":
                res.checks = 100
                res.violations = 2
                break
        for name, res in report.per_rule.items():
            if res.severity == "hard" and res.modality == "filings":
                res.checks = 50
                res.violations = 1
                break

        agg = report.per_modality
        assert agg["financials"]["checks"] == 100
        assert agg["financials"]["violations"] == 2
        assert agg["financials"]["cvr"] == 0.02
        assert agg["filings"]["checks"] == 50
        assert agg["filings"]["violations"] == 1
        assert agg["filings"]["cvr"] == 0.02

        # Sum of per-modality == aggregate (hard-only)
        sum_checks = sum(m["checks"] for m in agg.values())
        sum_violations = sum(m["violations"] for m in agg.values())
        assert sum_checks == report.total_hard_checks
        assert sum_violations == report.total_hard_violations

    def test_soft_rules_excluded_from_per_modality(self):
        report = _build_report()
        for res in report.per_rule.values():
            if res.severity == "soft":
                res.checks = 1000
                res.violations = 500
        # No hard checks anywhere → per_modality empty
        assert report.per_modality == {}

    def test_zero_check_modality_dropped_from_view(self):
        report = ConstraintReport()
        report.per_rule["r1"] = RuleResult(
            rule_name="r1", module="bwm.accounting", severity="hard",
            modality="financials", checks=0, violations=0,
        )
        # Noise suppression: zero-check modalities don't appear.
        assert "financials" not in report.per_modality
