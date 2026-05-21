"""Substrate-independent data-quality validation (v3 § 5.2–5.3 in Python).

Mirrors the semantics of the spec's Locy rule modules `bwm.accounting` and
`bwm.regulation` without taking on the Uni/Locy substrate. The Python rules
here run against the existing canonical Parquet shards and produce a
Constraint Violation Rate (CVR) report. v3 § 9.1 Phase A acceptance:
"Hard CVR on bwm.accounting + bwm.regulation ≤ 2%."
"""
from data.validation.rules import AccountingRule, RuleResult, ConstraintReport
from data.validation.runner import ConstraintRunner

__all__ = ["AccountingRule", "RuleResult", "ConstraintReport", "ConstraintRunner"]
