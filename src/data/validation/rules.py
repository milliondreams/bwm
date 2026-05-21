"""Rule schema + report dataclasses for the CVR (Constraint Violation Rate) gate.

A rule describes:
  - which XBRL concepts it needs as inputs
  - a `check(values) → bool` that returns True if the constraint holds
  - a tolerance for floating-point identities
  - severity: "hard" rules contribute to the CVR gate; "soft" rules are
    surfaced but excluded from the 2% threshold (used for diagnostics like
    "firm is insolvent" — true but not a data error)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


@dataclass(frozen=True)
class AccountingRule:
    """One declarative data-quality rule for canonical financial records.

    `check` receives a dict mapping each input concept name (as listed in
    `concepts`) to its numeric value for the period. Returns True if the
    identity holds (no violation), False otherwise.

    `extra_filters` lets a rule narrow which rows it applies to — e.g.,
    `{"dimensions_json": ""}` to restrict balance-sheet identity to
    consolidated rows only, excluding segment-level facts.
    """

    name: str
    description: str
    concepts: list[str]
    check: Callable[[dict[str, float]], bool]
    tolerance: float = 0.01
    severity: Literal["hard", "soft"] = "hard"
    extra_filters: dict[str, str] = field(default_factory=dict)
    module: str = "bwm.accounting"


@dataclass
class RuleResult:
    rule_name: str
    module: str
    severity: str
    checks: int = 0
    violations: int = 0
    example_violations: list[dict] = field(default_factory=list)

    @property
    def violation_rate(self) -> float:
        return self.violations / self.checks if self.checks else 0.0


@dataclass
class ConstraintReport:
    per_rule: dict[str, RuleResult] = field(default_factory=dict)
    cvr_threshold: float = 0.02  # v3 § 9.1: ≤ 2%

    @property
    def total_hard_checks(self) -> int:
        return sum(r.checks for r in self.per_rule.values() if r.severity == "hard")

    @property
    def total_hard_violations(self) -> int:
        return sum(r.violations for r in self.per_rule.values() if r.severity == "hard")

    @property
    def cvr(self) -> float:
        c = self.total_hard_checks
        return self.total_hard_violations / c if c else 0.0

    @property
    def passes_gate(self) -> bool:
        # Pass if no hard rule was checked yet (vacuous) OR CVR is within budget.
        if self.total_hard_checks == 0:
            return True
        return self.cvr <= self.cvr_threshold
