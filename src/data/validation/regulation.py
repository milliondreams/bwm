"""bwm.regulation — jurisdiction-aware regulatory constraints (Python).

Mirrors v3 § 5.3 without the Uni/Locy substrate. v1 ships a minimal rule
set; the framework supports adding per-jurisdiction rules later. Rules here
operate on filing-level metadata (form, dates) rather than financial values.
"""
from __future__ import annotations

from data.validation.rules import AccountingRule

# Same dataclass — "AccountingRule" is the registry/runner contract; the
# module label distinguishes regulation rules in the CVR report.

RULES: list[AccountingRule] = []


def _register(rule: AccountingRule) -> AccountingRule:
    RULES.append(rule)
    return rule


# 10-K filing window: SEC large accelerated filers must file within 60 days
# of fiscal year end, accelerated filers within 75, non-accelerated within 90.
# We use 90 as the permissive bound that covers all filer classes. The check
# operates on `availability_date - effective_date` in days.
_register(AccountingRule(
    name="tenk_filing_window",
    description=(
        "10-K filed within 90 days of fiscal year end. Beyond 90 days suggests "
        "either a late filing (SEC 12b-25 NT 10-K) or a data ingest error."
    ),
    concepts=["_filing_lag_days_10k"],  # synthetic concept computed by runner
    check=lambda v: v["_filing_lag_days_10k"] <= 90,
    module="bwm.regulation",
    modality="filings",
    extra_filters={"_form_is": "10-K"},
))
