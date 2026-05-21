"""bwm.accounting — hard accounting identities (Python implementation).

Mirrors the semantics of v3 § 5.2 without the Uni/Locy substrate. Each rule
asserts a relationship that should hold on every set of facts reported in
the same filing for the same period. Hard rules contribute to the CVR gate
(v3 § 9.1: hard CVR ≤ 2%).

Tolerance handling: floating-point comparisons use a relative tolerance of
1% of the largest magnitude operand. XBRL numbers are typically rounded to
millions or thousands, so 1% absorbs reporting rounding without masking
real identity failures.
"""
from __future__ import annotations

from data.validation.rules import AccountingRule


def _rel_close(a: float, b: float, tol: float) -> bool:
    """True if |a − b| / max(|a|, |b|, 1) ≤ tol."""
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


# Concept name reference. SEC XBRL standardizes these names across filers.
ASSETS = "us-gaap:Assets"
LIABILITIES = "us-gaap:Liabilities"
STOCKHOLDERS_EQUITY = "us-gaap:StockholdersEquity"
LIABILITIES_AND_EQUITY = "us-gaap:LiabilitiesAndStockholdersEquity"
CURRENT_ASSETS = "us-gaap:AssetsCurrent"
CURRENT_LIABILITIES = "us-gaap:LiabilitiesCurrent"
REVENUES = "us-gaap:Revenues"
REVENUE_606 = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"


# ---- rule definitions ------------------------------------------------------

RULES: list[AccountingRule] = []


def _register(rule: AccountingRule) -> AccountingRule:
    RULES.append(rule)
    return rule


_register(AccountingRule(
    name="assets_eq_liabilities_plus_equity",
    description=(
        "Balance sheet identity: Assets = Liabilities + StockholdersEquity. "
        "SOFT rule because firms with material non-controlling interests "
        "(minority interest) systematically violate this decomposed form — "
        "their true identity is Assets = Liab + SE + MinorityInterest. The "
        "hard balance-sheet check is `assets_eq_liabilities_and_equity_alias` "
        "below, which uses XBRL's `LiabilitiesAndStockholdersEquity` single "
        "concept and is unambiguous."
    ),
    concepts=[ASSETS, LIABILITIES, STOCKHOLDERS_EQUITY],
    check=lambda v: _rel_close(
        v[ASSETS], v[LIABILITIES] + v[STOCKHOLDERS_EQUITY], tol=0.01
    ),
    severity="soft",
    extra_filters={"dimensions_json": ""},
))

_register(AccountingRule(
    name="assets_eq_liabilities_and_equity_alias",
    description=(
        "Alternate balance sheet expression: Assets = LiabilitiesAndStockholdersEquity. "
        "Some filers report `us-gaap:LiabilitiesAndStockholdersEquity` directly "
        "as a single concept; when present it must match Assets exactly."
    ),
    concepts=[ASSETS, LIABILITIES_AND_EQUITY],
    check=lambda v: _rel_close(v[ASSETS], v[LIABILITIES_AND_EQUITY], tol=0.01),
    extra_filters={"dimensions_json": ""},
))

_register(AccountingRule(
    name="nonnegative_total_assets",
    description="Assets ≥ 0. Negative total assets would indicate a sign error.",
    concepts=[ASSETS],
    check=lambda v: v[ASSETS] >= 0,
    extra_filters={"dimensions_json": ""},
))

_register(AccountingRule(
    name="current_assets_leq_total_assets",
    description=(
        "CurrentAssets ≤ Assets (current assets are a subset of total assets). "
        "Allow 1% tolerance for occasional XBRL reclassifications between filings."
    ),
    concepts=[ASSETS, CURRENT_ASSETS],
    check=lambda v: v[CURRENT_ASSETS] <= v[ASSETS] * 1.01,
    extra_filters={"dimensions_json": ""},
))

_register(AccountingRule(
    name="current_liabilities_leq_total_liabilities",
    description=(
        "CurrentLiabilities ≤ Liabilities. Same rationale as current vs total assets."
    ),
    concepts=[LIABILITIES, CURRENT_LIABILITIES],
    check=lambda v: v[CURRENT_LIABILITIES] <= v[LIABILITIES] * 1.01,
    extra_filters={"dimensions_json": ""},
))

# Soft rule: insolvency. NOT a data error — some firms genuinely have negative
# equity (Boeing post-737 MAX, GameStop pre-2021 squeeze, etc.). We surface it
# for diagnostics but it doesn't contribute to the CVR gate.
_register(AccountingRule(
    name="solvent_positive_equity",
    description=(
        "StockholdersEquity ≥ 0 (diagnostic only; insolvent firms are real, not "
        "data errors)."
    ),
    concepts=[STOCKHOLDERS_EQUITY],
    check=lambda v: v[STOCKHOLDERS_EQUITY] >= 0,
    severity="soft",
    extra_filters={"dimensions_json": ""},
))
