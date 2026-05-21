"""Curated XBRL concept list for v0 financial features.

Frozen, hand-curated set of 30 concepts present in most large-cap 10-K
filings. Selected from the top-50 by row count in our canonical financials
(survey done 2026-05). Adding/removing concepts is a deliberate schema
change — bump V0_CONCEPT_SET_VERSION when modifying.

Normalization: `sign(x) · log1p(|x|) / 30` maps wide-range financials (cents
to trillions) into roughly [-1, 1] without losing sign. No z-score for v0:
the tiny model isn't sensitive to exact scale, and avoiding train/val stats
coupling keeps the data pipeline simple. Phase B switches to per-concept
z-score using the foundation snapshot.
"""
from __future__ import annotations

import math
from typing import Final

V0_CONCEPT_SET_VERSION: Final[str] = "v0.1"

# Order matters — fixed positional index in the feature tensor.
CONCEPTS: Final[tuple[str, ...]] = (
    # Balance sheet — assets
    "us-gaap:Assets",
    "us-gaap:AssetsCurrent",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "us-gaap:InventoryNet",
    "us-gaap:PropertyPlantAndEquipmentNet",
    "us-gaap:Goodwill",
    # Balance sheet — liabilities & equity
    "us-gaap:Liabilities",
    "us-gaap:LiabilitiesCurrent",
    "us-gaap:LiabilitiesAndStockholdersEquity",
    "us-gaap:StockholdersEquity",
    "us-gaap:CommonStockValue",
    "us-gaap:RetainedEarningsAccumulatedDeficit",
    "us-gaap:AccumulatedOtherComprehensiveIncomeLossNetOfTax",
    # Income statement
    "us-gaap:Revenues",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:CostOfRevenue",
    "us-gaap:GrossProfit",
    "us-gaap:OperatingIncomeLoss",
    "us-gaap:NetIncomeLoss",
    "us-gaap:ProfitLoss",
    "us-gaap:ComprehensiveIncomeNetOfTax",
    "us-gaap:IncomeTaxExpenseBenefit",
    "us-gaap:InterestExpense",
    # Cash flow
    "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "us-gaap:NetCashProvidedByUsedInInvestingActivities",
    "us-gaap:NetCashProvidedByUsedInFinancingActivities",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    # Per-share / shares
    "us-gaap:EarningsPerShareBasic",
    "us-gaap:EarningsPerShareDiluted",
    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
)

N_CONCEPTS: Final[int] = len(CONCEPTS)
CONCEPT_TO_IDX: Final[dict[str, int]] = {c: i for i, c in enumerate(CONCEPTS)}

# Normalization constant — log(1e13) ≈ 30 covers trillion-dollar magnitudes.
_LOG_SCALE: Final[float] = 30.0


def normalize_value(x: float) -> float:
    """Sign-preserving log1p, then scale into roughly [-1, 1].

    EPS values (-$50 to $50) land near 0; balance-sheet items ($1B-$1T)
    land around 0.7-0.9. The sign is preserved so the model sees losses
    distinctly from gains.
    """
    if x == 0:
        return 0.0
    sign = 1.0 if x > 0 else -1.0
    return sign * math.log1p(abs(x)) / _LOG_SCALE


def denormalize_value(z: float) -> float:
    """Inverse of normalize_value; for diagnostic / qualitative output only."""
    if z == 0:
        return 0.0
    sign = 1.0 if z > 0 else -1.0
    return sign * math.expm1(abs(z) * _LOG_SCALE)
