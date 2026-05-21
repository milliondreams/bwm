"""Financial fact: one XBRL concept's value for one entity at one period."""
from __future__ import annotations

from datetime import date
from typing import ClassVar, Optional

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class FinancialFact(PITRecord):
    """One observed financial measurement.

    Represents a single XBRL fact: e.g. (AAPL, us-gaap:Revenues, 2020-Q2,
    $59.685B, filed 2020-07-31, from accession 0000320193-20-000062).
    Multiple FinancialFact rows for the same (entity, concept, period) with
    different `availability_date`s encode the restatement history.

    `context_id` and `dimensions_json` are populated by the Stage 2 per-filing
    XBRL parser. They're declared here (with None/empty defaults) so the
    PARTITION_KEYS list is stable across the legacy and new connectors —
    legacy companyfacts rows just leave them blank.
    """

    PARTITION_KEYS: ClassVar[list[str]] = [
        "concept",
        "taxonomy",
        "fiscal_year",
        "fiscal_period",
        "unit",
        "context_id",
        "dimensions_json",
    ]

    # v3 source defaults (Stage 1.7) — overridable per-record. The companyfacts
    # connector overrides perspective to "sec_edgar:companyfacts"; per-filing
    # XBRL ingest sets it to "sec_edgar:xbrl_instance". Both share the same
    # source_certainty prior since regulatory filings are definitive.
    DEFAULT_PERSPECTIVE: ClassVar[str] = "sec_edgar:xbrl_instance"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 1.0

    modality: Modality = Modality.FINANCIALS

    concept: str = Field(..., description="XBRL concept, e.g. 'us-gaap:Revenues'")
    taxonomy: str = Field(default="us-gaap", description="Taxonomy namespace")
    period_start: Optional[date] = Field(
        default=None, description="For duration concepts; null for instant concepts"
    )
    period_end: date = Field(..., description="Equals effective_date for instant facts; period end for duration")
    fiscal_year: int
    fiscal_period: str = Field(..., description="FY, Q1, Q2, Q3, Q4")
    unit: str = Field(..., description="e.g. 'USD', 'shares', 'pure'")
    value: float
    form: str = Field(..., description="Filing form, e.g. '10-K', '10-Q', '8-K'")
    accession: str = Field(..., description="SEC accession number, e.g. '0000320193-20-000062'")

    # Stage-2-populated fields (legacy connector leaves these blank).
    context_id: str = Field(default="", description="XBRL context reference within the filing")
    dimensions_json: str = Field(
        default="",
        description="JSON-encoded segment/axis dimensions; empty for consolidated facts",
    )
