"""Point-in-time envelope wrapping every record in the data lake.

Every modality's records embed (or are decorated with) this envelope so the
PIT engine can answer "what was knowable about X on date D" uniformly across
financials, filings, market data, news, etc.

The three time fields encode the spec's PIT contract (§5.6, FR-2):
- effective_date:    when the underlying fact was true (≡ v3's `valid_from`)
- availability_date: when the fact became knowable to outsiders
                     (≡ v3's `knowable_from`)
- restated_at:       if this row was later superseded by a restatement, the
                     date that successor was filed; nullable

v3 spec field set (Stage 1.7) — additive metadata for richer training features
and substrate-independent data-quality auditing:

- recorded_at:                when BWM ingested the record (distinct from
                              availability_date; tracks ingest lag)
- belief:                     confidence in the value, 1.0 for definitive,
                              <1.0 for uncertain (news sentiment, LLM edges)
- perspective:                stable source identifier for multi-source merge
- policy_tags:                residency / sensitivity / jurisdiction tags
- source_certainty:           per-source reliability prior in [0, 1]
- valid_from_uncertainty_days: epistemic uncertainty around effective_date;
                              0 for exact dates, >0 for approximate (news
                              event timing, undated patent). Forward-looking
                              BTIC stand-in; actual BTIC encoding is deferred
                              to Uni-substrate work which is out of scope for
                              the training pipeline.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field


class Modality(str, Enum):
    FINANCIALS = "financials"
    FILINGS_TEXT = "filings_text"
    EARNINGS_CALLS = "earnings_calls"
    MARKET = "market"
    MACRO = "macro"
    NEWS = "news"
    INSIDER_TRADES = "insider_trades"
    PATENTS = "patents"
    HIRING = "hiring"
    SUPPLY_CHAIN = "supply_chain"


class PITRecord(BaseModel):
    """Base envelope. Concrete modalities subclass this and add a payload.

    Subclasses MAY declare `PARTITION_KEYS` — the extra columns (beyond
    `entity_id` and `effective_date`) that distinguish two rows as different
    logical facts vs. restatements of the same fact. The PIT engine reads this
    to decide which rows collapse under as-of dedup. Empty default means the
    modality has no extra dimensions and (entity, effective_date) is the full
    logical key.
    """

    PARTITION_KEYS: ClassVar[list[str]] = []

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(..., description="Canonical entity id (see entity.registry)")
    modality: Modality
    effective_date: date = Field(..., description="When the fact was true (v3: valid_from)")
    availability_date: date = Field(..., description="When the fact became knowable (v3: knowable_from)")
    restated_at: Optional[date] = Field(
        default=None,
        description="If this row was superseded by a restatement, the filing date of the successor",
    )
    source: str = Field(..., description="Source system identifier, e.g. 'sec_edgar:companyfacts'")
    source_ref: str = Field(
        ...,
        description="Provenance pointer (URL, accession number, file hash) — used for audit (NFR-2)",
    )

    # --- v3 spec field set (Stage 1.7) ---
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When BWM ingested this record (UTC). Distinct from availability_date.",
    )
    belief: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Confidence in the value. 1.0 = definitive (regulatory). <1.0 = uncertain.",
    )
    perspective: str = Field(
        default="",
        description="Stable source identifier for multi-source merge (e.g., 'sec_edgar', 'gdelt:events2')",
    )
    policy_tags: list[str] = Field(
        default_factory=list,
        description="Residency / sensitivity / jurisdiction tags (set semantics; empty for public data)",
    )
    source_certainty: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Per-source reliability prior in [0, 1].",
    )
    valid_from_uncertainty_days: int = Field(
        default=0,
        ge=0,
        description="Epistemic uncertainty around effective_date in days. 0 = exact.",
    )

    def is_known_on(self, as_of: date) -> bool:
        """Was this record knowable on `as_of` and not yet restated?"""
        if as_of < self.availability_date:
            return False
        if self.restated_at is not None and self.restated_at <= as_of:
            return False
        return True

    # --- v3 spec field defaults (Stage 1.7) ---

    @classmethod
    def pit_default_fields(cls) -> dict:
        """Return the v3 PIT field defaults for this schema as a dict.

        Spread this into a row dict during ingest so the v3 field set lands
        with correct per-source values without each CLI repeating defaults.
        Modality-specific values come from `DEFAULT_PERSPECTIVE` and
        `DEFAULT_SOURCE_CERTAINTY` ClassVars on the concrete schema; common
        fields (belief, policy_tags, recorded_at, valid_from_uncertainty_days)
        use stable defaults defined here.

        Note: `recorded_at` is evaluated at *call time*, so each call returns
        a fresh now-UTC. Callers writing many rows in a batch should call
        this once per batch to keep the timestamp consistent within the batch.
        """
        return {
            "recorded_at": datetime.now(timezone.utc),
            "belief": 1.0,
            "perspective": getattr(cls, "DEFAULT_PERSPECTIVE", ""),
            "policy_tags": [],
            "source_certainty": getattr(cls, "DEFAULT_SOURCE_CERTAINTY", 1.0),
            "valid_from_uncertainty_days": 0,
        }
