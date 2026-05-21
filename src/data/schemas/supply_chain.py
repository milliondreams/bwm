"""Supply-chain edge: one inter-entity relationship inferred from filing text.

Stage 6 outputs one row per (source_entity, target_entity, relationship_type,
filing) tuple. PIT semantics follow the source filing's report and filing
dates, so backtests built on the graph respect FR-2.

The graph rolls up to quarterly snapshots in `graph/{quarter}/edges.parquet`
for the spec's planning/simulation layers; this canonical table is the
ground-truth from which those snapshots derive.
"""
from __future__ import annotations

from typing import ClassVar, Optional

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class SupplyChainEdge(PITRecord):
    """One directional edge extracted from a 10-K narrative."""

    PARTITION_KEYS: ClassVar[list[str]] = [
        "target_entity_id",
        "relationship_type",
        "evidence_hash",
    ]

    # LLM-extracted edges carry meaningful uncertainty — set the source prior
    # below 1.0 so downstream training can reweight by source quality.
    DEFAULT_PERSPECTIVE: ClassVar[str] = "sec_edgar:ie"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 0.7

    modality: Modality = Modality.SUPPLY_CHAIN

    target_entity_id: str = Field(
        ...,
        description="Resolved canonical entity id of the counterparty; "
                    "may be an `unresolved:` placeholder when the name "
                    "could not be matched to the registry.",
    )
    target_name_as_filed: str = Field(
        ..., description="Counterparty name verbatim from the filing"
    )
    relationship_type: str = Field(
        ...,
        description="One of: customer, supplier, competitor, partner, "
                    "subsidiary, lender, regulator, other",
    )
    weight_qualifier: str = Field(
        default="",
        description="Free-form qualifier: '10% customer', 'sole supplier', "
                    "'minority owner', etc.",
    )
    evidence_span: str = Field(
        ..., description="Verbatim text supporting the edge (~200-500 chars)"
    )
    evidence_hash: str = Field(
        ..., description="SHA1[:12] of evidence_span; partition-key disambiguator"
    )
    extractor_model: str = Field(
        ..., description="LLM used, e.g. 'gpt-5.5' (Azure OpenAI deployment name)"
    )
    confidence: Optional[float] = Field(
        default=None, description="Model-reported confidence in [0, 1], if available"
    )

    form: str = Field(default="10-K")
    accession: str = Field(...)
    item_id: str = Field(default="", description="Filing item the edge was found in")
