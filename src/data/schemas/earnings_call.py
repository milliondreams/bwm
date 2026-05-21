"""Earnings call excerpt — derived from 8-K Item 2.02 press releases.

PIT semantics inherited from the source 8-K filing:
  effective_date    = filing's report_date (period being announced)
  availability_date = filing's filed date (public release timestamp)

`coverage_class` distinguishes:
  - "press_release": prepared remarks / official press release text, available
    via 8-K Item 2.02. This is what v1 emits.
  - "transcript_qa": Q&A portion of the earnings call. Not available from
    free SEC data; requires a paid transcript provider (Seeking Alpha,
    AlphaSense, etc.). v1 does NOT emit these; documented for v2.

Each row carries the original (form, accession, item_id, chunk_idx) pointer
back to canonical/filings_text so downstream consumers can join.
"""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class EarningsCallExcerpt(PITRecord):
    """One chunk of earnings-call-adjacent narrative derived from an 8-K."""

    PARTITION_KEYS: ClassVar[list[str]] = ["accession", "chunk_idx"]

    # Derived from a source filing; inherits its certainty (regulatory press
    # releases are definitive). v2 transcript_qa from paid sources will pick
    # its own prior at that time.
    DEFAULT_PERSPECTIVE: ClassVar[str] = "derived:filings_text/8-K_item_2.02"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 1.0

    modality: Modality = Modality.EARNINGS_CALLS

    coverage_class: Literal["press_release", "transcript_qa"] = Field(
        ..., description="v1 emits only 'press_release'; 'transcript_qa' is reserved"
    )
    form: str = Field(default="8-K", description="Source filing form")
    accession: str = Field(..., description="Source filing accession")
    item_id: str = Field(default="2.02", description="Source 8-K item id")
    chunk_idx: int = Field(..., description="Chunk index within the source item")
    text: str = Field(..., description="Excerpt text")
    n_chars: int = Field(...)
    n_tokens_approx: int = Field(...)
