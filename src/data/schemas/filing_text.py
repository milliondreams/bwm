"""Filing-text chunk: one extracted section of narrative from a 10-K/Q/8-K.

Filings are split by Item (10-K Item 1A "Risk Factors", Item 7 "MD&A", etc.)
and chunked into ~1000-token windows with overlap for downstream embedding.

PIT semantics:
  effective_date    = filing's report_date (period end)
  availability_date = filing's filed date
  source_ref        = SEC accession number

The dedup partition key is (item_id, chunk_idx) within an (entity, filing)
so re-extracting the same filing produces byte-identical rows and writes
are idempotent.
"""
from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class FilingTextChunk(PITRecord):
    """One chunk of narrative text from a filing."""

    PARTITION_KEYS: ClassVar[list[str]] = [
        "form",
        "item_id",
        "chunk_idx",
    ]

    DEFAULT_PERSPECTIVE: ClassVar[str] = "sec_edgar:filing_html"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 1.0

    modality: Modality = Modality.FILINGS_TEXT

    form: str = Field(..., description="Filing form, e.g. '10-K', '10-Q', '8-K'")
    accession: str = Field(..., description="SEC accession number")
    item_id: str = Field(..., description="e.g. '1A', '7', '7A', '1.01' for 8-K items")
    item_title: str = Field(default="", description="Heading text as filed")
    chunk_idx: int = Field(..., description="0-based index of chunk within the item")
    n_chunks_in_item: int = Field(..., description="Total chunks for this item")
    text: str = Field(..., description="Cleaned chunk text")
    n_chars: int = Field(..., description="Character count of the chunk")
    n_tokens_approx: int = Field(
        ..., description="Rough token count (chars/4) for budget planning"
    )
