"""Patent: one granted patent attributed to an entity (assignee).

PIT semantics:
  effective_date    = application_date (when the invention claim took effect)
  availability_date = grant_date (when the patent became public)
  restated_at       = unused (granted patents aren't withdrawn except in rare
                      invalidation rulings — out of v1 scope)

Entity resolution: assignees are resolved via NameResolver. Patents whose
assignee can't be matched to a registry CIK land under `entity_id =
"unresolved:..."` so they're still aggregatable.
"""
from __future__ import annotations

from datetime import date
from typing import ClassVar, Optional

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class Patent(PITRecord):
    """One granted USPTO patent."""

    PARTITION_KEYS: ClassVar[list[str]] = ["patent_number"]

    DEFAULT_PERSPECTIVE: ClassVar[str] = "uspto:patentsview"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 1.0

    modality: Modality = Modality.PATENTS

    patent_number: str = Field(..., description="USPTO patent number")
    application_date: date
    grant_date: date
    assignee_name_as_filed: str = Field(..., description="Assignee org name as filed")
    title: str = Field(default="", description="Patent title")
    abstract: str = Field(default="", description="Abstract text")
    claims_count: int = Field(default=0, description="Number of claims")
    cpc_classifications: str = Field(
        default="",
        description="JSON-encoded list of CPC classification codes",
    )
    inventors: str = Field(
        default="", description="JSON-encoded list of inventor names"
    )
