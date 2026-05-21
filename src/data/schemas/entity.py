"""Canonical entity record (DR-2 source-of-truth identifiers)."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    """A business entity. The `entity_id` is the canonical key used everywhere
    else in the data lake; external identifiers are stored for resolution but
    are not the primary key (one company can have multiple CIKs over time
    due to restructurings — see OQ-5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(..., description="Canonical id, e.g. 'us-cik-0000320193' for Apple")
    name: str
    cik: Optional[str] = Field(default=None, description="SEC Central Index Key, 10-digit zero-padded")
    ticker: Optional[str] = Field(default=None, description="Primary listing ticker")
    lei: Optional[str] = Field(default=None, description="Legal Entity Identifier (20-char)")
    isin: Optional[str] = Field(default=None, description="ISIN of primary security")
    country: Optional[str] = Field(default=None, description="ISO 3166-1 alpha-2")
    sic: Optional[str] = Field(default=None, description="SIC industry code")

    # Lifecycle (DR-2: survivorship-bias-free)
    incorporated_on: Optional[date] = None
    delisted_on: Optional[date] = None
