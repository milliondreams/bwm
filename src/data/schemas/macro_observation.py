"""Macro observation: one (series, date) data point from FRED.

PIT semantics — *critical for macro*:
  effective_date    = observation date (period the value describes)
  availability_date = release date (when FRED first published this value)
  restated_at       = filled in when a later FRED release supersedes this
                      value (using ALFRED real-time vintages)

Entity model: each FRED series is treated as a pseudo-entity with
entity_id = "fred:{series_id}". This fits the existing per-entity partition
storage layout without engine changes.

Example: CPI for 2020-06 was first released 2020-07-14 (value=257.797).
Subsequent FRED releases revised this in seasonal adjustment passes. A model
trained on retrospectively-revised values silently knows future information
about how the BLS will eventually adjust the seasonal factors.
"""
from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class MacroObservation(PITRecord):
    """One observation of one FRED series, as published at one release."""

    PARTITION_KEYS: ClassVar[list[str]] = ["series_id"]

    DEFAULT_PERSPECTIVE: ClassVar[str] = "fred:alfred"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 1.0

    modality: Modality = Modality.MACRO

    series_id: str = Field(..., description="FRED series identifier, e.g. 'CPIAUCSL'")
    value: float = Field(..., description="Observation value as published at release_date")
    series_title: str = Field(default="", description="Human-readable series title")
    units: str = Field(default="", description="Units as reported by FRED")
    frequency: str = Field(default="", description="'Daily', 'Monthly', 'Quarterly', etc.")
