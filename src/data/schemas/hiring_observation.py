"""Hiring observation — industry-aggregated from BLS JOLTS.

Documented as substitute coverage: per-company hiring data isn't available
on free APIs. The spec's intent (model uses hiring signal) is partially
satisfied by NAICS-industry aggregates from BLS JOLTS (Job Openings and
Labor Turnover Survey). Downstream training code looks up the focal
entity's NAICS code and attaches the matching JOLTS series as a context
feature.

PIT semantics:
  effective_date    = observation month end
  availability_date = release date (BLS publishes JOLTS ~6 weeks after the
                      observation month)
  restated_at       = used for BLS revisions (uncommon but does happen)

Entity model: each NAICS industry + measure is a pseudo-entity with
entity_id = "bls:naics-{code}". Measures: job_openings, hires,
total_separations, quits, layoffs_discharges.
"""
from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class HiringObservation(PITRecord):
    """One BLS JOLTS observation for one (NAICS industry, measure)."""

    PARTITION_KEYS: ClassVar[list[str]] = ["measure"]

    DEFAULT_PERSPECTIVE: ClassVar[str] = "bls:jolts"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 1.0

    modality: Modality = Modality.HIRING

    naics_industry: str = Field(..., description="NAICS industry code (3-digit)")
    naics_label: str = Field(default="", description="Human-readable industry name")
    measure: str = Field(
        ...,
        description="JOLTS measure: job_openings, hires, total_separations, "
                    "quits, layoffs_discharges",
    )
    value: float = Field(..., description="Level (thousands of persons)")
    seasonal_adjustment: str = Field(
        default="seasonally_adjusted", description="seasonally_adjusted | not_seasonally_adjusted"
    )
