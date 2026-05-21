"""News event from GDELT 2.0 Events Database.

PIT semantics:
  effective_date    = event_date (when the event happened)
  availability_date = date_added (when GDELT first indexed the event)
  restated_at       = unused (GDELT events aren't restated — corrections
                      appear as separate events)

GDELT's CAMEO event coding scheme:
  cameo_event_code  3- or 4-digit code (e.g. "1822" = Make pessimistic comment)
  event_root_code   first 2 digits (e.g. "18" = Coerce)
  quad_class        1=verbal-coop, 2=material-coop, 3=verbal-conflict, 4=material-conflict
  goldstein_scale   -10 (most conflict) to +10 (most cooperation)
  tone              average tone of source articles, -100 to +100

Entity attribution: the focal entity for each row is whichever GDELT actor
(actor1 or actor2) resolved to a registry CIK first. The other actor's
name is preserved for the model's relational reasoning.
"""
from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from data.schemas.pit import Modality, PITRecord


class NewsEvent(PITRecord):
    """One CAMEO-coded news event mentioning a tracked entity."""

    PARTITION_KEYS: ClassVar[list[str]] = ["global_event_id"]

    # GDELT event extraction is genuinely noisy — many events are weak
    # signals or aggregated from low-trust sources. Discount the prior.
    DEFAULT_PERSPECTIVE: ClassVar[str] = "gdelt:events2"
    DEFAULT_SOURCE_CERTAINTY: ClassVar[float] = 0.6

    modality: Modality = Modality.NEWS

    global_event_id: str = Field(..., description="GDELT GlobalEventID")
    cameo_event_code: str = Field(..., description="CAMEO event code (3-4 digits)")
    event_root_code: str = Field(..., description="CAMEO root code (first 2 digits)")
    quad_class: int = Field(..., description="1=v-coop, 2=m-coop, 3=v-conf, 4=m-conf")
    goldstein_scale: float = Field(..., description="-10 to +10")
    tone: float = Field(..., description="Average article tone, -100 to +100")
    source_url: str = Field(default="", description="Representative source URL")
    actor1_name: str = Field(default="", description="GDELT Actor1 (may be focal entity)")
    actor2_name: str = Field(default="", description="GDELT Actor2 (may be focal entity)")
    n_mentions: int = Field(default=1, description="Number of source articles mentioning this event")
    avg_tone: float = Field(default=0.0, description="Mean tone across mentions")
