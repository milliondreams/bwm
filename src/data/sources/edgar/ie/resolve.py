"""Entity-name resolution: fuzzy match company names to registry CIKs.

The LLM emits counterparty names as they appear in the filing ("Foxconn",
"TSMC", "Samsung Electronics"). The registry stores SEC-filing names like
"HON HAI PRECISION INDUSTRY CO LTD", "TAIWAN SEMICONDUCTOR MANUFACTURING
COMPANY LTD", "SAMSUNG ELECTRONICS CO LTD".

Strategy:
  1. Normalize both sides — uppercase, strip incorporation suffixes
     (CO, CORP, INC, LTD, LLC, AG, SA, NV, PLC), drop "the".
  2. Fuzzy match with rapidfuzz at multiple thresholds.
  3. If no good match, return an `unresolved:<sanitized-name>` placeholder
     so downstream code still has a stable id but can flag it for human
     review later.

This is a v0. A real production resolver would consult a curated
synonym table (Apple↔AAPL↔"Apple Inc"↔"Apple Computer Inc"), LEI/ISIN
cross-references, and a hand-curated alias file. We log unresolved names
so we can build that alias file from observed data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

# Suffixes that the SEC includes in legal entity names but humans/filings drop.
_SUFFIX_TOKENS = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LTD", "LIMITED", "LLC", "LLP", "LP", "PLC", "AG", "SA", "NV",
    "GMBH", "PTE", "HOLDINGS", "GROUP", "INTERNATIONAL", "INTL",
    "THE", "PRECISION", "INDUSTRY", "INDUSTRIES",
}

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(name: str) -> str:
    if not name:
        return ""
    s = _PUNCT_RE.sub(" ", name.upper())
    tokens = [t for t in s.split() if t and t not in _SUFFIX_TOKENS]
    return " ".join(tokens)


@dataclass(frozen=True)
class Resolution:
    entity_id: str  # canonical id or "unresolved:<slug>"
    canonical_name: str
    score: float
    is_resolved: bool


class NameResolver:
    """Lazy-loaded fuzzy resolver over the registry."""

    def __init__(self, registry_df: pd.DataFrame, min_score: float = 88.0) -> None:
        # Build the lookup once. Map normalized name → (entity_id, canonical name).
        self.min_score = min_score
        normalized_map: dict[str, tuple[str, str]] = {}
        for _, row in registry_df.iterrows():
            name = str(row.get("name", "") or "")
            ticker = str(row.get("ticker", "") or "")
            cik = str(row.get("cik", "") or "")
            if not cik:
                continue
            entity_id = f"us-cik-{cik.zfill(10)}"
            norm = normalize(name)
            if norm:
                normalized_map.setdefault(norm, (entity_id, name))
            if ticker:
                # Also index by ticker so "AAPL" → Apple matches.
                normalized_map.setdefault(ticker.upper(), (entity_id, name))
        self._norm_to_entity = normalized_map
        self._norm_choices = list(normalized_map.keys())

    def resolve(self, name_as_filed: str) -> Resolution:
        norm = normalize(name_as_filed)
        if not norm:
            return Resolution(
                entity_id=f"unresolved:{normalize(name_as_filed)[:50] or 'empty'}",
                canonical_name=name_as_filed,
                score=0.0,
                is_resolved=False,
            )
        # Exact normalized hit short-circuits the fuzzy search.
        if norm in self._norm_to_entity:
            eid, canonical = self._norm_to_entity[norm]
            return Resolution(eid, canonical, 100.0, True)

        # Fuzzy fallback — token-set ratio handles word reordering.
        match = process.extractOne(
            norm, self._norm_choices, scorer=fuzz.token_set_ratio
        )
        if match is None:
            return Resolution(
                entity_id=f"unresolved:{norm[:50]}",
                canonical_name=name_as_filed,
                score=0.0,
                is_resolved=False,
            )
        candidate_norm, score, _ = match
        if score < self.min_score:
            return Resolution(
                entity_id=f"unresolved:{norm[:50]}",
                canonical_name=name_as_filed,
                score=float(score),
                is_resolved=False,
            )
        eid, canonical = self._norm_to_entity[candidate_norm]
        return Resolution(eid, canonical, float(score), True)
