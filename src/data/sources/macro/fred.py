"""FRED macro data with vintage-aware (ALFRED) point-in-time semantics.

FRED's standard API returns the *latest* value for a series. ALFRED (Archive
of FRED) returns the value *as it was first published* on a given release
date, plus the full revision history. We use ALFRED so that PIT queries
get the values that were actually knowable at as_of dates — not values that
were later revised in subsequent BLS / BEA / Federal Reserve releases.

Why this matters for macro specifically: a model trained on retrospectively
revised CPI knows things about how BLS will eventually adjust seasonal
factors. CPI revisions can be 0.1-0.3 percentage points, which is large
relative to the signal a tactical model would extract from CPI surprises.
The naive FRED endpoint silently leaks these revisions.

The fredapi library provides `get_series_all_releases` which returns the
full vintage history. We translate each vintage into one PITRecord row:
the value FRED published on `realtime_start` describing observation
`date`, with later vintages providing `restated_at` for earlier rows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Iterator, Optional

import pandas as pd

# Lazy import — fredapi imports requests at module import and we want this
# module to be importable in environments without the package.


@dataclass(frozen=True)
class VintagedObservation:
    series_id: str
    observation_date: date
    release_date: date
    value: float


# A representative set of ~50 macro indicators covering the major themes used
# in business / macro analysis. The list is informative — callers can pass any
# series_id; this is the default universe.
DEFAULT_SERIES = (
    # Inflation
    "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE",
    # Activity
    "GDP", "GDPC1", "INDPRO", "TCU",
    # Labor
    "UNRATE", "PAYEMS", "CIVPART", "EMRATIO",
    # Rates / financial
    "FEDFUNDS", "DGS10", "DGS2", "T10Y2Y", "T10Y3M",
    "DTB3", "AAA", "BAA", "BAMLH0A0HYM2",
    # Money & credit
    "M2SL", "TOTRESNS", "BUSLOANS",
    # Trade / FX
    "EXJPUS", "EXUSEU", "EXCHUS", "DEXBZUS", "DTWEXBGS",
    # Commodities
    "DCOILWTICO", "DCOILBRENTEU", "DHHNGSP", "PALLFNFINDEXM",
    # Markets / sentiment
    "VIXCLS", "SP500", "NASDAQ100", "UMCSENT", "MICH",
    # Housing
    "HOUST", "PERMIT", "CSUSHPISA", "MORTGAGE30US",
    # Business confidence
    "BAA10Y", "AAA10Y",
    # Yield curve (term premium proxies)
    "DGS5", "DGS30",
)


def fetch_vintages(
    series_id: str,
    *,
    api_key: Optional[str] = None,
    realtime_start: Optional[date] = None,
    realtime_end: Optional[date] = None,
) -> Iterator[VintagedObservation]:
    """Yield one observation per (release_date, observation_date) for `series_id`.

    Uses ALFRED via `fredapi.Fred.get_series_all_releases`. Each row in the
    returned DataFrame is the value published on `realtime_start` describing
    observation `date`.
    """
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY") or os.environ.get("FREDAPI_KEY")
    if not key:
        raise RuntimeError(
            "Set FRED_API_KEY env var to access FRED. Get one free at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    fred = Fred(api_key=key)
    df = fred.get_series_all_releases(series_id)
    if df is None or df.empty:
        return
    # fredapi returns columns: date (observation), realtime_start (release), value
    for _, row in df.iterrows():
        obs = pd.Timestamp(row["date"]).date()
        rel = pd.Timestamp(row["realtime_start"]).date()
        if realtime_start is not None and rel < realtime_start:
            continue
        if realtime_end is not None and rel > realtime_end:
            continue
        val = row["value"]
        if pd.isna(val):
            continue
        yield VintagedObservation(
            series_id=series_id,
            observation_date=obs,
            release_date=rel,
            value=float(val),
        )


def fetch_series_info(series_id: str, *, api_key: Optional[str] = None) -> dict:
    """Return basic metadata for the series (title, units, frequency)."""
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY") or os.environ.get("FREDAPI_KEY")
    fred = Fred(api_key=key)
    info = fred.get_series_info(series_id)
    return {
        "title": str(info.get("title", "")),
        "units": str(info.get("units", "")),
        "frequency": str(info.get("frequency", "")),
    }
