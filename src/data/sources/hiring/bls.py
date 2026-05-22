"""BLS JOLTS API client (api.bls.gov v2).

BLS exposes labor data via JSON POST at https://api.bls.gov/publicAPI/v2/.
JOLTS series IDs follow the pattern `JTU{industry}{geographic}{level/rate}NSA`
or `JTS{...}SA` for seasonally adjusted. Example: `JTS000000000000000JOL`
is total US nonfarm job openings, SA.

For BWM's context-feature use, we want per-NAICS-3-digit series. The
seriesID structure is documented at: https://download.bls.gov/pub/time.series/jt/

This connector is intentionally light: takes a list of seriesIDs and a
date range, returns one observation per (series, month). The CLI wraps it
with a curated NAICS-industry seriesID list.

The v2 API allows up to 50 series per request (25 without a key, 50 with).
No-key access is rate-limited to 25 requests / day (sufficient for monthly
incremental refresh).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Iterator, Optional

import requests

BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


@dataclass(frozen=True)
class ParsedJOLTS:
    series_id: str
    period_end: date     # last day of the observation month
    release_date: date   # BLS release date (~6 weeks after period_end)
    value: float
    footnotes: str


def _last_day_of_month(y: int, m: int) -> date:
    if m == 12:
        return date(y, 12, 31)
    next_m = date(y, m + 1, 1)
    return date(next_m.year, next_m.month, 1).replace(day=1).replace(day=1)


def _approx_release_date(period_end: date) -> date:
    """BLS JOLTS releases come out ~6 weeks after the reference month.

    The exact schedule is published; for PIT we use 'first Tuesday ~42 days
    after period_end' as a deterministic proxy. Real production would fetch
    the exact release date from BLS's release schedule API.
    """
    # 42 days after the period_end, rounded to the next Tuesday.
    from datetime import timedelta
    target = period_end + timedelta(days=42)
    # Tuesday = 1 (Monday is 0)
    days_to_tue = (1 - target.weekday()) % 7
    return target + timedelta(days=days_to_tue)


def fetch_jolts_series(
    series_ids: list[str],
    *,
    start_year: int,
    end_year: int,
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
) -> Iterator[ParsedJOLTS]:
    """POST to BLS API, yield one ParsedJOLTS per (series, month).

    BLS responds with monthly data. We compute the period_end as the last
    day of the (year, M01..M12) month and approximate release_date with
    `_approx_release_date`.
    """
    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if api_key:
        payload["registrationkey"] = api_key
    headers = {"Content-Type": "application/json"}
    resp = requests.post(BLS_URL, json=payload, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "REQUEST_SUCCEEDED":
        msgs = data.get("message") or []
        raise RuntimeError(f"BLS API error: {msgs}")
    for series in data.get("Results", {}).get("series", []):
        sid = series.get("seriesID")
        for obs in series.get("data", []):
            year = int(obs["year"])
            period = obs["period"]  # 'M01' through 'M12'
            if not period.startswith("M"):
                continue
            month = int(period[1:])
            # last day of month
            if month == 12:
                period_end = date(year, 12, 31)
            else:
                from datetime import timedelta
                next_month_first = date(year, month + 1, 1)
                period_end = next_month_first - timedelta(days=1)
            try:
                value = float(obs.get("value", "").replace(",", ""))
            except (ValueError, AttributeError):
                continue
            yield ParsedJOLTS(
                series_id=sid,
                period_end=period_end,
                release_date=_approx_release_date(period_end),
                value=value,
                footnotes=str(obs.get("footnotes", "")),
            )


# Curated NAICS supersector JOLTS series. seriesID encoding (BLS JOLTS):
#   JT + Seasonal(S/U) + Industry(6) + State(2) + Area(5) + SizeClass(2) + DataElement(3)
# = 21 chars total. Example: JTS000000000000000JOL = total nonfarm job openings, SA.
# DataElement: JOL=Job Openings, HIL=Hires, TSL=Total Separations,
#              QUL=Quits, LDL=Layoffs/Discharges (all are *Levels*)
# State 00 + Area 00000 + SizeClass 00 = total US, all areas, all sizes.
# Industry codes are JOLTS supersector codes (note: NOT NAICS 3-digit;
# JOLTS uses its own coding listed at https://download.bls.gov/pub/time.series/jt/jt.industry).
DEFAULT_NAICS_INDUSTRIES = [
    ("000000", "Total nonfarm"),
    ("110099", "Mining and logging"),
    ("230000", "Construction"),
    ("300000", "Manufacturing"),
    ("400000", "Trade, transportation, and utilities"),
    ("510000", "Information"),
    ("520000", "Financial activities"),
    ("540099", "Professional and business services"),
    ("600000", "Education and health services"),
    ("700000", "Leisure and hospitality"),
]
MEASURES = [("JOL", "job_openings"), ("HIL", "hires"), ("TSL", "total_separations")]


def build_default_seriesids() -> list[tuple[str, str, str]]:
    """Return [(series_id, industry_code, measure_name), ...] for the curated set.

    seriesID format: JTS + industry(6) + 00 + 00000 + 00 + element(3) = 21 chars.
    """
    out = []
    for naics, _ in DEFAULT_NAICS_INDUSTRIES:
        for code, name in MEASURES:
            sid = f"JTS{naics}000000000{code}"  # JTS + 6 industry + 00 state + 00000 area + 00 sizeclass + 3 element
            out.append((sid, naics, name))
    return out
