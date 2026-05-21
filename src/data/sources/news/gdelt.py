"""GDELT 2.0 Events Database reader.

GDELT 2.0 publishes a CSV file every 15 minutes at:
    http://data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMM}00.export.CSV.zip

Each row is one event. We don't subscribe to the live stream from this
connector; that's a separate operational concern. The functions here
parse a single CSV file's rows and yield structured records.

CSV schema documented at: https://www.gdeltproject.org/data.html#documentation
The columns relevant to BWM:
  0  GlobalEventID
  1  Day                              YYYYMMDD
  ...
  5  Actor1Code, Actor1Name           CAMEO actor identifiers
  10 Actor2Code, Actor2Name
  26 EventCode                        CAMEO event code (e.g. 1822)
  27 EventRootCode                    First 2 digits of EventCode
  28 QuadClass                        1-4
  29 GoldsteinScale                   -10 to +10
  30 NumMentions
  31 NumSources
  32 NumArticles
  33 AvgTone
  ...
  57 SOURCEURL                        First source URL
  58 DATEADDED                        YYYYMMDDHHMMSS — when GDELT indexed it
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterator, Optional


@dataclass(frozen=True)
class ParsedNewsEvent:
    global_event_id: str
    event_date: date
    date_added: date
    actor1_name: str
    actor2_name: str
    cameo_event_code: str
    event_root_code: str
    quad_class: int
    goldstein_scale: float
    n_mentions: int
    avg_tone: float
    source_url: str


# Column indices in GDELT 2.0 export CSV (0-indexed).
_COL_GLOBAL_ID = 0
_COL_DAY = 1
_COL_ACTOR1_NAME = 6
_COL_ACTOR2_NAME = 16
_COL_EVENT_CODE = 26
_COL_EVENT_ROOT = 27
_COL_QUAD_CLASS = 28
_COL_GOLDSTEIN = 29
_COL_NUM_MENTIONS = 31
_COL_AVG_TONE = 34
_COL_SOURCE_URL = 57
_COL_DATE_ADDED = 59  # column 60 in some docs (DATEADDED)
# GDELT 2.0 has 61 columns total.


def _parse_yyyymmdd(s: str) -> Optional[date]:
    s = (s or "").strip()
    if len(s) < 8 or not s[:8].isdigit():
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def _parse_yyyymmddhhmmss(s: str) -> Optional[date]:
    s = (s or "").strip()
    if len(s) < 8:
        return None
    return _parse_yyyymmdd(s[:8])


def _to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _to_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def parse_gdelt_csv(body: bytes) -> Iterator[ParsedNewsEvent]:
    """Parse a GDELT 2.0 events CSV file body. Yields one record per row.

    GDELT exports are TSV (tab-separated despite the .CSV extension).
    """
    text = body.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < 60:
            continue
        global_id = (row[_COL_GLOBAL_ID] or "").strip()
        event_date = _parse_yyyymmdd(row[_COL_DAY])
        date_added = _parse_yyyymmddhhmmss(row[_COL_DATE_ADDED])
        if not (global_id and event_date and date_added):
            continue
        yield ParsedNewsEvent(
            global_event_id=global_id,
            event_date=event_date,
            date_added=date_added,
            actor1_name=(row[_COL_ACTOR1_NAME] or "").strip(),
            actor2_name=(row[_COL_ACTOR2_NAME] or "").strip(),
            cameo_event_code=(row[_COL_EVENT_CODE] or "").strip(),
            event_root_code=(row[_COL_EVENT_ROOT] or "").strip(),
            quad_class=_to_int(row[_COL_QUAD_CLASS]),
            goldstein_scale=_to_float(row[_COL_GOLDSTEIN]),
            n_mentions=_to_int(row[_COL_NUM_MENTIONS], 1),
            avg_tone=_to_float(row[_COL_AVG_TONE]),
            source_url=(row[_COL_SOURCE_URL] or "").strip(),
        )
