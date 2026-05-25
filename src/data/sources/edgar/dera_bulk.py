"""SEC DERA Financial Statement Data Sets — bulk XBRL facts.

DERA publishes ~85-90% of post-2012 US-GAAP XBRL facts as quarterly TSV
dumps. This is *bulk-shaped financials*: one file per quarter, ~50K filings
each, ~5-15M numeric facts. Vastly cheaper than re-pulling per-filing XBRL
instance documents via the EDGAR API (which is rate-limited at 9 req/s and
took 80+ hours for our first attempt at the same scope).

### Source

URL pattern: https://www.sec.gov/files/dera/data/financial-statement-data-sets/{YYYY}q{N}.zip

Coverage: 2009-Q1 → current. The pre-2012 quarters are sparse (XBRL mandate
phase-in); 2012+ is the meaningful full-coverage range.

### Files inside each zip

- num.txt: numeric facts (adsh, tag, version, ddate, qtrs, uom, segments,
  coreg, value, footnote)
- sub.txt: submissions/filings (adsh, cik, name, ..., form, period, fy, fp,
  filed, accepted, ...)
- tag.txt: tag definitions (skipped — we resolve via `version`)
- pre.txt: presentation linkbase (skipped — we don't render statements)

### Critical: `segments` preserves XBRL dimensions

The `segments` column in num.txt encodes the XBRL context axis/member pairs
that the broken legacy `companyfacts` ingest dropped, collapsing segment-
level facts onto consolidated facts. Example: `"EquityComponents=
RetainedEarnings;ProductOrService=iPhone;"`. We parse this into the
canonical `dimensions_json` field so the PIT engine treats dimensional
facts as distinct logical rows (per Stage 2 design).
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

SEC_USER_AGENT = "BWM Research rohit@dragonscale.ai"
DERA_URL_TMPL = (
    "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{q}.zip"
)


@dataclass(frozen=True)
class DeraQuarter:
    """Identifier for a DERA quarterly bundle."""

    year: int
    quarter: int  # 1..4

    def __str__(self) -> str:
        return f"{self.year}Q{self.quarter}"

    @property
    def url(self) -> str:
        return DERA_URL_TMPL.format(year=self.year, q=self.quarter)


def enumerate_quarters(start_year: int, end_year: int) -> list[DeraQuarter]:
    """All DERA quarters in [start_year, end_year] inclusive. ~52 quarters total."""
    out: list[DeraQuarter] = []
    for y in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            out.append(DeraQuarter(y, q))
    return out


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def download_quarter(q: DeraQuarter) -> bytes:
    """Fetch one quarterly zip. Retries on transient errors."""
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip"}
    r = requests.get(q.url, headers=headers, timeout=300)
    if r.status_code == 404:
        # Some pre-2012 quarters genuinely don't exist (XBRL pre-mandate); raise
        # a typed sentinel so callers can skip cleanly instead of crashing.
        raise FileNotFoundError(f"DERA quarter not available: {q}")
    r.raise_for_status()
    return r.content


def _parse_segments(segments) -> str:
    """Convert DERA's `axis=member;axis=member;` string to canonical JSON.

    Empty / null input → empty string (consolidated fact, no dimensions).
    Otherwise → JSON object with axis names as keys, e.g.
    `{"EquityComponents": "RetainedEarnings", "ProductOrService": "iPhone"}`.

    Stored as a string rather than a real JSON column so it slots into the
    existing FinancialFact.dimensions_json field without schema churn.

    NA-safe: handles None, np.nan, pd.NA via pd.isna() check first.
    """
    if pd.isna(segments):
        return ""
    s = str(segments)
    if not s or s.lower() == "nan":
        return ""
    parts = [p for p in s.strip(";").split(";") if "=" in p]
    if not parts:
        return ""
    pairs = {}
    for p in parts:
        k, _, v = p.partition("=")
        pairs[k.strip()] = v.strip()
    return json.dumps(pairs, separators=(",", ":"), sort_keys=True)


def _ddate_to_date(ddate) -> date | None:
    """DERA ddate is YYYYMMDD integer (or string). NA-safe."""
    if pd.isna(ddate):
        return None
    try:
        s = str(int(float(ddate)))
    except (TypeError, ValueError):
        return None
    if len(s) != 8:
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def _concept_from_tag(tag, version) -> str:
    """Build a fully-qualified concept name.

    DERA stores tag = "Revenues" and version = "us-gaap/2024". We emit
    "us-gaap:Revenues" to match the existing FinancialFact convention.
    Custom company-extension tags (where version is a CIK or other non-
    standard prefix without a `/`) are emitted as "custom:{tag}" to avoid
    namespace collision.

    NA-safe: pandas-NA inputs (either field) → "custom:unknown".
    """
    if pd.isna(tag) or pd.isna(version):
        return "custom:unknown"
    tag_s, ver_s = str(tag), str(version)
    if "/" not in ver_s:
        return f"custom:{tag_s}"
    namespace = ver_s.split("/", 1)[0]
    return f"{namespace}:{tag_s}"


def parse_quarter(
    zip_bytes: bytes,
    keep_ciks: set[str] | None = None,
) -> pd.DataFrame:
    """Parse one quarterly DERA zip into FinancialFact-shaped rows.

    Joins num.txt × sub.txt on adsh and emits one row per numeric fact. If
    `keep_ciks` is provided, only filings for those CIKs are retained
    (saves memory + time when ingesting against the registry).

    Returns a DataFrame with columns matching FinancialFact + PITRecord v3:
        entity_id, modality, effective_date, availability_date, restated_at,
        source, source_ref, concept, taxonomy, fiscal_year, fiscal_period,
        unit, context_id, dimensions_json, value, + v3 PIT fields (filled
        by PITEngine.write via schema defaults).
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open("sub.txt") as f:
            sub = pd.read_csv(
                f,
                sep="\t",
                dtype={"cik": "string", "adsh": "string", "form": "string",
                       "fy": "Int64", "fp": "string", "filed": "Int64"},
                usecols=["adsh", "cik", "form", "fy", "fp", "filed"],
                low_memory=False,
            )
        with zf.open("num.txt") as f:
            num = pd.read_csv(
                f,
                sep="\t",
                dtype={
                    "adsh": "string",
                    "tag": "string",
                    "version": "string",
                    "ddate": "Int64",
                    "qtrs": "Int64",
                    "uom": "string",
                    "segments": "string",
                    "value": "Float64",
                },
                usecols=["adsh", "tag", "version", "ddate", "qtrs", "uom",
                         "segments", "value"],
                low_memory=False,
            )

    # Filter to registered CIKs early to shrink the join.
    if keep_ciks is not None:
        # DERA stores CIK as integer-string without zero-padding. Our registry
        # uses zero-padded 10-digit strings; normalize both to integer for
        # comparison, then re-pad downstream.
        keep_int = {int(c) for c in keep_ciks}
        sub = sub[sub["cik"].astype("Int64").isin(keep_int)]
    if sub.empty:
        return pd.DataFrame()

    # Drop facts whose filing isn't in our filter set.
    num = num[num["adsh"].isin(set(sub["adsh"]))]
    if num.empty:
        return pd.DataFrame()

    merged = num.merge(sub, on="adsh", how="inner")

    # Build canonical columns
    cik_padded = merged["cik"].astype("Int64").astype(str).str.zfill(10)
    effective_dates = merged["ddate"].apply(_ddate_to_date)
    out = pd.DataFrame({
        # PITRecord-required
        "entity_id": "us-cik-" + cik_padded,
        "modality": "financials",
        "effective_date": effective_dates,
        "availability_date": merged["filed"].apply(_ddate_to_date),
        "restated_at": pd.NaT,
        "source": "sec_edgar:dera",
        "source_ref": merged["adsh"],
        # FinancialFact-required
        "concept": [
            _concept_from_tag(t, v) for t, v in zip(merged["tag"], merged["version"])
        ],
        # NA-safe taxonomy: split on '/' and fallback to 'custom' when version
        # is missing (rare — only a few hundred rows per quarter).
        "taxonomy": merged["version"].fillna("custom").str.split("/").str[0],
        "fiscal_year": merged["fy"].astype("Int64"),
        "fiscal_period": merged["fp"].fillna(""),
        "unit": merged["uom"].fillna(""),
        "value": merged["value"],
        # period_end == effective_date for both instant and duration facts in
        # DERA's encoding (ddate is the period end). period_start is left null;
        # callers that need it can derive from (period_end, qtrs).
        "period_end": effective_dates,
        "form": merged["form"].fillna(""),
        "accession": merged["adsh"],
        # context_id isn't published by DERA per-fact; synthesize a stable one
        # from (qtrs, ddate) so dedup within (entity, fact) still distinguishes
        # instant vs duration variants of the same concept on the same date.
        "context_id": (
            merged["qtrs"].fillna(-1).astype(int).astype(str) + "_"
            + merged["ddate"].fillna(0).astype(int).astype(str)
        ),
        "dimensions_json": merged["segments"].apply(_parse_segments),
        # Override the schema's default perspective (xbrl_instance) since DERA
        # is a different ingest path with the same source_certainty=1.0.
        "perspective": "sec_edgar:dera",
    })

    # Drop rows where the date parse failed (corrupt source data, rare).
    out = out[out["effective_date"].notna() & out["availability_date"].notna()]
    return out.reset_index(drop=True)


def iter_quarters(
    start: DeraQuarter, end: DeraQuarter, keep_ciks: set[str] | None = None,
) -> Iterator[tuple[DeraQuarter, pd.DataFrame]]:
    """Stream parsed DataFrames for each quarter in [start, end].

    Skips quarters that 404 (pre-2012 holes). Caller is responsible for
    writing each batch to canonical storage and updating watermarks.
    """
    for q in enumerate_quarters(start.year, end.year):
        if (q.year, q.quarter) < (start.year, start.quarter):
            continue
        if (q.year, q.quarter) > (end.year, end.quarter):
            break
        try:
            zip_bytes = download_quarter(q)
        except FileNotFoundError:
            continue
        df = parse_quarter(zip_bytes, keep_ciks=keep_ciks)
        yield q, df
