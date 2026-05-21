"""Per-company submissions index from SEC's `data.sec.gov/submissions/CIK*.json`.

The submissions JSON has a fixed structure:

    {
      "cik": "320193",
      "name": "Apple Inc.",
      "tickers": ["AAPL"],
      "filings": {
        "recent": {
          "accessionNumber": [...],
          "filingDate":      [...],
          "form":            [...],
          "primaryDocument": [...],
          "reportDate":      [...],
          ...
        },
        "files": [ {"name": "CIK0000320193-submissions-001.json"}, ... ]
      }
    }

`filings.recent` is the most recent ~1000 filings inline. Older filings are
paginated into separate JSON files referenced by `filings.files[].name`.

This module returns a flat list of `Filing` records. Per-filing XBRL parsing
is in `xbrl.py`; primary-doc HTML extraction is in `filings_text.py` (Stage 4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from data.sources.edgar.client import EdgarClient

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"

# Form classes we ingest. The constants are exposed so callers can specify
# what they want without re-declaring strings.
XBRL_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F"})
FORM4_FORMS = frozenset({"4", "4/A"})
TEXT_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A"})
ALL_INGESTED_FORMS = XBRL_FORMS | FORM4_FORMS | TEXT_FORMS


@dataclass(frozen=True)
class Filing:
    cik: str            # 10-digit zero-padded
    accession: str      # "0000320193-20-000052"
    form: str           # "10-K", "10-Q", ...
    filed: date         # filing/availability date
    report_date: date   # period end (effective_date for the period)
    primary_doc: str    # e.g. "aapl-20200328.htm"

    @property
    def accession_dashless(self) -> str:
        return self.accession.replace("-", "")

    @property
    def primary_doc_url(self) -> str:
        # SEC URL form: /Archives/edgar/data/{cik_unpadded}/{accn_dashless}/{doc}
        cik_unpadded = self.cik.lstrip("0") or "0"
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_unpadded}/{self.accession_dashless}/{self.primary_doc}"
        )

    @property
    def primary_doc_raw_url(self) -> str:
        """URL to the *raw* primary document, bypassing SEC's XSL renderers.

        For Form 4/3/5 filings, `primaryDocument` often points at a path like
        `xslF345X06/form4.xml` which returns the HTML rendering. The raw XML
        sits at the same accession directory without the `xsl*/` segment.
        """
        cik_unpadded = self.cik.lstrip("0") or "0"
        doc = self.primary_doc
        # Strip any leading `xsl.../` segment.
        if "/" in doc:
            head, tail = doc.split("/", 1)
            if head.lower().startswith("xsl"):
                doc = tail
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_unpadded}/{self.accession_dashless}/{doc}"
        )


def _parse_block(cik: str, block: dict, forms_filter: frozenset[str]) -> list[Filing]:
    """`block` is one of the per-page filings dicts with parallel arrays.

    Each row at index `i` represents one filing; we project the columns we
    care about and skip rows whose `form` isn't in `forms_filter`.
    """
    accns = block.get("accessionNumber", [])
    forms = block.get("form", [])
    fileds = block.get("filingDate", [])
    reports = block.get("reportDate", [])
    primaries = block.get("primaryDocument", [])
    out: list[Filing] = []
    for i in range(len(accns)):
        form = forms[i]
        if form not in forms_filter:
            continue
        rd_s = reports[i] or fileds[i]
        try:
            filed = date.fromisoformat(fileds[i])
            rd = date.fromisoformat(rd_s)
        except (TypeError, ValueError):
            continue
        out.append(
            Filing(
                cik=cik.zfill(10),
                accession=accns[i],
                form=form,
                filed=filed,
                report_date=rd,
                primary_doc=primaries[i] or "",
            )
        )
    return out


async def list_filings(
    client: EdgarClient,
    cik: str,
    *,
    forms: Optional[frozenset[str]] = None,
    include_archived: bool = False,
) -> list[Filing]:
    """Return all filings for a CIK matching `forms` (default: XBRL_FORMS).

    By default, only the inline `filings.recent` block is consumed. SEC's
    submissions API caps that at ~1000 most-recent filings, which covers the
    last ~25 years for actively-filing issuers. Pass `include_archived=True`
    to follow the paginated archive (slower; more API calls).
    """
    cik10 = cik.zfill(10)
    forms_filter = forms if forms is not None else XBRL_FORMS
    body = await client.get_bytes(SUBMISSIONS_URL.format(cik10=cik10))
    payload = json.loads(body)
    filings_blob = payload.get("filings", {})

    out = _parse_block(cik10, filings_blob.get("recent", {}), forms_filter)

    if include_archived:
        for page in filings_blob.get("files", []) or []:
            name = page.get("name")
            if not name:
                continue
            page_body = await client.get_bytes(SUBMISSIONS_PAGE_URL.format(name=name))
            page_payload = json.loads(page_body)
            out.extend(_parse_block(cik10, page_payload, forms_filter))

    return out


def filter_by_form(filings: Iterable[Filing], forms: Iterable[str]) -> list[Filing]:
    s = set(forms)
    return [f for f in filings if f.form in s]
