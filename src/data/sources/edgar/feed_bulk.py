"""SEC daily Feed bulk: tar.gz archives of every filing of every day.

Replaces the rate-limited per-filing EDGAR API fetch path. Each daily
tarball at `Archives/Feed/{YYYY}/QTR{N}/{YYYYMMDD}.nc.tar.gz` contains
every filing accepted by SEC that day in their internal "SUBMISSION"
container format.

### File layout inside a daily tarball

    ./{accession}.nc                # one per filing; format below
    ./{accession}.corr01            # correction files (keep them; SEC's audit trail)

Each `.nc` file is SGML-like:

    <SUBMISSION>
    <ACCESSION-NUMBER>0001193125-15-011275
    <TYPE>10-D
    <PERIOD>20141231
    <FILING-DATE>20150115
    <FILER>
    <COMPANY-DATA>
    <CIK>0001622117
    ...
    </COMPANY-DATA>
    ...
    </FILER>
    <DOCUMENT>
    <TYPE>10-D
    <SEQUENCE>1
    <FILENAME>d851261d10d.htm
    <TEXT>
    ... raw body bytes (HTML, XML, txt, uuencoded blob) ...
    </TEXT>
    </DOCUMENT>
    ... more DOCUMENT blocks ...

### Our extraction

For each filing:
1. Parse the SUBMISSION header into `FeedFiling` (form, primary CIK, period,
   filed date, all FILER CIKs)
2. Skip if no FILER CIK is in our registry (efficient filter — drops ~70%)
3. Extract every `<DOCUMENT>` block as a separate FeedDocument row
4. Filter out the XBRL family of documents (instance, schema, linkbase,
   rendered tables, financial Excel) — these are covered by DERA; keep
   the primary doc + Item-level exhibits + miscellaneous attachments

The "keep everything else" stance is per the user's explicit decision: we
mirror SEC narrative content faithfully and let downstream consumers
filter. Storage cost is rounding-error.

### Output

FeedFiling-level metadata + per-document text/bytes → `canonical/filings_full.lance`
partitioned per CIK / per year. One Lance row per (accession, document_name).
"""
from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterator

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from data.sources.edgar.form4 import ParsedForm4, parse_form4_filing

SEC_USER_AGENT = "BWM Research rohit@dragonscale.ai"
FEED_URL_TMPL = (
    "https://www.sec.gov/Archives/edgar/Feed/{year}/QTR{q}/{yyyymmdd}.nc.tar.gz"
)

# Document types covered by DERA — we skip these so filings_full isn't
# polluted with XBRL artifacts that have a better structured home.
_XBRL_DOC_TYPES = {
    "EX-101.INS",       # XBRL instance
    "EX-101.SCH",       # XBRL schema
    "EX-101.CAL", "EX-101.DEF", "EX-101.LAB", "EX-101.PRE",  # XBRL linkbases
    "EX-100.INS", "EX-100.SCH",  # legacy XBRL
}

# Headline header fields we extract from the SUBMISSION block.
_HEADER_FIELDS = {
    "accession_number": "ACCESSION-NUMBER",
    "form": "TYPE",
    "period": "PERIOD",
    "filed_date": "FILING-DATE",
    "accepted_datetime": "DATE-OF-FILING-DATE-CHANGE",
}

_TAG_VALUE_RE = re.compile(r"^<([A-Z][A-Z0-9-]*)>(.*?)$", re.MULTILINE)


@dataclass(frozen=True)
class FeedDocument:
    accession: str
    sequence: int
    doc_type: str
    filename: str
    description: str
    content_bytes: bytes        # raw body (may be HTML/XML/text/uuencoded)
    n_chars: int                # len of content_bytes


@dataclass
class FeedFiling:
    accession: str
    form: str
    period: date | None
    filed_date: date | None
    primary_cik: str            # 10-digit zero-padded
    all_ciks: list[str] = field(default_factory=list)
    documents: list[FeedDocument] = field(default_factory=list)


@dataclass(frozen=True)
class FeedDay:
    day: date

    @property
    def quarter(self) -> int:
        return (self.day.month - 1) // 3 + 1

    @property
    def url(self) -> str:
        return FEED_URL_TMPL.format(
            year=self.day.year, q=self.quarter,
            yyyymmdd=self.day.strftime("%Y%m%d"),
        )


def _parse_yyyymmdd(s: str | None) -> date | None:
    if not s:
        return None
    s = s.strip()
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def _parse_submission_header(header_text: str) -> dict:
    """Pull headline fields and FILER CIK list from the SUBMISSION SGML header.

    Multi-FILER filings (e.g. 13F co-filings) yield multiple CIKs; the first
    one in declaration order is treated as the "primary" for partition
    purposes. The full list is preserved on FeedFiling.all_ciks so any of
    the registrants can be queried later.
    """
    out: dict = {"all_ciks": []}
    for tag, val in _TAG_VALUE_RE.findall(header_text):
        val = val.strip()
        if tag == "ACCESSION-NUMBER":
            out["accession_number"] = val
        elif tag == "TYPE":
            out.setdefault("form", val)
        elif tag == "PERIOD":
            out["period"] = _parse_yyyymmdd(val)
        elif tag == "FILING-DATE":
            out["filed_date"] = _parse_yyyymmdd(val)
        elif tag == "CIK":
            # Every <FILER>/<COMPANY-DATA>/<CIK> block contributes one.
            if val.isdigit():
                out["all_ciks"].append(val.zfill(10))
    return out


def _split_documents(body: str) -> list[tuple[str, str]]:
    """Slice the body into [(metadata_block, text_block), ...] per <DOCUMENT>.

    Within each `<DOCUMENT>...</DOCUMENT>` the structure is:
        <TYPE>... <SEQUENCE>... <FILENAME>... <DESCRIPTION>...
        <TEXT>
        ... actual body ...
        </TEXT>
    The metadata sits before `<TEXT>`; the body is between `<TEXT>` and `</TEXT>`.
    Returns the raw string slices so callers can decide encoding handling.
    """
    out: list[tuple[str, str]] = []
    for match in re.finditer(r"<DOCUMENT>(.*?)</DOCUMENT>", body, re.DOTALL):
        chunk = match.group(1)
        text_match = re.search(r"<TEXT>(.*?)</TEXT>", chunk, re.DOTALL)
        if not text_match:
            continue
        metadata = chunk[: text_match.start()]
        text = text_match.group(1).strip()
        out.append((metadata, text))
    return out


def parse_nc_file(nc_bytes: bytes) -> FeedFiling | None:
    """Parse one .nc filing into a FeedFiling (or None if header is unparseable).

    Encoding: SEC .nc files are nominally ISO-8859-1 / UTF-8 mixed; we decode
    via 'latin-1' (lossless) so even legacy filings with weird bytes don't
    raise. Downstream consumers re-encode HTML/XML bodies as needed.
    """
    text = nc_bytes.decode("latin-1", errors="replace")
    # Header runs from <SUBMISSION> to the first <DOCUMENT>.
    submission_end = text.find("<DOCUMENT>")
    header = text[:submission_end] if submission_end != -1 else text
    body = text[submission_end:] if submission_end != -1 else ""
    hdr = _parse_submission_header(header)
    if "accession_number" not in hdr or not hdr.get("all_ciks"):
        return None
    primary_cik = hdr["all_ciks"][0]
    filing = FeedFiling(
        accession=hdr["accession_number"],
        form=hdr.get("form", ""),
        period=hdr.get("period"),
        filed_date=hdr.get("filed_date"),
        primary_cik=primary_cik,
        all_ciks=hdr["all_ciks"],
    )
    seq = 0
    for metadata, text_block in _split_documents(body):
        # Per-document fields (also SGML-tagged within the metadata slice)
        doc_fields = dict(_TAG_VALUE_RE.findall(metadata))
        doc_type = doc_fields.get("TYPE", "").strip()
        if doc_type in _XBRL_DOC_TYPES:
            continue  # DERA covers these
        try:
            seq = int(doc_fields.get("SEQUENCE", str(seq + 1)).strip())
        except ValueError:
            seq = seq + 1
        filename = doc_fields.get("FILENAME", "").strip()
        description = doc_fields.get("DESCRIPTION", "").strip()
        content = text_block.encode("latin-1", errors="replace")
        filing.documents.append(FeedDocument(
            accession=filing.accession,
            sequence=seq,
            doc_type=doc_type,
            filename=filename,
            description=description,
            content_bytes=content,
            n_chars=len(content),
        ))
    return filing


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def download_day(day: FeedDay) -> bytes:
    """Fetch one daily Feed tarball. Retries on transient errors."""
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip"}
    r = requests.get(day.url, headers=headers, timeout=600)
    if r.status_code == 404:
        raise FileNotFoundError(f"Feed not available for {day.day}")
    r.raise_for_status()
    return r.content


def iter_filings(
    tarball_bytes: bytes,
    keep_ciks: set[str] | None = None,
    keep_forms: set[str] | None = None,
) -> Iterator[FeedFiling]:
    """Stream FeedFiling objects from a daily tarball.

    `keep_ciks` (zero-padded 10-digit strings) filters by registrant CIK
    (matches if ANY of the filing's FILER CIKs is in the set). Default None
    keeps all filings.

    `keep_forms` filters by form type (matches form OR form/A). Default None
    keeps all form types (true "mirror SEC" mode per the design decision).
    """
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.rsplit("/", 1)[-1]
            # Keep .nc and .corr* (corrections are part of audit trail)
            if not (name.endswith(".nc") or ".corr" in name):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                filing = parse_nc_file(f.read())
            except Exception:
                continue
            if filing is None:
                continue
            if keep_ciks is not None and not any(c in keep_ciks for c in filing.all_ciks):
                continue
            if keep_forms is not None:
                form_base = filing.form.rstrip("/A")
                if filing.form not in keep_forms and form_base not in keep_forms:
                    continue
            yield filing


def filings_to_rows(filings: Iterator[FeedFiling]) -> Iterator[dict]:
    """Flatten FeedFiling.documents into one canonical row per (accession, doc).

    Output schema (slots into a Lance dataset for `filings_full` modality):
        cik, year, accession, form, filed_date, period_of_report,
        document_name, document_type, sequence, mime_type, content_bytes,
        content_text, n_chars, sha256, + PIT v3 fields (filled by writer)
    """
    import hashlib

    for f in filings:
        if not f.filed_date:
            continue
        year = f.filed_date.year
        for doc in f.documents:
            mime = _guess_mime(doc.filename, doc.doc_type)
            # Decode text representations only for HTML/XML; binary stays bytes-only.
            text = None
            if mime.startswith("text/") or mime in {"application/xml", "application/xhtml+xml"}:
                try:
                    text = doc.content_bytes.decode("utf-8", errors="replace")
                except Exception:
                    text = doc.content_bytes.decode("latin-1", errors="replace")
            yield {
                "entity_id": f"us-cik-{f.primary_cik}",
                "cik": f.primary_cik,
                "year": year,
                "accession": f.accession,
                "form": f.form,
                "filed_date": f.filed_date,
                "period_of_report": f.period,
                "document_name": doc.filename,
                "document_type": doc.doc_type,
                "sequence": doc.sequence,
                "mime_type": mime,
                "content_bytes": doc.content_bytes,
                "content_text": text,
                "n_chars": doc.n_chars,
                "sha256": hashlib.sha256(doc.content_bytes).hexdigest(),
                "all_ciks_json": ",".join(f.all_ciks),
            }


FORM4_FORMS = ("4", "4/A")


def extract_form4(filing: FeedFiling) -> ParsedForm4 | None:
    """If `filing` is a Form 4 / 4/A, parse its ownership XML into trades.

    Form 4 filings carry the structured `<ownershipDocument>` as one of the
    DOCUMENT blocks (usually a `.xml` primary). We scan documents in order and
    return the first parse that yields at least one trade. Returns None for
    non-Form-4 filings, malformed XML, or filings with zero recognizable
    transactions (e.g. a Form 4 that reports only a position-disclosure).
    """
    if filing.form not in FORM4_FORMS:
        return None
    for doc in filing.documents:
        if b"<ownershipDocument" not in doc.content_bytes:
            continue
        parsed = parse_form4_filing(doc.content_bytes)
        if parsed is not None and parsed.trades:
            return parsed
    return None


def _guess_mime(filename: str, doc_type: str) -> str:
    fn = filename.lower()
    if fn.endswith((".htm", ".html")):
        return "text/html"
    if fn.endswith(".xml"):
        return "application/xml"
    if fn.endswith(".xsd"):
        return "application/xml"
    if fn.endswith(".txt"):
        return "text/plain"
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".gif"):
        return "image/gif"
    if fn.endswith((".xlsx", ".xls")):
        return "application/vnd.ms-excel"
    return "application/octet-stream"
