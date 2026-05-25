"""Shared pytest fixtures for BWM tests.

Conventions:
- All tests are filesystem-only (no live network); fixtures build synthetic
  inputs that exercise the same parse + write paths as real data
- `local_storage` and `lance_storage` fixtures give a tmp-rooted Storage
  backend so tests don't pollute .data/
- `sample_financial_rows` is the canonical PIT restatement scenario used by
  multiple tests
"""
from __future__ import annotations

import io
import os
import tarfile
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# Make `src/` importable from tests without an editable install.
import sys
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------- storage backends ----------

@pytest.fixture
def local_storage(tmp_path):
    """LocalStorage rooted at a pytest tmp dir. Default backend (parquet)."""
    from data.storage.backend import LocalStorage
    return LocalStorage(tmp_path)


@pytest.fixture
def lance_storage(tmp_path):
    """LanceStorage rooted at a pytest tmp dir."""
    from data.storage.lance_backend import LanceStorage
    return LanceStorage(tmp_path)


@pytest.fixture(params=["parquet", "lance"])
def any_storage(request, tmp_path):
    """Parameterized — runs the test against both backends."""
    if request.param == "parquet":
        from data.storage.backend import LocalStorage
        return LocalStorage(tmp_path)
    from data.storage.lance_backend import LanceStorage
    return LanceStorage(tmp_path)


# ---------- canonical PIT restatement scenario ----------

@pytest.fixture
def aapl_restatement_rows():
    """Two filings: an original at T1 and a restatement at T2 for the same
    (entity, concept, period). The PIT engine should return the original
    for as_of in [T1, T2) and the restatement for as_of >= T2.
    """
    return pd.DataFrame([
        {
            "entity_id": "us-cik-0000320193",
            "modality": "financials",
            "effective_date": date(2023, 12, 31),
            "availability_date": date(2024, 1, 15),
            "restated_at": None,
            "source": "sec_edgar",
            "source_ref": "0000320193-24-000001",
            "concept": "us-gaap:Revenues",
            "taxonomy": "us-gaap",
            "fiscal_year": 2023,
            "fiscal_period": "Q4",
            "unit": "USD",
            "context_id": "ctx1",
            "dimensions_json": "",
            "value": 100.0,
        },
        {
            "entity_id": "us-cik-0000320193",
            "modality": "financials",
            "effective_date": date(2023, 12, 31),
            "availability_date": date(2024, 2, 15),
            "restated_at": None,
            "source": "sec_edgar",
            "source_ref": "0000320193-24-000002",
            "concept": "us-gaap:Revenues",
            "taxonomy": "us-gaap",
            "fiscal_year": 2023,
            "fiscal_period": "Q4",
            "unit": "USD",
            "context_id": "ctx1",
            "dimensions_json": "",
            "value": 110.0,  # restated
        },
    ])


# ---------- synthetic DERA zip ----------

@pytest.fixture
def synthetic_dera_zip() -> bytes:
    """Build a minimal DERA-shaped zip with 2 filings and 5 facts.

    Schema deliberately matches the real DERA num.txt / sub.txt so the
    parser exercises the same code paths as a live download.
    Includes one row with `segments` to verify dimension preservation
    and one row with NA values to verify the NA-safe parse path.
    """
    sub_lines = [
        "adsh\tcik\tname\tform\tfy\tfp\tfiled",
        "0000320193-24-000001\t320193\tAPPLE INC\t10-K\t2023\tFY\t20240115",
        "0000320193-24-000002\t320193\tAPPLE INC\t10-K/A\t2023\tFY\t20240215",
        "0000000999-24-000001\t999\tIGNORED CO\t10-K\t2023\tFY\t20240115",  # not in keep_ciks
    ]
    num_lines = [
        "adsh\ttag\tversion\tddate\tqtrs\tuom\tsegments\tcoreg\tvalue\tfootnote",
        # Original Revenues (consolidated)
        "0000320193-24-000001\tRevenues\tus-gaap/2024\t20231231\t1\tUSD\t\t\t100000000.0\t",
        # Restated Revenues (different accession, later filing)
        "0000320193-24-000002\tRevenues\tus-gaap/2024\t20231231\t1\tUSD\t\t\t110000000.0\t",
        # Segment-level Revenues (dimensions preserved)
        "0000320193-24-000001\tRevenues\tus-gaap/2024\t20231231\t1\tUSD\tProductOrService=iPhone;\t\t70000000.0\t",
        # NA segments + NA version → triggers NA-safe paths
        "0000320193-24-000001\tCustomConcept\t\t20231231\t1\tUSD\t\t\t1.0\t",
        # Ignored CIK row (filtered out)
        "0000000999-24-000001\tRevenues\tus-gaap/2024\t20231231\t1\tUSD\t\t\t9.0\t",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sub.txt", "\n".join(sub_lines))
        zf.writestr("num.txt", "\n".join(num_lines))
        # pre.txt + tag.txt aren't read by our parser; include empty files
        zf.writestr("pre.txt", "")
        zf.writestr("tag.txt", "")
    return buf.getvalue()


# ---------- synthetic SEC Feed tarball ----------

def _make_nc_filing(
    accession: str,
    cik: str,
    form: str,
    period: str = "20231231",
    filed: str = "20240115",
    documents: list[tuple[str, str, str, str]] | None = None,
) -> bytes:
    """Build a minimal SEC SUBMISSION-format .nc filing body."""
    docs = documents or [
        ("10-K", "primary.htm", "FORM 10-K", "<html><body>Hello</body></html>"),
    ]
    lines = [
        "<SUBMISSION>",
        f"<ACCESSION-NUMBER>{accession}",
        f"<TYPE>{form}",
        f"<PERIOD>{period}",
        f"<FILING-DATE>{filed}",
        "<FILER>",
        "<COMPANY-DATA>",
        f"<CIK>{cik}",
        "<CONFORMED-NAME>TEST CO",
        "</COMPANY-DATA>",
        "</FILER>",
    ]
    for i, (dtype, fname, descr, body) in enumerate(docs, start=1):
        lines.extend([
            "<DOCUMENT>",
            f"<TYPE>{dtype}",
            f"<SEQUENCE>{i}",
            f"<FILENAME>{fname}",
            f"<DESCRIPTION>{descr}",
            "<TEXT>",
            body,
            "</TEXT>",
            "</DOCUMENT>",
        ])
    lines.append("</SUBMISSION>")
    return "\n".join(lines).encode("latin-1")


@pytest.fixture
def synthetic_feed_tarball() -> bytes:
    """Build a daily Feed-shaped tar.gz with 3 filings:
    - AAPL 10-K with primary doc + XBRL instance (XBRL must be skipped)
    - MSFT 8-K with primary doc only
    - IGNORED-CO Form 4 (filtered out when keep_ciks excludes 999)
    """
    files = {
        "0000320193-24-000001.nc": _make_nc_filing(
            "0000320193-24-000001", "0000320193", "10-K",
            documents=[
                ("10-K", "aapl-10k.htm", "FORM 10-K", "<html><body>10-K body</body></html>"),
                ("EX-101.INS", "aapl-20231231.xml", "XBRL INSTANCE",
                 "<xbrl>...</xbrl>"),  # should be skipped
                ("EX-21", "subs.htm", "SUBSIDIARIES", "<html>List of subs</html>"),
            ],
        ),
        "0000789019-24-000001.nc": _make_nc_filing(
            "0000789019-24-000001", "0000789019", "8-K",
        ),
        "0000000999-24-000001.nc": _make_nc_filing(
            "0000000999-24-000001", "0000000999", "4",
        ),
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


# ---------- minimal entity registry ----------

@pytest.fixture
def seeded_registry(local_storage):
    """Write a 3-CIK registry parquet to local_storage so tests that hit
    `EntityRegistry.load()` see something.
    """
    df = pd.DataFrame({
        "cik": ["320193", "789019", "1318605"],
        "ticker": ["AAPL", "MSFT", "TSLA"],
        "name": ["APPLE INC", "MICROSOFT CORP", "TESLA INC"],
        "entity_id": ["us-cik-0000320193", "us-cik-0000789019", "us-cik-0001318605"],
    })
    local_storage.write_table("entities/registry", df)
    return local_storage
