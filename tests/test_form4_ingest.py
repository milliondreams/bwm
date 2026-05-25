"""Form 4 (insider trade) extraction via the daily Feed ingest path.

These tests exercise the inline Form-4 extraction added to `ingest_feed`:
the bulk feed loop should parse Form 4 / 4/A ownership XML found inside the
daily tarball, write insider_trade rows via PITEngine, and run amendment
supersession when a 4/A is processed.

The fixtures build self-contained synthetic tarballs with real Form-4 XML
bodies so we cover the same parse path as production.
"""
from __future__ import annotations

import io
import tarfile
from datetime import date

import pandas as pd
import pytest

from data.pit.engine import PITEngine, _entity_path
from data.schemas.insider_trade import InsiderTrade
from data.schemas.pit import Modality
from data.sources.edgar.feed_bulk import extract_form4, iter_filings


# A minimal but realistic Form 4 ownership XML. Two transactions: a sale and
# a derivative grant. periodOfReport drives 4/A supersession.
_FORM4_XML_ORIGINAL = b"""<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2024-01-10</periodOfReport>
  <issuer><issuerCik>0000320193</issuerCik></issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001214156</rptOwnerCik>
      <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-01-10</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>185.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>3287000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""

# Amendment: same period_of_report + insider, but the corrected share count
# is 600 (and only one transaction, where the original had one too). After
# supersession, an as-of query past 2024-01-25 should see only the amendment
# row.
_FORM4_XML_AMENDMENT = b"""<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2024-01-10</periodOfReport>
  <issuer><issuerCik>0000320193</issuerCik></issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001214156</rptOwnerCik>
      <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-01-10</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>600</value></transactionShares>
        <transactionPricePerShare><value>185.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>3286900</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


def _make_form4_nc(accession: str, cik: str, form: str, filed: str, xml_body: bytes) -> bytes:
    """Build a SUBMISSION-format .nc filing containing one Form 4 XML doc."""
    lines = [
        "<SUBMISSION>",
        f"<ACCESSION-NUMBER>{accession}",
        f"<TYPE>{form}",
        f"<PERIOD>20240110",
        f"<FILING-DATE>{filed}",
        "<FILER>",
        "<COMPANY-DATA>",
        f"<CIK>{cik}",
        "<CONFORMED-NAME>APPLE INC",
        "</COMPANY-DATA>",
        "</FILER>",
        "<DOCUMENT>",
        "<TYPE>4",
        "<SEQUENCE>1",
        f"<FILENAME>wf-form4_{accession}.xml",
        "<DESCRIPTION>FORM 4",
        "<TEXT>",
    ]
    body = "\n".join(lines).encode("latin-1") + xml_body + b"\n</TEXT>\n</DOCUMENT>\n</SUBMISSION>\n"
    return body


@pytest.fixture
def form4_tarball() -> bytes:
    """Daily tarball with one original Form 4 (filed 2024-01-12) and one 4/A
    amendment (filed 2024-01-25), both for AAPL CIK 320193."""
    files = {
        "0001214156-24-000001.nc": _make_form4_nc(
            "0001214156-24-000001", "0000320193", "4", "20240112",
            _FORM4_XML_ORIGINAL,
        ),
        "0001214156-24-000002.nc": _make_form4_nc(
            "0001214156-24-000002", "0000320193", "4/A", "20240125",
            _FORM4_XML_AMENDMENT,
        ),
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class TestExtractForm4:
    def test_parses_original(self, form4_tarball):
        filings = {f.accession: f for f in iter_filings(form4_tarball)}
        f = filings["0001214156-24-000001"]
        parsed = extract_form4(f)
        assert parsed is not None
        assert parsed.period_of_report == date(2024, 1, 10)
        assert len(parsed.trades) == 1
        t = parsed.trades[0]
        assert t.insider_cik == "0001214156"
        assert t.shares == 500.0
        assert t.transaction_code == "S"
        assert t.is_officer is True

    def test_parses_amendment(self, form4_tarball):
        filings = {f.accession: f for f in iter_filings(form4_tarball)}
        parsed = extract_form4(filings["0001214156-24-000002"])
        assert parsed is not None
        assert parsed.trades[0].shares == 600.0  # corrected count

    def test_non_form4_returns_none(self, synthetic_feed_tarball):
        # The vanilla feed fixture has a 10-K, an 8-K, and a content-less "4"
        # filing. None of them have an ownershipDocument body.
        for f in iter_filings(synthetic_feed_tarball):
            assert extract_form4(f) is None


class TestIngestFeedForm4Integration:
    """End-to-end: run the day-level ingest and verify canonical + supersession."""

    def _patch_feed_download(self, monkeypatch, tarball: bytes):
        # ingest_feed imports `download_day` by name, so patching the source
        # module wouldn't reach it — patch the local binding too.
        from data.cli import ingest_feed as ingest_feed_mod
        from data.sources.edgar import feed_bulk
        monkeypatch.setattr(feed_bulk, "download_day", lambda day: tarball)
        monkeypatch.setattr(ingest_feed_mod, "download_day", lambda day: tarball)

    def test_writes_insider_trades_and_supersedes(
        self, lance_storage, monkeypatch, form4_tarball
    ):
        from data.cli.ingest_feed import _ingest_day
        from data.sources.edgar.state import WatermarkStore

        self._patch_feed_download(monkeypatch, form4_tarball)
        pit = PITEngine(lance_storage)
        wm = WatermarkStore(lance_storage)

        stats = _ingest_day(
            day=date(2024, 1, 25),
            keep_ciks={"0000320193"},
            storage=lance_storage,
            watermark=wm,
            pit=pit,
        )

        assert stats.status == "ok_with_records"
        assert stats.form4_filings == 2
        assert stats.form4_trades == 2  # one per filing
        # The amendment supersedes the original row (same insider + period).
        assert stats.form4_supersedes == 1

        # Verify canonical content via PIT engine.
        path = _entity_path(Modality.INSIDER_TRADES, "us-cik-0000320193")
        assert lance_storage.exists(path)
        df = lance_storage.read_parquet(path)
        assert len(df) == 2
        assert set(df["accession"]) == {
            "0001214156-24-000001",
            "0001214156-24-000002",
        }
        # The original row should be marked restated by the amendment's filed date.
        orig = df[df["accession"] == "0001214156-24-000001"].iloc[0]
        amend = df[df["accession"] == "0001214156-24-000002"].iloc[0]
        assert pd.notna(orig["restated_at"])
        assert pd.to_datetime(orig["restated_at"]).date() == date(2024, 1, 25)
        assert pd.isna(amend["restated_at"])

        # PIT as-of query: only the amendment survives after 2024-01-25.
        view = pit.query(
            Modality.INSIDER_TRADES,
            entity_ids=["us-cik-0000320193"],
            as_of=date(2024, 1, 25),
        )
        # availability_date <= 2024-01-25 AND not restated before that date.
        # Original is restated_at == 2024-01-25 (not strictly > 2024-01-25)
        # so it should be filtered out. Only amendment remains.
        assert len(view) == 1
        assert view.iloc[0]["shares"] == 600.0

    def test_no_form4_no_writes(self, lance_storage, monkeypatch, synthetic_feed_tarball):
        """A day with no parseable Form 4s should not create canonical/insider_trades."""
        from data.cli.ingest_feed import _ingest_day
        from data.sources.edgar.state import WatermarkStore

        self._patch_feed_download(monkeypatch, synthetic_feed_tarball)
        pit = PITEngine(lance_storage)
        wm = WatermarkStore(lance_storage)

        stats = _ingest_day(
            day=date(2024, 1, 15),
            keep_ciks={"0000320193", "0000789019"},
            storage=lance_storage,
            watermark=wm,
            pit=pit,
        )
        assert stats.form4_filings == 0
        assert stats.form4_trades == 0
        # No insider_trades shards written.
        assert not list(lance_storage.list("canonical/insider_trades"))
