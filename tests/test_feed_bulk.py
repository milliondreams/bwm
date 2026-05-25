"""SEC Feed bulk connector tests.

Uses a synthetic tarball (built in conftest) modeling the real SEC daily
Feed format with three filings:
  - AAPL 10-K with primary HTML + XBRL instance (XBRL skipped) + Exhibit 21
  - MSFT 8-K with single doc
  - IGNORED Form 4 (filtered when keep_ciks excludes 999)
"""
from __future__ import annotations

from datetime import date

import pytest


class TestParseNcFile:
    def test_parses_header_fields(self, synthetic_feed_tarball):
        """Extract and parse one .nc filing directly."""
        import io
        import tarfile
        from data.sources.edgar.feed_bulk import parse_nc_file

        with tarfile.open(fileobj=io.BytesIO(synthetic_feed_tarball), mode="r:gz") as tar:
            member = tar.getmember("0000320193-24-000001.nc")
            data = tar.extractfile(member).read()
        filing = parse_nc_file(data)
        assert filing is not None
        assert filing.accession == "0000320193-24-000001"
        assert filing.form == "10-K"
        assert filing.primary_cik == "0000320193"
        assert filing.filed_date == date(2024, 1, 15)
        assert filing.period == date(2023, 12, 31)

    def test_extracts_documents(self, synthetic_feed_tarball):
        import io, tarfile
        from data.sources.edgar.feed_bulk import parse_nc_file

        with tarfile.open(fileobj=io.BytesIO(synthetic_feed_tarball), mode="r:gz") as tar:
            data = tar.extractfile(tar.getmember("0000320193-24-000001.nc")).read()
        filing = parse_nc_file(data)
        # 3 docs in fixture, but XBRL instance (EX-101.INS) should be dropped
        doc_types = [d.doc_type for d in filing.documents]
        assert "EX-101.INS" not in doc_types
        assert "10-K" in doc_types
        assert "EX-21" in doc_types  # subsidiaries — keep, useful for entity resolution

    def test_unparseable_returns_none(self):
        from data.sources.edgar.feed_bulk import parse_nc_file
        assert parse_nc_file(b"not a valid submission") is None


class TestIterFilings:
    def test_count_matches_fixture(self, synthetic_feed_tarball):
        from data.sources.edgar.feed_bulk import iter_filings
        filings = list(iter_filings(synthetic_feed_tarball))
        assert len(filings) == 3  # AAPL, MSFT, IGNORED

    def test_cik_filter(self, synthetic_feed_tarball):
        """keep_ciks excluding 0000000999 should drop the IGNORED filing."""
        from data.sources.edgar.feed_bulk import iter_filings
        kept = list(iter_filings(
            synthetic_feed_tarball, keep_ciks={"0000320193", "0000789019"},
        ))
        assert len(kept) == 2
        ciks = {f.primary_cik for f in kept}
        assert ciks == {"0000320193", "0000789019"}

    def test_form_filter(self, synthetic_feed_tarball):
        from data.sources.edgar.feed_bulk import iter_filings
        kept = list(iter_filings(synthetic_feed_tarball, keep_forms={"10-K"}))
        assert len(kept) == 1
        assert kept[0].form == "10-K"


class TestFilingsToRows:
    def test_one_row_per_document(self, synthetic_feed_tarball):
        from data.sources.edgar.feed_bulk import iter_filings, filings_to_rows
        rows = list(filings_to_rows(iter_filings(
            synthetic_feed_tarball, keep_ciks={"0000320193"},
        )))
        # AAPL 10-K has 2 non-XBRL docs (primary + EX-21)
        assert len(rows) == 2

    def test_row_schema(self, synthetic_feed_tarball):
        from data.sources.edgar.feed_bulk import iter_filings, filings_to_rows
        rows = list(filings_to_rows(iter_filings(
            synthetic_feed_tarball, keep_ciks={"0000320193"},
        )))
        row = rows[0]
        required = {
            "entity_id", "cik", "year", "accession", "form", "filed_date",
            "period_of_report", "document_name", "document_type", "sequence",
            "mime_type", "content_bytes", "content_text", "n_chars", "sha256",
        }
        assert required <= set(row.keys())
        assert row["entity_id"] == "us-cik-0000320193"
        assert row["year"] == 2024
        assert isinstance(row["content_bytes"], bytes)
        # HTML docs get content_text decoded
        if row["mime_type"] == "text/html":
            assert row["content_text"] is not None

    def test_sha256_deterministic(self, synthetic_feed_tarball):
        from data.sources.edgar.feed_bulk import iter_filings, filings_to_rows
        rows1 = list(filings_to_rows(iter_filings(
            synthetic_feed_tarball, keep_ciks={"0000320193"},
        )))
        rows2 = list(filings_to_rows(iter_filings(
            synthetic_feed_tarball, keep_ciks={"0000320193"},
        )))
        assert [r["sha256"] for r in rows1] == [r["sha256"] for r in rows2]


class TestMimeDetection:
    def test_html_extensions(self):
        from data.sources.edgar.feed_bulk import _guess_mime
        assert _guess_mime("foo.htm", "10-K") == "text/html"
        assert _guess_mime("foo.html", "10-K") == "text/html"

    def test_xml_extensions(self):
        from data.sources.edgar.feed_bulk import _guess_mime
        assert _guess_mime("foo.xml", "EX-21") == "application/xml"
        assert _guess_mime("foo.xsd", "EX-21") == "application/xml"

    def test_binary_fallback(self):
        from data.sources.edgar.feed_bulk import _guess_mime
        assert _guess_mime("blob.bin", "EX-99") == "application/octet-stream"
