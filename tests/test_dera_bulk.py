"""DERA bulk connector tests.

Uses a synthetic in-memory DERA zip (built in conftest) so tests are fast
and don't hit network. Exercises the same parse paths as live data.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest


def _parse_segments(s):
    from data.sources.edgar.dera_bulk import _parse_segments
    return _parse_segments(s)


def _ddate_to_date(d):
    from data.sources.edgar.dera_bulk import _ddate_to_date
    return _ddate_to_date(d)


def _concept_from_tag(t, v):
    from data.sources.edgar.dera_bulk import _concept_from_tag
    return _concept_from_tag(t, v)


class TestSegmentParsing:
    def test_empty_segments_returns_empty(self):
        assert _parse_segments("") == ""

    def test_none_segments_returns_empty(self):
        assert _parse_segments(None) == ""

    def test_pandas_na_returns_empty(self):
        assert _parse_segments(pd.NA) == ""

    def test_single_axis(self):
        out = _parse_segments("EquityComponents=RetainedEarnings;")
        assert out == '{"EquityComponents":"RetainedEarnings"}'

    def test_multi_axis_sorted(self):
        """Multi-axis segments should produce sorted-key JSON (deterministic)."""
        out = _parse_segments("ProductOrService=iPhone;Geographical=US;")
        # Sorted keys → Geographical before ProductOrService
        assert out == '{"Geographical":"US","ProductOrService":"iPhone"}'

    def test_malformed_segment_dropped(self):
        """Segments without '=' are dropped, not raised."""
        out = _parse_segments("validkey=val;malformed;another=ok;")
        assert "validkey" in out and "another" in out and "malformed" not in out


class TestDdateConversion:
    def test_valid_yyyymmdd_int(self):
        assert _ddate_to_date(20240115) == date(2024, 1, 15)

    def test_valid_yyyymmdd_str(self):
        assert _ddate_to_date("20240115") == date(2024, 1, 15)

    def test_na_returns_none(self):
        assert _ddate_to_date(pd.NA) is None

    def test_none_returns_none(self):
        assert _ddate_to_date(None) is None

    def test_invalid_format_returns_none(self):
        assert _ddate_to_date("not-a-date") is None
        assert _ddate_to_date(123) is None  # too short

    def test_invalid_date_returns_none(self):
        # February 30th — valid format, invalid date
        assert _ddate_to_date(20240230) is None


class TestConceptFromTag:
    def test_us_gaap_namespace(self):
        assert _concept_from_tag("Revenues", "us-gaap/2024") == "us-gaap:Revenues"

    def test_custom_when_no_slash(self):
        # Company extension tag (version is the CIK or random)
        assert _concept_from_tag("Foo", "320193") == "custom:Foo"

    def test_na_inputs_return_unknown(self):
        assert _concept_from_tag(pd.NA, "us-gaap/2024") == "custom:unknown"
        assert _concept_from_tag("Foo", pd.NA) == "custom:unknown"


class TestParseQuarter:
    def test_full_quarter_parse(self, synthetic_dera_zip):
        from data.sources.edgar.dera_bulk import parse_quarter
        # Keep CIK = 320193 only (the IGNORED CIK 999 must be filtered)
        df = parse_quarter(synthetic_dera_zip, keep_ciks={"0000320193"})
        assert len(df) > 0
        # All rows are for AAPL
        assert set(df["entity_id"].unique()) == {"us-cik-0000320193"}
        # IGNORED CIK 999 is absent
        assert (df["entity_id"] == "us-cik-0000000999").sum() == 0

    def test_filter_keeps_all_when_none(self, synthetic_dera_zip):
        from data.sources.edgar.dera_bulk import parse_quarter
        df = parse_quarter(synthetic_dera_zip, keep_ciks=None)
        # No filter → all CIKs present
        assert set(df["entity_id"].unique()) >= {
            "us-cik-0000320193", "us-cik-0000000999",
        }

    def test_dimensions_preserved(self, synthetic_dera_zip):
        """The fixture has one row with `ProductOrService=iPhone;` segments.
        Verify it lands as parsed JSON in dimensions_json, not dropped.
        """
        from data.sources.edgar.dera_bulk import parse_quarter
        df = parse_quarter(synthetic_dera_zip, keep_ciks={"0000320193"})
        dim_rows = df[df["dimensions_json"] != ""]
        assert len(dim_rows) >= 1
        assert any("iPhone" in d for d in dim_rows["dimensions_json"])

    def test_na_segments_become_empty_string(self, synthetic_dera_zip):
        """NA-handling path: rows with NA segments → '' (consolidated)."""
        from data.sources.edgar.dera_bulk import parse_quarter
        df = parse_quarter(synthetic_dera_zip, keep_ciks={"0000320193"})
        assert (df["dimensions_json"] == "").sum() > 0

    def test_required_pit_columns_present(self, synthetic_dera_zip):
        from data.sources.edgar.dera_bulk import parse_quarter
        df = parse_quarter(synthetic_dera_zip, keep_ciks={"0000320193"})
        required = {
            "entity_id", "modality", "effective_date", "availability_date",
            "restated_at", "source", "source_ref",
        }
        assert required <= set(df.columns)

    def test_perspective_is_dera(self, synthetic_dera_zip):
        from data.sources.edgar.dera_bulk import parse_quarter
        df = parse_quarter(synthetic_dera_zip, keep_ciks={"0000320193"})
        assert (df["perspective"] == "sec_edgar:dera").all()


class TestEnumerateQuarters:
    def test_enumerate_range(self):
        from data.sources.edgar.dera_bulk import enumerate_quarters
        qs = enumerate_quarters(2020, 2021)
        assert len(qs) == 8
        assert str(qs[0]) == "2020Q1"
        assert str(qs[-1]) == "2021Q4"


class TestDeraQuarterURL:
    def test_url_pattern(self):
        from data.sources.edgar.dera_bulk import DeraQuarter
        q = DeraQuarter(2024, 4)
        assert "2024q4.zip" in q.url
        assert q.url.startswith("https://www.sec.gov/")
