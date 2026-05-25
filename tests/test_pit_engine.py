"""PIT engine tests against BOTH storage backends.

The engine itself is format-agnostic; these tests verify that's actually
true by parameterizing on backend. Restatement, tiebreak, and partition-key
semantics must be byte-identical regardless of physical storage format.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest


def _engine(storage):
    from data.pit.engine import PITEngine
    return PITEngine(storage)


def _modality():
    from data.schemas.pit import Modality
    return Modality.FINANCIALS


def _partition_keys():
    from data.schemas.financial import FinancialFact
    return FinancialFact.PARTITION_KEYS


class TestPITRestatement:
    def test_restatement_as_of_pre(self, any_storage, aapl_restatement_rows):
        engine = _engine(any_storage)
        engine.write(_modality(), aapl_restatement_rows, partition_keys=_partition_keys())
        out = engine.query(
            _modality(), entity_ids=["us-cik-0000320193"],
            as_of=date(2024, 1, 20), partition_keys=_partition_keys(),
        )
        assert len(out) == 1
        assert out.iloc[0]["value"] == 100.0  # original

    def test_restatement_as_of_post(self, any_storage, aapl_restatement_rows):
        engine = _engine(any_storage)
        engine.write(_modality(), aapl_restatement_rows, partition_keys=_partition_keys())
        out = engine.query(
            _modality(), entity_ids=["us-cik-0000320193"],
            as_of=date(2024, 3, 1), partition_keys=_partition_keys(),
        )
        assert len(out) == 1
        assert out.iloc[0]["value"] == 110.0  # restated

    def test_restatement_as_of_before_first_filing(self, any_storage, aapl_restatement_rows):
        engine = _engine(any_storage)
        engine.write(_modality(), aapl_restatement_rows, partition_keys=_partition_keys())
        out = engine.query(
            _modality(), entity_ids=["us-cik-0000320193"],
            as_of=date(2024, 1, 1), partition_keys=_partition_keys(),
        )
        # Neither filing was knowable yet
        assert len(out) == 0


class TestPartitionKeys:
    def test_dimensional_rows_not_collapsed(self, any_storage, aapl_restatement_rows):
        """Same (entity, concept, period) with different dimensions_json
        should be treated as distinct rows, NOT collapsed as a restatement.
        This is the bug that broke the legacy companyfacts ingest.
        """
        rows = aapl_restatement_rows.copy()
        # Add a segment-level Revenues row for the same period
        seg = rows.iloc[0].copy()
        seg["source_ref"] = "0000320193-24-000003"
        seg["dimensions_json"] = '{"ProductOrService":"iPhone"}'
        seg["value"] = 70.0
        rows = pd.concat([rows, pd.DataFrame([seg])], ignore_index=True)

        engine = _engine(any_storage)
        engine.write(_modality(), rows, partition_keys=_partition_keys())
        out = engine.query(
            _modality(), entity_ids=["us-cik-0000320193"],
            as_of=date(2024, 3, 1), partition_keys=_partition_keys(),
        )
        # Consolidated (restated to 110) AND segment-level (70) both present
        values = sorted(out["value"].tolist())
        assert values == [70.0, 110.0]


class TestCrossEntityScan:
    def test_scan_without_entity_filter(self, any_storage, aapl_restatement_rows):
        """query() with entity_ids=None should glob the modality directory
        and aggregate across all entities — must work for both backends.
        """
        # Write rows for two different entities
        rows1 = aapl_restatement_rows
        rows2 = aapl_restatement_rows.copy()
        rows2["entity_id"] = "us-cik-0000789019"
        rows2["source_ref"] = rows2["source_ref"].str.replace("320193", "789019")

        engine = _engine(any_storage)
        engine.write(_modality(), rows1, partition_keys=_partition_keys())
        engine.write(_modality(), rows2, partition_keys=_partition_keys())

        out = engine.query(
            _modality(), as_of=date(2024, 3, 1),
            partition_keys=_partition_keys(),
        )
        assert set(out["entity_id"].unique()) == {
            "us-cik-0000320193", "us-cik-0000789019",
        }


class TestTiebreaker:
    def test_source_ref_breaks_same_day_tie(self, any_storage):
        """Two filings on the same day for the same fact → source_ref breaks
        the tie deterministically (lex-max wins). Same answer across runs.
        """
        rows = pd.DataFrame([
            {
                "entity_id": "us-cik-0000320193", "modality": "financials",
                "effective_date": date(2023, 12, 31),
                "availability_date": date(2024, 1, 15),
                "restated_at": None, "source": "x",
                "source_ref": "ACC-A",
                "concept": "us-gaap:Revenues", "taxonomy": "us-gaap",
                "fiscal_year": 2023, "fiscal_period": "Q4", "unit": "USD",
                "context_id": "ctx1", "dimensions_json": "", "value": 100.0,
            },
            {
                "entity_id": "us-cik-0000320193", "modality": "financials",
                "effective_date": date(2023, 12, 31),
                "availability_date": date(2024, 1, 15),  # SAME DAY
                "restated_at": None, "source": "x",
                "source_ref": "ACC-B",  # lex-greater
                "concept": "us-gaap:Revenues", "taxonomy": "us-gaap",
                "fiscal_year": 2023, "fiscal_period": "Q4", "unit": "USD",
                "context_id": "ctx1", "dimensions_json": "", "value": 200.0,
            },
        ])
        engine = _engine(any_storage)
        engine.write(_modality(), rows, partition_keys=_partition_keys())
        out = engine.query(
            _modality(), entity_ids=["us-cik-0000320193"],
            as_of=date(2024, 2, 1), partition_keys=_partition_keys(),
        )
        assert len(out) == 1
        # ACC-B wins by source_ref lex-max tiebreak
        assert out.iloc[0]["value"] == 200.0
        assert out.iloc[0]["source_ref"] == "ACC-B"


class TestIdempotency:
    def test_double_write_is_noop(self, any_storage, aapl_restatement_rows):
        """Writing the same batch twice should produce identical canonical
        contents — dedup catches the re-ingest case.
        """
        engine = _engine(any_storage)
        engine.write(_modality(), aapl_restatement_rows, partition_keys=_partition_keys())
        df1 = any_storage.read_table(
            "canonical/financials/entity=us-cik-0000320193"
        )
        engine.write(_modality(), aapl_restatement_rows, partition_keys=_partition_keys())
        df2 = any_storage.read_table(
            "canonical/financials/entity=us-cik-0000320193"
        )
        assert len(df1) == len(df2)
