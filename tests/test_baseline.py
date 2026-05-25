"""Baseline + regression-diff infrastructure tests.

Auto-promote semantics: only passing runs write a baseline; failing runs
preserve the prior good baseline so a regression can't silently become
the new normal.
"""
from __future__ import annotations

import json
import time

import pytest

from data.observability.baseline import (
    BASELINE_DIR,
    diff,
    has_material_drift,
    latest_baseline,
    write_baseline,
)


class TestRoundTrip:
    def test_write_then_latest_returns_same_payload(self, local_storage):
        payload = {"cvr": 0.01, "hard_checks": 100, "hard_violations": 1,
                   "per_modality": {"financials": {"checks": 100, "violations": 1, "cvr": 0.01}}}
        path = write_baseline(local_storage, payload)
        assert path.startswith(BASELINE_DIR + "/")
        got = latest_baseline(local_storage)
        assert got is not None
        latest_path, latest_payload = got
        assert latest_path == path
        assert latest_payload["cvr"] == 0.01

    def test_no_baselines_returns_none(self, local_storage):
        assert latest_baseline(local_storage) is None

    def test_latest_picks_max_filename(self, local_storage):
        # Two writes; second wins because filenames are chronological.
        write_baseline(local_storage, {"cvr": 0.1})
        time.sleep(1.05)  # ensure compact-timestamp tick
        write_baseline(local_storage, {"cvr": 0.2})
        _, payload = latest_baseline(local_storage)
        assert payload["cvr"] == 0.2


class TestDiff:
    def test_scalar_deltas(self):
        prior = {"cvr": 0.01, "hard_checks": 1000, "hard_violations": 10}
        current = {"cvr": 0.02, "hard_checks": 1100, "hard_violations": 22}
        d = diff(prior, current)
        assert d["scalars"]["cvr"]["delta"] == pytest.approx(0.01)
        assert d["scalars"]["hard_checks"]["delta"] == 100
        # rel_delta for cvr: (0.02 - 0.01) / max(|0.01|, 1) = 0.01 (denom clamped to 1)
        assert d["scalars"]["cvr"]["rel_delta"] == pytest.approx(0.01)
        # rel_delta for hard_checks: (100) / 1000 = 0.1
        assert d["scalars"]["hard_checks"]["rel_delta"] == pytest.approx(0.1)

    def test_per_modality_deltas(self):
        prior = {"per_modality": {"financials": {"checks": 100, "violations": 1, "cvr": 0.01}}}
        current = {"per_modality": {
            "financials": {"checks": 80, "violations": 2, "cvr": 0.025},
            "filings": {"checks": 50, "violations": 0, "cvr": 0.0},
        }}
        d = diff(prior, current)
        assert d["per_modality"]["financials"]["checks"]["delta"] == -20
        assert d["per_modality"]["filings"]["checks"]["prior"] == 0
        assert d["per_modality"]["filings"]["checks"]["current"] == 50

    def test_coverage_changed_flag(self):
        prior = {"coverage": {"row_count_financials": {"observed": 1000}}}
        current = {"coverage": {"row_count_financials": {"observed": 1200}}}
        d = diff(prior, current)
        assert d["coverage"]["row_count_financials"]["changed"] is True

    def test_coverage_unchanged_observed(self):
        prior = {"coverage": {"row_count_financials": {"observed": 1000}}}
        current = {"coverage": {"row_count_financials": {"observed": 1000}}}
        d = diff(prior, current)
        assert d["coverage"]["row_count_financials"]["changed"] is False

    def test_empty_payloads(self):
        d = diff({}, {})
        assert d["scalars"]["cvr"]["delta"] == 0.0
        assert d["per_modality"] == {}
        assert d["coverage"] == {}


class TestMaterialDrift:
    def test_below_threshold_not_material(self):
        prior = {"cvr": 0.01, "hard_checks": 1000, "hard_violations": 100}
        current = {"cvr": 0.011, "hard_checks": 1010, "hard_violations": 101}
        d = diff(prior, current)
        # All rel_deltas < 5%
        assert has_material_drift(d, threshold=0.05) is False

    def test_above_threshold_is_material(self):
        prior = {"hard_checks": 1000, "hard_violations": 10, "cvr": 0.01}
        current = {"hard_checks": 1200, "hard_violations": 24, "cvr": 0.02}
        d = diff(prior, current)
        assert has_material_drift(d, threshold=0.05) is True

    def test_per_modality_triggers_material(self):
        prior = {"per_modality": {"financials": {"checks": 100, "violations": 1, "cvr": 0.01}}}
        current = {"per_modality": {"financials": {"checks": 90, "violations": 1, "cvr": 0.011}}}
        d = diff(prior, current)
        # checks delta is 10% — above 5% threshold
        assert has_material_drift(d, threshold=0.05) is True
