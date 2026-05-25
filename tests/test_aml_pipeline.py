"""AML Phase A2 v2 pipeline tests.

Build-time only — we don't submit to AML. Verifies the pipeline DAG is
correctly constructed: all expected nodes registered, env vars set, etc.

Real submission is an operational decision tested manually against AML.
"""
from __future__ import annotations

import pytest


class TestPipelineConstruction:
    def test_all_nodes_registered(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        nodes = set(p.jobs.keys())
        expected = {
            "dera", "feed", "market", "macro", "patents", "news", "hiring",
            "finalize", "earnings_calls", "constraints", "coverage",
        }
        assert expected <= nodes, f"missing nodes: {expected - nodes}"

    def test_lance_backend_env_set_on_all_nodes(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        for name, node in p.jobs.items():
            env = node.environment_variables or {}
            assert env.get("BWM_STORAGE_BACKEND") == "lance", (
                f"node {name} missing BWM_STORAGE_BACKEND=lance: {env}"
            )

    def test_pipeline_parameters_default_correctly(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline(dera_start="2020Q1", dera_end="2020Q2")
        # The pipeline accepts the override; build-time success is the assertion.
        assert p is not None


class TestStepCommands:
    def test_dera_command_references_start_end_params(self):
        """AML resolves the literal values at submission time via
        ${{parent.inputs.X}} placeholders. We assert the placeholders exist
        rather than the literal values themselves.
        """
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline(dera_start="2023Q1", dera_end="2023Q4")
        dera_cmd = p.jobs["dera"].command
        assert "ingest_dera" in dera_cmd
        assert "--start" in dera_cmd and "--end" in dera_cmd

    def test_feed_command_references_start_end_params(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline(feed_start="2022-01-01", feed_end="2022-12-31")
        feed_cmd = p.jobs["feed"].command
        assert "ingest_feed" in feed_cmd
        assert "--start" in feed_cmd and "--end" in feed_cmd

    def test_validate_steps_no_longer_use_fallback_true(self):
        """A2.6-3: with wave_ready sentinel DAG edges, validators run only
        after Wave 1+2 complete, so `|| true` is no longer needed and would
        in fact mask real failures.
        """
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        for name in ("constraints", "coverage"):
            cmd = p.jobs[name].command
            assert "|| true" not in cmd, (
                f"{name} command still has `|| true`: {cmd}"
            )


class TestStepOutputs:
    def test_all_nodes_mount_data_root(self):
        """Every node must write to the shared canonical blob root."""
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        for name, node in p.jobs.items():
            outputs = node.outputs
            assert "data_root" in outputs, f"{name} missing data_root output"


class TestStepTimeouts:
    """A2.5-3: every pipeline node must have an explicit wall-clock timeout
    matching the calibrated _STEP_TIMEOUTS table. Prevents runaway jobs
    burning a week of cluster time on AML's 7-day default.
    """

    # Expected timeouts (must match _STEP_TIMEOUTS in phase_a2_v2.py)
    EXPECTED = {
        "dera": 7200, "feed": 86400, "market": 3600, "macro": 1800,
        "patents": 14400, "news": 86400, "hiring": 1800,
        "finalize": 300,
        "earnings_calls": 1800, "constraints": 600, "coverage": 600,
    }

    def test_every_node_has_timeout(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        for node_name, node in p.jobs.items():
            lim = getattr(node, "limits", None)
            assert lim is not None, f"{node_name} has no .limits"
            assert lim.timeout is not None, f"{node_name} has no .limits.timeout"

    def test_timeouts_match_expected_values(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        for node_name, expected in self.EXPECTED.items():
            node = p.jobs[node_name]
            actual = node.limits.timeout
            assert actual == expected, (
                f"{node_name}: expected timeout {expected}s, got {actual}s"
            )


class TestWaveReadyDag:
    """A2.6-3: wave_ready sentinel DAG edges replace the previous
    `|| true` ordering hack. finalize_wave1 gates all Wave 1 outputs,
    earnings_calls gates on finalize, validators gate on earnings_calls.
    """

    def test_finalize_node_emits_wave_ready(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        assert "finalize" in p.jobs
        assert "wave_ready" in p.jobs["finalize"].outputs

    def test_finalize_consumes_all_wave1_outputs(self):
        """finalize_wave1's inputs must include one per Wave 1 step so AML
        schedules it after all of them complete.
        """
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        finalize_inputs = set(p.jobs["finalize"].inputs.keys())
        expected = {"dera_root", "feed_root", "market_root", "macro_root",
                    "patents_root", "news_root", "hiring_root"}
        assert expected <= finalize_inputs, (
            f"finalize missing wave1 inputs: {expected - finalize_inputs}"
        )

    def test_earnings_calls_consumes_wave_ready(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        assert "wave_ready" in p.jobs["earnings_calls"].inputs

    def test_earnings_calls_emits_wave_ready(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        # earnings_calls also emits wave_ready so validators gate on its
        # completion (and thus the derived modality being present).
        assert "wave_ready" in p.jobs["earnings_calls"].outputs

    def test_validators_consume_wave_ready(self):
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        for name in ("constraints", "coverage"):
            assert "wave_ready" in p.jobs[name].inputs, (
                f"{name} missing wave_ready input"
            )

    def test_wave1_nodes_have_no_wave_ready_input(self):
        """Sanity: Wave 1 ingest nodes must NOT depend on wave_ready
        (would be a circular dependency).
        """
        from aml.pipelines.phase_a2_v2 import _build_pipeline
        p = _build_pipeline()
        for name in ("dera", "feed", "market", "macro", "patents", "news", "hiring"):
            assert "wave_ready" not in (p.jobs[name].inputs or {}), (
                f"{name} (Wave 1) must not depend on wave_ready"
            )
