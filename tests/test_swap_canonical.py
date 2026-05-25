"""Atomic canonical swap tests.

Exercises the swap CLI's promotion + archival flow against a real tmp
filesystem. Doesn't need network or AML.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _make_v2_dir(root: Path, modality: str) -> Path:
    """Create canonical/{modality}_v2 with a token file inside."""
    v2 = root / "canonical" / f"{modality}_v2"
    v2.mkdir(parents=True)
    (v2 / "marker.txt").write_text("v2 staging contents")
    return v2


def _make_existing_canonical(root: Path, modality: str) -> Path:
    canonical = root / "canonical" / modality
    canonical.mkdir(parents=True)
    (canonical / "marker.txt").write_text("v1 contents")
    return canonical


def _run_swap_cli(root: Path, modality: str, *extra) -> subprocess.CompletedProcess:
    """Invoke the swap CLI as a subprocess with isolated env."""
    env = {
        "BWM_DATA_ROOT": str(root),
        "PATH": "/usr/bin:/usr/local/bin",
        "HOME": str(root.parent),
    }
    cmd = [
        sys.executable, "-m", "data.cli.swap_canonical",
        "--modality", modality, "--skip-validation", *extra,
    ]
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


class TestSwap:
    def test_swap_promotes_v2_to_canonical(self, tmp_path):
        _make_v2_dir(tmp_path, "financials")
        r = _run_swap_cli(tmp_path, "financials")
        assert r.returncode == 0, f"swap failed: {r.stderr}"
        # _v2 is gone; canonical now has v2 contents
        assert not (tmp_path / "canonical/financials_v2").exists()
        assert (tmp_path / "canonical/financials/marker.txt").read_text() == "v2 staging contents"

    def test_swap_archives_existing_canonical(self, tmp_path):
        _make_v2_dir(tmp_path, "financials")
        _make_existing_canonical(tmp_path, "financials")
        r = _run_swap_cli(tmp_path, "financials")
        assert r.returncode == 0, f"swap failed: {r.stderr}"
        # Old canonical archived as _v1_deprecated_{date}
        deprecated = list((tmp_path / "canonical").glob("financials_v1_deprecated_*"))
        assert len(deprecated) == 1
        assert (deprecated[0] / "marker.txt").read_text() == "v1 contents"

    def test_dry_run_does_not_rename(self, tmp_path):
        _make_v2_dir(tmp_path, "financials")
        _make_existing_canonical(tmp_path, "financials")
        r = _run_swap_cli(tmp_path, "financials", "--dry-run")
        assert r.returncode == 0
        # Both paths still exist
        assert (tmp_path / "canonical/financials_v2").exists()
        assert (tmp_path / "canonical/financials").exists()

    def test_missing_v2_aborts(self, tmp_path):
        # No _v2 dir created
        r = _run_swap_cli(tmp_path, "financials")
        assert r.returncode != 0
        assert "missing" in (r.stderr + r.stdout).lower()

    def test_swap_log_written(self, tmp_path):
        _make_v2_dir(tmp_path, "market")
        r = _run_swap_cli(tmp_path, "market")
        assert r.returncode == 0
        log = tmp_path / "state" / "swap_log.jsonl"
        assert log.exists()
        events = [json.loads(line) for line in log.read_text().splitlines() if line]
        assert any(e["modality"] == "market" for e in events)
