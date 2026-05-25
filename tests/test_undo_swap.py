"""undo_swap CLI tests — reverses the most-recent swap_canonical operation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_swap(root: Path, modality: str, *extra) -> subprocess.CompletedProcess:
    env = {"BWM_DATA_ROOT": str(root), "PATH": "/usr/bin:/usr/local/bin",
           "HOME": str(root.parent)}
    return subprocess.run(
        [sys.executable, "-m", "data.cli.swap_canonical",
         "--modality", modality, "--skip-validation", *extra],
        env=env, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


def _run_undo(root: Path, modality: str, *extra) -> subprocess.CompletedProcess:
    env = {"BWM_DATA_ROOT": str(root), "PATH": "/usr/bin:/usr/local/bin",
           "HOME": str(root.parent)}
    return subprocess.run(
        [sys.executable, "-m", "data.cli.undo_swap",
         "--modality", modality, *extra],
        env=env, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


def _seed_swap(root: Path, modality: str) -> None:
    v2 = root / "canonical" / f"{modality}_v2"
    v2.mkdir(parents=True)
    (v2 / "marker.txt").write_text("v2 staging")
    canonical = root / "canonical" / modality
    canonical.mkdir(parents=True)
    (canonical / "marker.txt").write_text("v1 original")


class TestUndoSwap:
    def test_undo_restores_prior_canonical(self, tmp_path):
        _seed_swap(tmp_path, "financials")
        r = _run_swap(tmp_path, "financials")
        assert r.returncode == 0, r.stderr

        # Pre-undo: canonical has v2 contents
        canonical = tmp_path / "canonical" / "financials"
        assert canonical.joinpath("marker.txt").read_text() == "v2 staging"

        r = _run_undo(tmp_path, "financials")
        assert r.returncode == 0, r.stderr

        # Post-undo: canonical restored to v1 original
        assert canonical.joinpath("marker.txt").read_text() == "v1 original"
        # The v2 content is set aside under _undone_{ts}
        undone = list((tmp_path / "canonical").glob("financials_undone_*"))
        assert len(undone) == 1
        assert undone[0].joinpath("marker.txt").read_text() == "v2 staging"

    def test_undo_writes_undo_event(self, tmp_path):
        _seed_swap(tmp_path, "market")
        _run_swap(tmp_path, "market")
        _run_undo(tmp_path, "market")
        log = tmp_path / "state" / "swap_log.jsonl"
        events = [json.loads(line) for line in log.read_text().splitlines() if line]
        undos = [e for e in events if e.get("event") == "undo"]
        assert len(undos) == 1
        assert undos[0]["modality"] == "market"
        assert undos[0]["restored_from"].startswith("market_v1_deprecated_")

    def test_undo_with_no_swap_history_fails(self, tmp_path):
        r = _run_undo(tmp_path, "financials")
        assert r.returncode != 0
        assert "no pending swap" in r.stdout.lower()

    def test_undo_refuses_when_no_prior_canonical(self, tmp_path):
        """Swap with no prior canonical (no archive_to). Undo would leave
        canonical empty — must refuse.
        """
        v2 = tmp_path / "canonical" / "financials_v2"
        v2.mkdir(parents=True)
        (v2 / "marker.txt").write_text("v2 staging")
        r = _run_swap(tmp_path, "financials")
        assert r.returncode == 0

        r = _run_undo(tmp_path, "financials")
        assert r.returncode == 2
        assert "refusing" in r.stdout.lower()

    def test_dry_run_does_not_rename(self, tmp_path):
        _seed_swap(tmp_path, "financials")
        _run_swap(tmp_path, "financials")
        canonical_before = (tmp_path / "canonical" / "financials" / "marker.txt").read_text()
        r = _run_undo(tmp_path, "financials", "--dry-run")
        assert r.returncode == 0
        # Paths unchanged
        canonical_after = (tmp_path / "canonical" / "financials" / "marker.txt").read_text()
        assert canonical_before == canonical_after
        # No undo event recorded
        log = tmp_path / "state" / "swap_log.jsonl"
        events = [json.loads(line) for line in log.read_text().splitlines() if line]
        assert not any(e.get("event") == "undo" for e in events)

    def test_undo_then_redo_via_swap(self, tmp_path):
        """After undoing, the operator can re-promote the set-aside content
        by manually renaming it back to _v2 and re-running swap. This test
        just verifies the undo doesn't break the swap state.
        """
        _seed_swap(tmp_path, "financials")
        _run_swap(tmp_path, "financials")
        _run_undo(tmp_path, "financials")

        # A second undo should fail — nothing pending to undo.
        r = _run_undo(tmp_path, "financials")
        assert r.returncode != 0
