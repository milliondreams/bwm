"""prune_deprecated CLI tests."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


def _make_deprecated(root: Path, modality: str, days_ago: int) -> Path:
    """Create canonical/{modality}_v1_deprecated_{YYYYMMDD} dated days_ago days ago."""
    d = date.today() - timedelta(days=days_ago)
    p = root / "canonical" / f"{modality}_v1_deprecated_{d:%Y%m%d}"
    p.mkdir(parents=True)
    (p / "marker.txt").write_text(f"deprecated {days_ago} days ago")
    return p


def _run_prune(root: Path, *extra) -> subprocess.CompletedProcess:
    env = {"BWM_DATA_ROOT": str(root), "PATH": "/usr/bin:/usr/local/bin",
           "HOME": str(root.parent)}
    return subprocess.run(
        [sys.executable, "-m", "data.cli.prune_deprecated", *extra],
        env=env, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


class TestPrune:
    def test_keeps_newest_count_regardless_of_age(self, tmp_path):
        # 5 archives, all very old. --keep-count 3 should keep the 3 newest.
        for days in (100, 90, 80, 70, 60):
            _make_deprecated(tmp_path, "financials", days)
        r = _run_prune(tmp_path, "--age-days", "30", "--keep-count", "3")
        assert r.returncode == 0, r.stderr
        remaining = sorted(p.name for p in (tmp_path / "canonical").iterdir())
        # Keep 3 newest (60, 70, 80)
        assert len(remaining) == 3
        d = date.today()
        keep_days = sorted([60, 70, 80])
        for days in keep_days:
            expected = f"financials_v1_deprecated_{(d - timedelta(days=days)):%Y%m%d}"
            assert expected in remaining

    def test_keeps_within_age_threshold(self, tmp_path):
        # All very new; --keep-count 0, --age-days 30 keeps all under 30 days
        for days in (5, 10, 20, 25):
            _make_deprecated(tmp_path, "financials", days)
        r = _run_prune(tmp_path, "--age-days", "30", "--keep-count", "0")
        assert r.returncode == 0, r.stderr
        remaining = list((tmp_path / "canonical").iterdir())
        assert len(remaining) == 4

    def test_deletes_old_beyond_keep_count(self, tmp_path):
        _make_deprecated(tmp_path, "financials", 5)    # keep — within age
        _make_deprecated(tmp_path, "financials", 40)   # candidate
        _make_deprecated(tmp_path, "financials", 50)   # candidate
        _make_deprecated(tmp_path, "financials", 100)  # candidate
        # keep_count=1 keeps only the newest (age=5). The rest are >30 days → deleted.
        r = _run_prune(tmp_path, "--age-days", "30", "--keep-count", "1")
        assert r.returncode == 0, r.stderr
        remaining = list((tmp_path / "canonical").iterdir())
        assert len(remaining) == 1
        assert "5" in str(date.today() - timedelta(days=5))

    def test_dry_run_does_not_delete(self, tmp_path):
        _make_deprecated(tmp_path, "financials", 100)
        _make_deprecated(tmp_path, "financials", 200)
        r = _run_prune(tmp_path, "--age-days", "30", "--keep-count", "0",
                       "--dry-run")
        assert r.returncode == 0, r.stderr
        remaining = list((tmp_path / "canonical").iterdir())
        assert len(remaining) == 2
        assert "DRY" in r.stdout

    def test_modality_filter(self, tmp_path):
        _make_deprecated(tmp_path, "financials", 100)
        _make_deprecated(tmp_path, "market", 100)
        r = _run_prune(tmp_path, "--age-days", "30", "--keep-count", "0",
                       "--modality", "financials")
        assert r.returncode == 0, r.stderr
        remaining = sorted(p.name for p in (tmp_path / "canonical").iterdir())
        # market_* survives; financials_* deleted
        assert any(name.startswith("market_") for name in remaining)
        assert not any(name.startswith("financials_") for name in remaining)

    def test_prune_log_written(self, tmp_path):
        _make_deprecated(tmp_path, "financials", 100)
        _run_prune(tmp_path, "--age-days", "30", "--keep-count", "0")
        log = tmp_path / "state" / "prune_log.jsonl"
        assert log.exists()
        events = [json.loads(line) for line in log.read_text().splitlines() if line]
        assert len(events) == 1
        assert events[0]["modality"] == "financials"
        assert events[0]["age_days"] == 100

    def test_no_archives_is_no_op(self, tmp_path):
        # No canonical dir at all
        r = _run_prune(tmp_path, "--age-days", "30")
        assert r.returncode == 0
        assert "nothing to prune" in r.stdout.lower()

    def test_undone_archives_pruned_too(self, tmp_path):
        # Build an _undone_ directory by hand
        d = date.today() - timedelta(days=100)
        ts = d.strftime("%Y%m%dT000000Z")
        p = tmp_path / "canonical" / f"financials_undone_{ts}"
        p.mkdir(parents=True)
        (p / "marker.txt").write_text("undone")
        r = _run_prune(tmp_path, "--age-days", "30", "--keep-count", "0")
        assert r.returncode == 0, r.stderr
        assert not p.exists()
