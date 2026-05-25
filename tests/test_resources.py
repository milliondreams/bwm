"""Pre-flight resource check tests."""
from __future__ import annotations

import pytest


class TestCheckDiskSpace:
    def test_passes_when_disk_available(self, tmp_path):
        from data.observability.resources import check_disk_space
        # tmp_path has plenty of space; 0.001 GiB threshold can't fail
        check_disk_space(tmp_path, min_gb=0.001)

    def test_raises_with_actionable_message(self, tmp_path, monkeypatch):
        """Mock shutil.disk_usage to report ~10 MiB free; expect raise."""
        from data.observability import resources
        FakeUsage = type("U", (), {"total": 10 ** 9, "used": 0, "free": 10 ** 7})()
        monkeypatch.setattr(resources.shutil, "disk_usage", lambda p: FakeUsage)
        with pytest.raises(RuntimeError, match="insufficient disk space"):
            resources.check_disk_space(tmp_path, min_gb=1.0, hint="DERA needs 5 GiB")

    def test_hint_included_in_error(self, tmp_path, monkeypatch):
        from data.observability import resources
        FakeUsage = type("U", (), {"total": 10 ** 9, "used": 0, "free": 10 ** 7})()
        monkeypatch.setattr(resources.shutil, "disk_usage", lambda p: FakeUsage)
        with pytest.raises(RuntimeError, match="DERA needs 5 GiB"):
            resources.check_disk_space(tmp_path, min_gb=1.0, hint="DERA needs 5 GiB")

    def test_resolves_nonexistent_path_to_parent(self, tmp_path):
        """Caller can check a not-yet-created subdir; we walk up to a real parent."""
        from data.observability.resources import check_disk_space
        future = tmp_path / "does" / "not" / "exist"
        # Should not raise — the actual tmp filesystem has plenty of space
        check_disk_space(future, min_gb=0.001)


class TestCheckMemory:
    def test_passes_with_available_memory(self):
        from data.observability.resources import check_memory
        # System has at least 100 MiB available almost always
        check_memory(min_gb=0.1)

    def test_raises_when_below_threshold(self, monkeypatch):
        """Mock psutil to report a tiny available memory."""
        from data.observability import resources
        fake = type("VM", (), {"available": 10 ** 7})  # 10 MiB
        monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: fake)
        with pytest.raises(RuntimeError, match="insufficient memory"):
            resources.check_memory(min_gb=1.0)

    def test_actionable_message(self, monkeypatch):
        from data.observability import resources
        fake = type("VM", (), {"available": 10 ** 7})
        monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: fake)
        with pytest.raises(RuntimeError, match="--skip-resource-check"):
            resources.check_memory(min_gb=1.0, hint="Feed peak days need 4 GiB")


class TestPreflight:
    def test_disk_failure_short_circuits(self, tmp_path, monkeypatch):
        """preflight() raises on the first failure (disk before memory)."""
        from data.observability import resources
        FakeUsage = type("U", (), {"total": 10 ** 9, "used": 0, "free": 10 ** 7})()
        monkeypatch.setattr(resources.shutil, "disk_usage", lambda p: FakeUsage)
        with pytest.raises(RuntimeError, match="insufficient disk space"):
            resources.preflight(tmp_path, disk_min_gb=1.0, mem_min_gb=100.0)

    def test_memory_failure_after_disk_passes(self, tmp_path, monkeypatch):
        from data.observability import resources
        fake = type("VM", (), {"available": 10 ** 7})
        monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: fake)
        with pytest.raises(RuntimeError, match="insufficient memory"):
            resources.preflight(tmp_path, disk_min_gb=0.001, mem_min_gb=100.0)

    def test_both_pass(self, tmp_path):
        from data.observability.resources import preflight
        # Both thresholds trivially satisfied
        preflight(tmp_path, disk_min_gb=0.001, mem_min_gb=0.1)
