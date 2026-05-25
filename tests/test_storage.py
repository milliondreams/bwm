"""Storage backend contract tests.

Runs against both LocalStorage (parquet) and LanceStorage so we know the
abstraction stays in sync. Lance-specific behavior (versioning, parquet
fallback, list semantics) is covered in dedicated tests below.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


# ---------- contract tests (both backends) ----------

class TestStorageContract:
    """Every backend MUST satisfy these invariants."""

    def test_write_then_read_round_trip(self, any_storage):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        any_storage.write_table("foo/bar", df)
        out = any_storage.read_table("foo/bar")
        assert len(out) == 3
        assert set(out.columns) == {"a", "b"}
        assert out["a"].tolist() == [1, 2, 3]

    def test_exists_returns_true_after_write(self, any_storage):
        df = pd.DataFrame({"x": [1]})
        any_storage.write_table("a/b", df)
        assert any_storage.exists("a/b")

    def test_exists_returns_false_for_missing(self, any_storage):
        assert not any_storage.exists("never/written")

    def test_legacy_parquet_alias_works(self, any_storage):
        """write_parquet / read_parquet should accept legacy `.parquet` paths
        and round-trip regardless of which physical backend is in play.
        """
        df = pd.DataFrame({"v": [42]})
        any_storage.write_parquet("legacy/data.parquet", df)
        out = any_storage.read_parquet("legacy/data.parquet")
        assert out["v"].tolist() == [42]

    def test_bytes_round_trip(self, any_storage):
        data = b"some raw bytes \x00\x01\xff"
        any_storage.write_bytes("blob.bin", data)
        assert any_storage.read_bytes("blob.bin") == data
        assert any_storage.exists("blob.bin")

    def test_gz_round_trip(self, any_storage):
        data = b"this is the content " * 100
        any_storage.write_gz_bytes("blob.gz", data)
        assert any_storage.read_gz_bytes("blob.gz") == data

    def test_append_bytes(self, any_storage):
        any_storage.append_bytes("log.jsonl", b'{"a":1}\n')
        any_storage.append_bytes("log.jsonl", b'{"b":2}\n')
        assert any_storage.read_bytes("log.jsonl") == b'{"a":1}\n{"b":2}\n'

    def test_path_escape_raises(self, any_storage):
        with pytest.raises(ValueError, match="escapes root"):
            any_storage.write_table("../outside", pd.DataFrame({"x": [1]}))


# ---------- Lance-specific behavior ----------

class TestLanceBackend:
    def test_writes_lance_directory_not_parquet_file(self, lance_storage, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        lance_storage.write_table("data/sample", df)
        # Lance dataset is a directory ending in .lance
        assert (tmp_path / "data/sample.lance").is_dir()
        # No parquet file created
        assert not (tmp_path / "data/sample.parquet").exists()

    def test_parquet_fallback_for_read(self, lance_storage, tmp_path):
        """Pre-existing parquet should be readable through LanceStorage
        during migration (covers the 'first write upgrades to Lance' path).
        """
        # Manually write a parquet file (simulating pre-migration data)
        parquet_dir = tmp_path / "legacy"
        parquet_dir.mkdir()
        df = pd.DataFrame({"old": [1, 2, 3]})
        df.to_parquet(parquet_dir / "data.parquet", index=False)
        # Lance backend should fall back to reading the parquet
        out = lance_storage.read_table("legacy/data")
        assert out["old"].tolist() == [1, 2, 3]

    def test_versioning_on_overwrite(self, lance_storage):
        """Each write creates a new Lance version (built-in time-travel)."""
        import lance
        df1 = pd.DataFrame({"v": [1]})
        df2 = pd.DataFrame({"v": [2]})
        lance_storage.write_table("vers/data", df1)
        lance_storage.write_table("vers/data", df2)
        ds = lance.dataset(f"{lance_storage.root}/vers/data.lance")
        versions = [v["version"] for v in ds.versions()]
        assert len(versions) >= 2

    def test_list_strips_lance_suffix(self, lance_storage):
        """list() should surface dataset paths without the `.lance` suffix
        so the PIT engine's regex matches both backends.
        """
        for cik in ["320193", "789019"]:
            lance_storage.write_table(
                f"canonical/financials/entity=us-cik-{cik}",
                pd.DataFrame({"x": [1]}),
            )
        paths = list(lance_storage.list("canonical/financials"))
        # Should find both, with no .lance suffix
        assert any("entity=us-cik-320193" in p and not p.endswith(".lance") for p in paths)
        assert any("entity=us-cik-789019" in p and not p.endswith(".lance") for p in paths)


# ---------- get_storage() factory ----------

class TestGetStorageFactory:
    def test_lance_backend_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BWM_DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("BWM_STORAGE_BACKEND", "lance")
        from data.storage import get_storage
        from data.storage.lance_backend import LanceStorage
        s = get_storage()
        assert isinstance(s, LanceStorage)

    def test_default_is_parquet(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BWM_DATA_ROOT", str(tmp_path))
        monkeypatch.delenv("BWM_STORAGE_BACKEND", raising=False)
        from data.storage import get_storage
        from data.storage.backend import LocalStorage
        s = get_storage()
        assert isinstance(s, LocalStorage)

    def test_missing_config_raises(self, monkeypatch):
        monkeypatch.delenv("BWM_DATA_ROOT", raising=False)
        monkeypatch.delenv("BWM_DATASTORE_URI", raising=False)
        monkeypatch.delenv("BWM_STORAGE_BACKEND", raising=False)
        from data.storage import get_storage
        with pytest.raises(RuntimeError, match="BWM_DATA_ROOT"):
            get_storage()
