"""Storage abstraction over local filesystem and AzureML default datastore.

Gzipped writes go through `write_gz_bytes` / `read_gz_bytes` so callers do not
sprinkle gzip handling across modalities. The on-disk format is plain gzip
(magic 1f 8b), readable by any tool — no custom container.

The layout under the storage root is fixed (so the PIT engine and downstream
training code can find data without per-deployment config):

    raw/{source}/{ingest_date}/...                   immutable raw API pulls
    canonical/{modality}/{partition}.parquet         normalized PIT records
    entities/registry.parquet                        entity resolution
    graph/{quarter}/edges.parquet                    graph snapshots

`get_storage()` returns LocalStorage when BWM_DATA_ROOT is set (dev), else
AzureMLStorage targeting the workspace's default datastore (prod).
"""
from __future__ import annotations

import gzip
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd


class Storage(ABC):
    """Read/write Parquet and JSON blobs under a fixed root."""

    @abstractmethod
    def write_parquet(self, path: str, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def read_parquet(self, path: str) -> pd.DataFrame: ...

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str) -> Iterable[str]: ...

    def write_gz_bytes(self, path: str, data: bytes) -> None:
        """Write `data` gzip-compressed. `path` should end in `.gz`."""
        self.write_bytes(path, gzip.compress(data, compresslevel=6))

    def read_gz_bytes(self, path: str) -> bytes:
        return gzip.decompress(self.read_bytes(path))

    def append_bytes(self, path: str, data: bytes) -> None:
        """Append `data` to the file at `path`.

        Local backend opens the file for append in 'ab' mode — O(1) regardless
        of existing file size. Cloud backends override this because blob
        storage does not support true append; they fall back to a
        read-modify-write OR a one-file-per-event scheme. The intended use is
        append-only JSONL logs (one JSON line per event), where the append is
        a hot path called once per ingest event.
        """
        raise NotImplementedError


class LocalStorage(Storage):
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _abs(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError(f"path escapes root: {path}")
        return p

    def write_parquet(self, path: str, df: pd.DataFrame) -> None:
        p = self._abs(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)

    def read_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(self._abs(path))

    def write_bytes(self, path: str, data: bytes) -> None:
        p = self._abs(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def append_bytes(self, path: str, data: bytes) -> None:
        p = self._abs(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "ab") as f:
            f.write(data)

    def read_bytes(self, path: str) -> bytes:
        return self._abs(path).read_bytes()

    def exists(self, path: str) -> bool:
        return self._abs(path).exists()

    def list(self, prefix: str) -> Iterable[str]:
        base = self._abs(prefix)
        if not base.exists():
            return []
        return sorted(
            str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()
        )


class AzureMLStorage(Storage):
    """Targets the workspace default datastore via fsspec/adlfs.

    Lazy import so dev environments without azureml-fsspec installed still work.
    """

    def __init__(self, datastore_uri: str):
        import fsspec  # type: ignore  # noqa: F401

        self.uri = datastore_uri.rstrip("/")

    def _fs_path(self, path: str) -> str:
        return f"{self.uri}/{path.lstrip('/')}"

    def _fs(self):
        import fsspec

        return fsspec.filesystem("azureml")

    def write_parquet(self, path: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._fs_path(path), index=False, storage_options={})

    def read_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(self._fs_path(path), storage_options={})

    def write_bytes(self, path: str, data: bytes) -> None:
        with self._fs().open(self._fs_path(path), "wb") as f:
            f.write(data)

    def append_bytes(self, path: str, data: bytes) -> None:
        """Azure Blob doesn't support true append. Read-modify-write fallback.

        For high-frequency event logging at scale, prefer the one-file-per-event
        pattern instead of calling this method. The fallback is provided so
        single-process dev/test code that uses .append_bytes() doesn't crash
        when run against the cloud backend, but it's O(file-size) per call.
        """
        existing = b""
        if self.exists(path):
            existing = self.read_bytes(path)
        self.write_bytes(path, existing + data)

    def read_bytes(self, path: str) -> bytes:
        with self._fs().open(self._fs_path(path), "rb") as f:
            return f.read()

    def exists(self, path: str) -> bool:
        return self._fs().exists(self._fs_path(path))

    def list(self, prefix: str) -> Iterable[str]:
        return self._fs().ls(self._fs_path(prefix), detail=False)


def get_storage() -> Storage:
    """Pick a backend based on env. BWM_DATA_ROOT wins (dev); else use AzureML."""
    local = os.environ.get("BWM_DATA_ROOT")
    if local:
        return LocalStorage(local)

    uri = os.environ.get("BWM_DATASTORE_URI")
    if not uri:
        raise RuntimeError(
            "Set BWM_DATA_ROOT (local dev) or BWM_DATASTORE_URI "
            "(azureml://datastores/<name>/paths/<root>)"
        )
    return AzureMLStorage(uri)
