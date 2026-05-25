"""Storage abstraction over local filesystem and AzureML default datastore.

Gzipped writes go through `write_gz_bytes` / `read_gz_bytes` so callers do not
sprinkle gzip handling across modalities. The on-disk format is plain gzip
(magic 1f 8b), readable by any tool — no custom container.

The layout under the storage root is fixed (so the PIT engine and downstream
training code can find data without per-deployment config):

    raw/{source}/{ingest_date}/...                   immutable raw API pulls
    canonical/{modality}/{partition}                 normalized PIT records (no ext)
    entities/registry                                entity resolution
    graph/{quarter}/edges                            graph snapshots
    state/edgar/{cik10}                              per-CIK watermarks

The `path` strings handed to `write_table` / `read_table` are *logical* — they
do NOT include a format suffix. Each backend appends its own suffix
(`.parquet` for LocalStorage, `.lance` for LanceStorage). This lets the same
PIT engine code run against either physical format.

Legacy `write_parquet` / `read_parquet` methods are kept as deprecated
aliases for callers that haven't migrated yet; they strip a trailing
`.parquet` from the path before delegating to `write_table` / `read_table`.

`get_storage()` returns LocalStorage when BWM_DATA_ROOT is set (dev), else
AzureMLStorage targeting the workspace's default datastore (prod). Set
BWM_STORAGE_BACKEND=lance to swap in the Lance backend for new ingest runs.
"""
from __future__ import annotations

import gzip
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd


def _strip_parquet_suffix(path: str) -> str:
    """Legacy callers pass `foo.parquet`; new write_table API expects `foo`.

    Keeping the suffix-stripping behavior on the deprecated alias means we can
    run the existing PIT engine against either backend without a rename pass.
    """
    return path[:-8] if path.endswith(".parquet") else path


class Storage(ABC):
    """Read/write tabular and binary blobs under a fixed root.

    Subclasses pick the physical format for tabular data (parquet for
    LocalStorage, Lance for LanceStorage). Logical paths passed to
    `write_table`/`read_table` are format-suffix-free.
    """

    # ----- tabular (format-agnostic) -----

    @abstractmethod
    def write_table(self, path: str, df: pd.DataFrame) -> None:
        """Persist a DataFrame at the logical `path` (no format suffix)."""

    @abstractmethod
    def read_table(self, path: str) -> pd.DataFrame:
        """Read a DataFrame from the logical `path` (no format suffix)."""

    # ----- legacy aliases (deprecated) -----

    def write_parquet(self, path: str, df: pd.DataFrame) -> None:
        """Deprecated. Strips `.parquet` from path and delegates to `write_table`."""
        self.write_table(_strip_parquet_suffix(path), df)

    def read_parquet(self, path: str) -> pd.DataFrame:
        """Deprecated. Strips `.parquet` from path and delegates to `read_table`."""
        return self.read_table(_strip_parquet_suffix(path))

    # ----- binary -----

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
    """Filesystem backend. Tabular data lands as Parquet (`{path}.parquet`)."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _abs(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError(f"path escapes root: {path}")
        return p

    def write_table(self, path: str, df: pd.DataFrame) -> None:
        # Auto-add .parquet if the caller didn't (new API contract). Idempotent
        # for callers that already include the suffix via the legacy alias.
        physical = path if path.endswith(".parquet") else f"{path}.parquet"
        p = self._abs(physical)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)

    def read_table(self, path: str) -> pd.DataFrame:
        physical = path if path.endswith(".parquet") else f"{path}.parquet"
        return pd.read_parquet(self._abs(physical))

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
        # Existence checks happen against the logical path. We honor either
        # `foo` (new) or `foo.parquet` (legacy) so the check works regardless
        # of which API the caller used.
        if self._abs(path).exists():
            return True
        if not path.endswith(".parquet"):
            return self._abs(f"{path}.parquet").exists()
        return False

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

    def write_table(self, path: str, df: pd.DataFrame) -> None:
        physical = path if path.endswith(".parquet") else f"{path}.parquet"
        df.to_parquet(self._fs_path(physical), index=False, storage_options={})

    def read_table(self, path: str) -> pd.DataFrame:
        physical = path if path.endswith(".parquet") else f"{path}.parquet"
        return pd.read_parquet(self._fs_path(physical), storage_options={})

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
        if self._fs().exists(self._fs_path(path)):
            return True
        if not path.endswith(".parquet"):
            return self._fs().exists(self._fs_path(f"{path}.parquet"))
        return False

    def list(self, prefix: str) -> Iterable[str]:
        return self._fs().ls(self._fs_path(prefix), detail=False)


def get_storage() -> Storage:
    """Pick a backend based on env vars.

    BWM_STORAGE_BACKEND=lance selects LanceStorage for new ingest writes. The
    default ("parquet" or unset) preserves LocalStorage / AzureMLStorage
    behavior so existing canonical data keeps working without migration.

    BWM_DATA_ROOT (local dev) wins over BWM_DATASTORE_URI (azureml prod) when
    both are set.
    """
    backend = os.environ.get("BWM_STORAGE_BACKEND", "parquet").lower()
    local = os.environ.get("BWM_DATA_ROOT")

    if backend == "lance":
        # Lazy import — only required when actually using the Lance backend.
        from data.storage.lance_backend import LanceStorage

        if local:
            return LanceStorage(local)
        uri = os.environ.get("BWM_DATASTORE_URI")
        if not uri:
            raise RuntimeError(
                "BWM_STORAGE_BACKEND=lance requires BWM_DATA_ROOT (local) "
                "or BWM_DATASTORE_URI (azureml://...)"
            )
        return LanceStorage(uri)

    if local:
        return LocalStorage(local)
    uri = os.environ.get("BWM_DATASTORE_URI")
    if not uri:
        raise RuntimeError(
            "Set BWM_DATA_ROOT (local dev) or BWM_DATASTORE_URI "
            "(azureml://datastores/<name>/paths/<root>)"
        )
    return AzureMLStorage(uri)
