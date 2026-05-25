"""Lance-backed Storage implementation.

Implements the same `Storage` interface as `LocalStorage`/`AzureMLStorage`
but persists tabular data as Lance datasets instead of Parquet files. The
PIT engine and all modality CLIs interact with this through the abstract
`write_table` / `read_table` methods — no caller knows about Lance directly.

### Physical layout

A logical path like `canonical/financials/entity=us-cik-0000320193` is stored
on disk as the Lance dataset directory
`canonical/financials/entity=us-cik-0000320193.lance/` containing fragments,
manifest, and version log. Lance datasets are directories, not files — this
is the one operational difference from Parquet that callers see when they
list the filesystem directly.

### Versioning

Lance datasets carry built-in versioning: every `write_dataset(..., mode=...)`
commit creates a new version. We do NOT yet expose this through the Storage
API (existing PITEngine uses read-merge-write which collapses history into a
single canonical view). The versioning is preserved on disk for any later
inspection or rollback via `lance.dataset(path).checkout_version(N)`.

### Append semantics

`write_table` uses `mode="overwrite"` to match the existing PITEngine
contract (which already performs the read-merge-dedupe before calling write).
A future optimization could switch to `mode="append"` + post-write dedup,
which would be faster on large tables and preserve per-batch version history.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Iterable

import lance
import pandas as pd
import pyarrow as pa

from data.storage.backend import Storage, _strip_parquet_suffix


def _dataset_suffix(path: str) -> str:
    """Append `.lance` to a logical path, idempotently."""
    return path if path.endswith(".lance") else f"{path}.lance"


class LanceStorage(Storage):
    """Filesystem-backed Lance storage.

    Tabular data lands as `{path}.lance/` directories. Binary data lands as
    plain files (same as LocalStorage). The root is either a local filesystem
    path (dev) or an azureml://... URI (prod).
    """

    def __init__(self, root: str | Path):
        # Lance supports both local paths and cloud URIs (s3://, az://, gs://).
        # Local roots are normalized; URI roots are passed through.
        s = str(root)
        if "://" in s:
            self.root = s.rstrip("/")
            self._is_local = False
        else:
            self.root = str(Path(s).resolve())
            Path(self.root).mkdir(parents=True, exist_ok=True)
            self._is_local = True

    def _abs(self, path: str) -> str:
        """Resolve a logical path against the root.

        For local roots, returns an absolute filesystem path string. For URI
        roots, returns a URI string. Path-escape protection is applied to
        local roots only (URI semantics don't have the same risk).
        """
        if self._is_local:
            p = (Path(self.root) / path).resolve()
            if not str(p).startswith(self.root):
                raise ValueError(f"path escapes root: {path}")
            return str(p)
        return f"{self.root}/{path.lstrip('/')}"

    # ----- tabular -----

    def write_table(self, path: str, df: pd.DataFrame) -> None:
        physical = self._abs(_dataset_suffix(path))
        if self._is_local:
            Path(physical).parent.mkdir(parents=True, exist_ok=True)
        # Convert via PyArrow so dtype handling matches Lance's expectations
        # (Lance is Arrow-native; pandas DataFrame → arrow Table is the
        # single best-supported path).
        table = pa.Table.from_pandas(df, preserve_index=False)
        # mode="overwrite" matches the PITEngine's read-merge-write contract:
        # the caller has already performed dedup, so we want to replace the
        # dataset's logical contents (a new version is created either way).
        lance.write_dataset(table, physical, mode="overwrite")

    def read_table(self, path: str) -> pd.DataFrame:
        # Prefer Lance; fall back to legacy Parquet if the Lance dataset
        # doesn't exist yet. This lets the Lance backend coexist with
        # pre-existing Parquet data during incremental migration — the first
        # write of any dataset upgrades it to Lance, while reads work
        # against either format on day one.
        lance_path = self._abs(_dataset_suffix(path))
        if self._exists_raw(lance_path):
            return lance.dataset(lance_path).to_table().to_pandas()
        parquet_path = self._abs(
            path if path.endswith(".parquet") else f"{path}.parquet"
        )
        if self._exists_raw(parquet_path):
            return pd.read_parquet(parquet_path)
        # Neither format exists; let Lance raise its native NotFound so the
        # error message points at the canonical (Lance) path.
        return lance.dataset(lance_path).to_table().to_pandas()

    def _exists_raw(self, physical: str) -> bool:
        """Raw existence check at a fully-resolved physical path."""
        if self._is_local:
            return Path(physical).exists()
        import fsspec
        fs = fsspec.filesystem(self.root.split("://", 1)[0])
        return fs.exists(physical)

    # ----- binary -----

    def write_bytes(self, path: str, data: bytes) -> None:
        physical = self._abs(path)
        if self._is_local:
            p = Path(physical)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        else:
            # For cloud roots, delegate to fsspec — same shape as
            # AzureMLStorage.write_bytes. Lazy import keeps dev light.
            import fsspec
            with fsspec.open(physical, "wb") as f:
                f.write(data)

    def read_bytes(self, path: str) -> bytes:
        physical = self._abs(path)
        if self._is_local:
            return Path(physical).read_bytes()
        import fsspec
        with fsspec.open(physical, "rb") as f:
            return f.read()

    def append_bytes(self, path: str, data: bytes) -> None:
        physical = self._abs(path)
        if self._is_local:
            p = Path(physical)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "ab") as f:
                f.write(data)
            return
        # Cloud append falls back to read-modify-write, same as AzureMLStorage.
        existing = b""
        if self.exists(path):
            existing = self.read_bytes(path)
        self.write_bytes(path, existing + data)

    def exists(self, path: str) -> bool:
        # Tabular existence is true if EITHER the Lance dataset OR a legacy
        # Parquet file exists. Plus we honor the bare `path` for binary blobs
        # (gzipped raw archives) that don't carry either suffix.
        #
        # The Parquet fallback matters during the Parquet→Lance migration:
        # ingest CLIs check exists() before doing read-merge-write, and we
        # want those checks to succeed against either format.
        bare = _strip_parquet_suffix(path)
        candidates = [bare, _dataset_suffix(bare), f"{bare}.parquet"]
        for c in candidates:
            if self._exists_raw(self._abs(c)):
                return True
        return False

    def list(self, prefix: str) -> Iterable[str]:
        """List logical paths under `prefix`.

        Lance datasets are directories; we surface them as a single logical
        path (the directory name, no `.lance` suffix) rather than enumerating
        their internal fragment files. This matches what LocalStorage does
        with parquet files and lets the PIT engine's directory scan find
        per-CIK datasets the same way.
        """
        if self._is_local:
            base = Path(self._abs(prefix))
            if not base.exists():
                return []
            out: list[str] = []
            for p in base.iterdir():
                if p.is_dir() and p.name.endswith(".lance"):
                    # Strip .lance and report the logical path
                    rel = p.relative_to(self.root).as_posix()[:-6]
                    out.append(rel)
                elif p.is_file():
                    out.append(p.relative_to(self.root).as_posix())
                elif p.is_dir():
                    # Recurse into non-Lance subdirs (e.g., nested prefixes)
                    for sub in p.rglob("*"):
                        if sub.is_dir() and sub.name.endswith(".lance"):
                            rel = sub.relative_to(self.root).as_posix()[:-6]
                            out.append(rel)
                        elif sub.is_file():
                            out.append(sub.relative_to(self.root).as_posix())
            return sorted(out)
        # Cloud: similar logic via fsspec.
        import fsspec
        fs = fsspec.filesystem(self.root.split("://", 1)[0])
        physical_prefix = self._abs(prefix)
        if not fs.exists(physical_prefix):
            return []
        out_c: list[str] = []
        for entry in fs.find(physical_prefix, detail=False):
            rel = entry[len(self.root):].lstrip("/")
            if "/" in rel and rel.split("/")[-2].endswith(".lance"):
                # File inside a Lance dataset; collapse to dataset path
                ds_path = rel.rsplit("/", 1)[0][:-6]
                if ds_path not in out_c:
                    out_c.append(ds_path)
            else:
                out_c.append(rel)
        return sorted(out_c)
