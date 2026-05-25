"""Per-process run identifier for log disambiguation in parallel AML runs.

When multiple modality ingests run simultaneously on Azure ML, their stdout
streams interleave in the aggregator. Prefixing every user-visible line with
a stable run identifier lets operators filter by job after the fact.

Resolution order for the identifier:
  1. AZUREML_RUN_ID — set by AML on every job step.
  2. fallback: a short random UUID prefix (`local-xxxxxxxx`), generated once
     per process so all lines from the same local run share the same tag.
"""
from __future__ import annotations

import os
import uuid
from functools import lru_cache


@lru_cache(maxsize=1)
def get_run_id() -> str:
    return os.environ.get("AZUREML_RUN_ID") or f"local-{uuid.uuid4().hex[:8]}"


def tag(msg: str) -> str:
    """Prefix `msg` with the run id. Use at every user-visible print site."""
    return f"[{get_run_id()}] {msg}"
