"""Cross-pipeline observability helpers.

`resources` — pre-flight disk + memory checks so ingest CLIs fail fast on
              insufficient host capacity instead of OOMing mid-job.
`log`       — structured (JSON-line) event emission to `state/job_log.jsonl`
              with opt-in stdout mirroring via `BWM_STRUCTURED_LOG=1`.

Both modules are additive — existing print-based progress lines continue to
work unchanged. Operators can tail one unified event stream across all
pipelines instead of grepping per-CLI stdout.
"""
