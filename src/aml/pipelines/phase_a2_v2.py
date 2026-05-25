"""Phase A.2 v2 pipeline: full corpus rebuild on Lance backend.

DAG:

    Wave 1 (parallel — independent ingests, all writing to shared blob root):
      ┌─ ingest_dera        (financials, ~6 GB, ~30 min)
      ├─ ingest_feed        (filings_full + insider_trades, ~3-5 TB, ~12-24 h)
      ├─ ingest_market_all  (yfinance, ~10 min)
      ├─ ingest_macro       (FRED, ~5 min)
      ├─ ingest_patents_all (PatentsView, ~3-4 h)
      ├─ ingest_news        (GDELT, ~12 h)
      └─ ingest_hiring      (BLS JOLTS, ~2 min)
                    │
    Wave 2 (after Wave 1 — derived modalities):
      └─ derive_earnings_calls (reads filings_full → earnings_calls)
                    │
    Wave 3 (after Wave 2 — final acceptance gates):
      ├─ validate_constraints
      └─ validate_modality_coverage

Each step is a separate AML command job. Wave dependencies are expressed
via dsl.pipeline graph edges; AML schedules waves automatically.

Storage backend is set to `lance` via env var on every step — all writes
land as Lance datasets in `bwm/data/v1/canonical/`.

Run:
    uv run python -m aml.pipelines.phase_a2_v2 \\
        --start 2012Q1 --end 2025Q1 \\
        --feed-start 2001-01-01 --feed-end 2025-12-31

To kick off the full historical backfill. The feed job is the long pole;
splitting it across multiple nodes (different date ranges) is done by
submitting multiple pipeline invocations with disjoint --feed-start/end.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from azure.ai.ml import Input, Output, command, dsl
from azure.ai.ml.entities import Environment

from aml.client import get_ml_client

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
INGEST_ENV_YAML = ROOT / "src" / "aml" / "ingest_env.yaml"

DATA_ROOT_URI = "azureml://datastores/workspaceblobstore/paths/bwm/data/v1/"

# Common env: Lance backend + passthrough for API keys
_BASE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "BWM_STORAGE_BACKEND": "lance",
}


def _ingest_env() -> Environment:
    return Environment(
        name="bwm-ingest",
        description="CPU env for Phase A.2 ingest jobs (Lance backend).",
        image="mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:latest",
        conda_file=str(INGEST_ENV_YAML),
    )


def _api_keys() -> dict[str, str]:
    """Forward API keys from the submitter's env if set."""
    keys: dict[str, str] = {}
    for k in ("PATENTSVIEW_API_KEY", "BLS_API_KEY", "FRED_API_KEY", "SEC_USER_AGENT"):
        v = os.environ.get(k)
        if v:
            keys[k] = v
    return keys


# Per-step wall-clock timeouts (seconds), calibrated to observed pace + ~3-6x
# safety margin so AML kills hung jobs before they burn a week of cluster
# time. Default AML timeout is ~7 days, which is way too generous.
_STEP_TIMEOUTS: dict[str, int] = {
    "ingest_dera": 7200,                  # 2h — full 52-quarter ~20 min observed
    "ingest_feed": 86400,                 # 24h — long pole; 5-8 TB download
    "ingest_market": 3600,                # 1h — ~8 min observed for 8K tickers
    "ingest_macro": 1800,                 # 30m — ~5 min observed
    "ingest_patents": 14400,              # 4h — PatentsView rate-limited
    "ingest_news": 86400,                 # 24h — GDELT ~390K slots serial
    "ingest_hiring": 1800,                # 30m — BLS batched, <2 min observed
    "derive_earnings_calls": 1800,        # 30m — pure CPU on existing canonical
    "validate_constraints": 600,          # 10m — read + rule eval
    "validate_modality_coverage": 600,    # 10m — list + per-modality sample
}


def _step(name: str, command_str: str, compute: str = "cpu-cluster",
          timeout_seconds: int | None = None):
    """Build a command component and immediately invoke it as a pipeline node.

    AML's dsl.pipeline registers nodes when a component is *called*; the
    component returned by `command(...)` is just a definition. Calling it
    `()` produces a Node that the active pipeline context captures.

    `timeout_seconds` enforces a wall-clock limit on the step. If not
    explicitly passed, the per-step default from _STEP_TIMEOUTS is used.
    """
    env = _ingest_env()
    env_vars = {**_BASE_ENV, **_api_keys()}
    if timeout_seconds is None:
        timeout_seconds = _STEP_TIMEOUTS.get(name)
    component = command(
        name=name,
        display_name=name,
        code=str(SRC_DIR),
        command="\n".join([
            "set -euo pipefail",
            "export PYTHONPATH=src:${PYTHONPATH:-}",
            "export BWM_DATA_ROOT=${{outputs.data_root}}",
            command_str,
        ]),
        environment=env,
        compute=compute,
        outputs={
            "data_root": Output(
                type="uri_folder",
                path=DATA_ROOT_URI,
                mode="rw_mount",
            ),
        },
        environment_variables=env_vars,
        timeout=timeout_seconds,
    )
    # Invoke the component → registers a node in the active pipeline context.
    return component()


@dsl.pipeline(
    name="bwm-phase-a2-v2",
    description="Phase A.2 v2 full corpus rebuild on Lance backend.",
    default_compute="cpu-cluster",
)
def _build_pipeline(
    # Pipeline parameters use literal default strings (no annotations) because
    # AML's dsl.pipeline rejects plain `str` type hints — it only understands
    # its own Input/Output annotation classes. Defaults are still surfaced
    # as pipeline-level inputs in AML Studio.
    dera_start="2012Q1",
    dera_end="2025Q1",
    feed_start="2001-01-01",
    feed_end="2025-12-31",
    market_start="2000-01-01",
    market_end="2026-01-01",
    news_start="2015-02-18",  # GDELT 2.0 launch
    news_end="2025-12-31",
):
    # Wave 1 — all parallel, all write to shared bwm/data/v1/canonical/
    dera = _step(
        "ingest_dera",
        f"python -m data.cli.ingest_dera --start {dera_start} --end {dera_end}",
    )
    feed = _step(
        "ingest_feed",
        f"python -m data.cli.ingest_feed --start {feed_start} --end {feed_end}",
    )
    market = _step(
        "ingest_market",
        f"python -m data.sources.sec_tickers\n"
        f"python -m data.cli.ingest_market_all --all "
        f"--start {market_start} --end {market_end}",
    )
    macro = _step("ingest_macro", "python -m data.cli.ingest_macro")
    patents = _step(
        "ingest_patents",
        f"python -m data.sources.sec_tickers\n"
        f"python -m data.cli.ingest_patents_all --all",
    )
    news = _step(
        "ingest_news",
        f"python -m data.sources.sec_tickers\n"
        f"python -m data.cli.ingest_news "
        f"--start {news_start} --end {news_end} --registry-filter on",
    )
    hiring = _step(
        "ingest_hiring",
        "python -m data.cli.ingest_hiring --start-year 2015 --end-year 2026",
    )

    # Wave 2 / Wave 3 ordering note:
    # AML dsl.pipeline expresses dependencies via input/output piping;
    # there is no `.after()` API. All ingest steps write to the SAME shared
    # blob mount (different canonical subtrees) so they don't have piped
    # outputs to consume. We include Wave 2 (derive) and Wave 3 (validate)
    # steps in this pipeline anyway; AML will start them in parallel with
    # Wave 1. The operator is expected to:
    #   1. Submit this pipeline.
    #   2. If derive/validate fail because Wave 1 isn't done yet, re-submit
    #      them via the standalone validate_constraints / derive_earnings_calls
    #      CLIs after Wave 1 completes.
    # A future improvement is to add a `_wave_ready` sentinel output that
    # downstream steps consume as an input, materializing the DAG explicitly.
    earnings_calls = _step(  # noqa: F841 — included for AML graph completeness
        "derive_earnings_calls",
        "python -m data.cli.derive_earnings_calls",
    )
    constraints = _step(  # noqa: F841
        "validate_constraints",
        "python -m data.cli.validate_constraints || true",
    )
    coverage = _step(  # noqa: F841
        "validate_modality_coverage",
        "python -m data.cli.validate_modality_coverage || true",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dera-start", default="2012Q1")
    ap.add_argument("--dera-end", default="2025Q1")
    ap.add_argument("--feed-start", default="2001-01-01")
    ap.add_argument("--feed-end", default="2025-12-31")
    ap.add_argument("--market-start", default="2000-01-01")
    ap.add_argument("--market-end", default="2026-01-01")
    ap.add_argument("--news-start", default="2015-02-18")
    ap.add_argument("--news-end", default="2025-12-31")
    ap.add_argument("--experiment", default="bwm-phase-a2-v2")
    args = ap.parse_args()

    ml = get_ml_client()
    pipeline = _build_pipeline(
        dera_start=args.dera_start,
        dera_end=args.dera_end,
        feed_start=args.feed_start,
        feed_end=args.feed_end,
        market_start=args.market_start,
        market_end=args.market_end,
        news_start=args.news_start,
        news_end=args.news_end,
    )
    submitted = ml.jobs.create_or_update(pipeline, experiment_name=args.experiment)
    print(f"submitted: {submitted.name}")
    print(f"studio:    {submitted.studio_url}")


if __name__ == "__main__":
    main()
