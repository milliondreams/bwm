"""Submit a non-EDGAR modality ingest job to Azure ML compute.

Generic AML wrapper for the M-stage modalities (market, patents, news,
hiring). Each modality has its own CLI; this wrapper just maps a
`--modality` flag to the right `python -m data.cli.ingest_*` command.

Run:
    # Hiring (smallest, fastest — ~5 min)
    uv run python -m aml.modality_job --modality hiring

    # Market (whole registry, daily bars 2015→today)
    uv run python -m aml.modality_job --modality market --scope all

    # Market pilot
    uv run python -m aml.modality_job --modality market --scope 100

    # Patents — needs PATENTSVIEW_API_KEY in env to be useful
    uv run python -m aml.modality_job --modality patents --scope all

    # News
    uv run python -m aml.modality_job --modality news --news-start 2020-01-01 --news-end 2025-01-01

Writes to the SHARED named output path `bwm/data/v1/` so all modalities
land in the same canonical tree alongside EDGAR's output. Different
modalities write disjoint subdirectories (`canonical/market/`,
`canonical/patents/`, etc.) so there's no collision; watermark stores are
partitioned by source key (`yahoo_finance`, `patentsview`, `gdelt:events2`,
`bls:jolts`) so no contention with EDGAR's keys.

IMPORTANT: do not submit while an EDGAR `--all` job is still running.
EDGAR is the single largest writer to this namespace; parallel writers
should wait until it completes or stabilizes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from azure.ai.ml import Output, command
from azure.ai.ml.entities import Environment
from aml.client import get_ml_client

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
INGEST_ENV_YAML = ROOT / "src" / "aml" / "ingest_env.yaml"


def build_env() -> Environment:
    return Environment(
        name="bwm-ingest",
        description="CPU env for EDGAR + macro + market ingest. No torch/CUDA.",
        image="mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:latest",
        conda_file=str(INGEST_ENV_YAML),
    )


def _build_modality_cmd(args) -> str:
    """Translate --modality + flags into the right ingest CLI command line."""
    m = args.modality
    if m == "hiring":
        return (
            f"python -m data.cli.ingest_hiring "
            f"--start-year {args.hiring_start_year} --end-year {args.hiring_end_year}"
        )
    if m == "macro":
        # ingest_macro pulls DEFAULT_SERIES if --series is omitted.
        return "python -m data.cli.ingest_macro"
    if m == "market":
        scope = "--all" if args.scope == "all" else f"--limit-ciks {int(args.scope)}"
        return (
            f"python -m data.cli.ingest_market_all {scope} "
            f"--start {args.market_start} --end {args.market_end} "
            f"--interval {args.market_interval} --concurrency {args.concurrency}"
        )
    if m == "patents":
        scope = "--all" if args.scope == "all" else f"--limit-ciks {int(args.scope)}"
        return (
            f"python -m data.cli.ingest_patents_all {scope} "
            f"--start-grant {args.patents_start} --end-grant {args.patents_end} "
            f"--concurrency {args.concurrency}"
        )
    if m == "news":
        return (
            f"python -m data.cli.ingest_news "
            f"--start {args.news_start} --end {args.news_end} "
            f"--registry-filter {args.news_filter}"
        )
    raise ValueError(f"unknown modality: {m}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True,
                    choices=("hiring", "macro", "market", "patents", "news"))
    ap.add_argument("--scope", default="all",
                    help="for market/patents: 'all' or an integer (--limit-ciks N)")
    ap.add_argument("--concurrency", type=int, default=4)

    # Per-modality date / scope overrides
    ap.add_argument("--hiring-start-year", type=int, default=2015)
    ap.add_argument("--hiring-end-year", type=int, default=2026)
    ap.add_argument("--market-start", default="2015-01-01")
    ap.add_argument("--market-end", default="2026-01-01")
    ap.add_argument("--market-interval", default="1d", choices=("1d", "1wk", "1mo"))
    ap.add_argument("--patents-start", default="2000-01-01")
    ap.add_argument("--patents-end", default="2026-01-01")
    ap.add_argument("--news-start", default="2020-01-01")
    ap.add_argument("--news-end", default="2025-01-01")
    ap.add_argument("--news-filter", default="on", choices=("on", "off"))

    ap.add_argument("--display-name", default="")
    args = ap.parse_args()

    ml = get_ml_client()
    compute_name = os.environ.get("AZUREML_COMPUTE_NAME", "cpu-cluster")

    ingest_cmd = _build_modality_cmd(args)
    default_display = f"{args.modality}-ingest-{args.scope}"

    # Bootstrap differs by modality — patents needs PATENTSVIEW_API_KEY, news
    # has no key, hiring has optional BLS_API_KEY, macro needs FRED_API_KEY.
    env_passthrough = ["PYTHONUNBUFFERED"]
    env_vars: dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    for k in ("PATENTSVIEW_API_KEY", "BLS_API_KEY", "FRED_API_KEY", "SEC_USER_AGENT"):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v
            env_passthrough.append(k)

    job_command = "\n".join([
        "set -euo pipefail",
        "export PYTHONPATH=src:${PYTHONPATH:-}",
        "export BWM_DATA_ROOT=${{outputs.data_root}}",
        "echo '== bootstrap =='",
        "python -c 'import sys, pandas, pyarrow, requests; print(sys.version)'",
        # Re-seed the registry only when the modality needs it; news + market
        # + patents resolve CIKs from the registry, so seeding is mandatory.
        # (hiring + macro use pseudo-entity_ids and don't need the registry.)
        *(["echo '== seed entity registry =='", "python -m data.sources.sec_tickers"]
          if args.modality in ("market", "patents", "news") else []),
        "echo '== ingest =='",
        ingest_cmd,
        "echo '== validation (best effort) =='",
        f"python -m data.cli.validate_{args.modality} || true",
    ])

    job = command(
        code=str(SRC_DIR),
        command=job_command,
        environment=build_env(),
        compute=compute_name,
        outputs={
            "data_root": Output(
                type="uri_folder",
                path="azureml://datastores/workspaceblobstore/paths/bwm/data/v1/",
                mode="rw_mount",
            ),
        },
        environment_variables=env_vars,
        experiment_name="bwm-phase-a-ingest",
        display_name=args.display_name or default_display,
        description=(
            f"Phase A.2 modality ingest: {args.modality}. "
            f"Output → workspace default blob datastore under `bwm/data/v1/`. "
            f"Disjoint canonical subtree from EDGAR; watermark store partitioned by source key."
        ),
    )
    submitted = ml.jobs.create_or_update(job)
    print(f"submitted: {submitted.name}")
    print(f"studio:    {submitted.studio_url}")
    print(f"compute:   {compute_name}")
    print(f"command:   {ingest_cmd}")


if __name__ == "__main__":
    main()
