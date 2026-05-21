"""Submit an EDGAR ingest job to Azure ML compute.

Pattern:
  1. Snapshot the project's `src/` tree as the job code source
  2. Build (or reuse) the bwm-ingest conda environment image
  3. Submit a CommandJob that runs `data.cli.ingest_edgar_full` on the
     CPU cluster, writing canonical outputs to the workspace's default
     blob datastore under a versioned subdirectory

Usage:
    # Pilot — 10 CIKs, 3 filings per modality each (~minutes)
    uv run python -m aml.ingest_job --pilot

    # Wider pilot — 100 CIKs, 5 filings each (~30 min)
    uv run python -m aml.ingest_job --limit-ciks 100 --max-filings 5

    # Full corpus — ~10K CIKs, all filings, ~47 wall hours
    uv run python -m aml.ingest_job --all
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
    """The CPU ingest environment. Reusable across pilot + full runs.

    Versioning: Azure ML environments are content-addressed on the conda
    spec + base image. Re-submitting the same env yaml after a no-op edit
    reuses the existing image (no rebuild); changing a pin triggers a
    fresh build (~5-15 min one-time cost).
    """
    return Environment(
        name="bwm-ingest",
        description="CPU env for EDGAR + macro + market ingest. No torch/CUDA.",
        # CPU base image — no CUDA, smaller container.
        image="mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:latest",
        conda_file=str(INGEST_ENV_YAML),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    scope = ap.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--pilot", action="store_true",
        help="Submit a tiny pilot: 10 CIKs, 3 filings/modality each",
    )
    scope.add_argument(
        "--limit-ciks", type=int, default=0,
        help="Submit a wider pilot of N CIKs (still capped per-modality unless --max-filings)",
    )
    scope.add_argument(
        "--all", action="store_true",
        help="Full corpus backfill (~10K CIKs, ~47h wall time at 9 req/s)",
    )
    ap.add_argument(
        "--max-filings", type=int, default=0,
        help="Per-CIK per-modality cap (0 = no cap)",
    )
    ap.add_argument(
        "--concurrency", type=int, default=4,
        help="Number of CIKs in flight at once (rate-limit is global so this is "
             "about parallelism between modalities + per-CIK retries, not throughput)",
    )
    ap.add_argument(
        "--skip-financials", action="store_true",
    )
    ap.add_argument(
        "--skip-form4", action="store_true",
    )
    ap.add_argument(
        "--skip-text", action="store_true",
    )
    ap.add_argument(
        "--display-name", default="",
        help="Override the AzureML job display name (defaults to scope-derived)",
    )
    args = ap.parse_args()

    ml = get_ml_client()
    compute_name = os.environ.get("AZUREML_COMPUTE_NAME", "cpu-cluster")

    # Stage 5 Hotfix: --all defaults to --max-filings 80 to avoid the
    # 14-day-runtime problem from the original unconfigured submission.
    # 80 filings/modality covers ≥20 years (the spec FR target ≥80 quarters)
    # since SEC submissions are returned newest-first.
    if args.pilot:
        scope_args = "--limit-ciks 10 --max-filings-per-cik 3"
        default_display = "edgar-ingest-pilot-10ciks"
        output_subpath = "pilot"
    elif args.limit_ciks > 0:
        max_filings_arg = (
            f"--max-filings-per-cik {args.max_filings}" if args.max_filings > 0 else ""
        )
        scope_args = f"--limit-ciks {args.limit_ciks} {max_filings_arg}".strip()
        default_display = f"edgar-ingest-pilot-{args.limit_ciks}ciks"
        output_subpath = "pilot"
    else:  # --all
        # Force the cap on for --all even if the operator forgot to pass it.
        effective_cap = args.max_filings if args.max_filings > 0 else 80
        scope_args = f"--all --max-filings-per-cik {effective_cap}"
        default_display = "edgar-ingest-full-corpus"
        output_subpath = "v1"

    skip_flags: list[str] = []
    if args.skip_financials:
        skip_flags.append("--skip-financials")
    if args.skip_form4:
        skip_flags.append("--skip-form4")
    if args.skip_text:
        skip_flags.append("--skip-text")
    skip_str = " ".join(skip_flags)

    # The job command sets BWM_DATA_ROOT to the mounted output folder so
    # the existing ingest CLIs (which already honor this env var) write
    # canonical/, raw/, state/, entities/ into the AzureML datastore.
    # ${{outputs.data_root}} is Azure ML's templating syntax — substituted
    # with the actual mount path at job runtime.
    ingest_cmd = (
        f"python -m data.cli.ingest_edgar_full {scope_args} {skip_str} "
        f"--concurrency {args.concurrency}"
    )
    job_command = "\n".join([
        "set -euo pipefail",
        "export PYTHONPATH=src:${PYTHONPATH:-}",
        "export BWM_DATA_ROOT=${{outputs.data_root}}",
        "echo '== bootstrap =='",
        "python -c 'import sys, aiohttp, lxml, pandas, pyarrow; "
        "print(sys.version); print(\"aiohttp\", aiohttp.__version__); "
        "print(\"lxml\", lxml.__version__)'",
        "echo '== seed entity registry =='",
        "python -m data.sources.sec_tickers",
        "echo '== ingest =='",
        ingest_cmd,
        "echo '== summary =='",
        "python -m data.cli.validate_stage5 || true",
    ])

    job = command(
        code=str(SRC_DIR),
        command=job_command,
        environment=build_env(),
        compute=compute_name,
        outputs={
            # Named output path (R5: mid-run failure recovery). Pilots and
            # `--all` write to separate paths so pilot data doesn't pollute
            # the canonical, and re-submissions inherit watermarks from the
            # prior run for true resumability.
            "data_root": Output(
                type="uri_folder",
                path=f"azureml://datastores/workspaceblobstore/paths/bwm/data/{output_subpath}/",
                mode="rw_mount",
            ),
        },
        environment_variables={
            "SEC_USER_AGENT": os.environ.get(
                "SEC_USER_AGENT", "norn-bwm research rohit@dragonscale.ai"
            ),
            "PYTHONUNBUFFERED": "1",
        },
        experiment_name="bwm-phase-a-ingest",
        display_name=args.display_name or default_display,
        description=(
            "Stage 5 corpus ingest: SEC EDGAR (financials via per-filing XBRL, "
            "Form 4, filings text). Output → workspace default blob datastore "
            "under `data_root`. Resumable via per-CIK watermarks."
        ),
    )
    submitted = ml.jobs.create_or_update(job)
    print(f"submitted: {submitted.name}")
    print(f"studio:    {submitted.studio_url}")
    print(f"compute:   {compute_name}")
    print(f"scope:     {scope_args}")


if __name__ == "__main__":
    main()
