"""Submit a read-only validation job against the EDGAR canonical blob.

Runs `validate_stage5` (Phase A acceptance) and a configurable list of
additional validators inside the AML cluster, mounting `bwm/data/v1/`
read-only so it cannot perturb canonical data.

Run:
    uv run python -m aml.validate_job
    uv run python -m aml.validate_job --validators validate_stage5 validate_constraints
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from azure.ai.ml import Input, command
from azure.ai.ml.entities import Environment
from aml.client import get_ml_client

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
INGEST_ENV_YAML = ROOT / "src" / "aml" / "ingest_env.yaml"

DEFAULT_VALIDATORS = ["validate_stage5"]


def build_env() -> Environment:
    return Environment(
        name="bwm-ingest",
        description="CPU env reused for read-only validation.",
        image="mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:latest",
        conda_file=str(INGEST_ENV_YAML),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validators", nargs="+", default=DEFAULT_VALIDATORS)
    ap.add_argument("--display-name", default="validate-edgar-canonical")
    args = ap.parse_args()

    ml = get_ml_client()
    compute_name = os.environ.get("AZUREML_COMPUTE_NAME", "cpu-cluster")

    validator_cmds = "\n".join(
        f"echo '== {v} ==' && python -m data.cli.{v}"
        for v in args.validators
    )

    job_command = "\n".join([
        "set -euo pipefail",
        "export PYTHONPATH=src:${PYTHONPATH:-}",
        "export BWM_DATA_ROOT=${{inputs.data_root}}",
        "echo '== bootstrap =='",
        "python -c 'import sys, pandas, pyarrow; print(sys.version)'",
        "echo \"BWM_DATA_ROOT=$BWM_DATA_ROOT\"",
        "ls -la $BWM_DATA_ROOT | head -20",
        validator_cmds,
    ])

    job = command(
        code=str(SRC_DIR),
        command=job_command,
        environment=build_env(),
        compute=compute_name,
        inputs={
            "data_root": Input(
                type="uri_folder",
                path="azureml://datastores/workspaceblobstore/paths/bwm/data/v1/",
                mode="ro_mount",
            ),
        },
        environment_variables={"PYTHONUNBUFFERED": "1"},
        experiment_name="bwm-phase-a-validate",
        display_name=args.display_name,
        description=(
            "Read-only validation pass against bwm/data/v1/ canonical. "
            f"Validators: {', '.join(args.validators)}"
        ),
    )
    submitted = ml.jobs.create_or_update(job)
    print(f"submitted: {submitted.name}")
    print(f"studio:    {submitted.studio_url}")
    print(f"validators: {args.validators}")


if __name__ == "__main__":
    main()
