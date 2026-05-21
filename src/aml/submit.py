"""Submit a training job to AzureML.

Run: poetry run python -m aml.submit
"""
import os
from pathlib import Path

from azure.ai.ml import command, Output
from azure.ai.ml.entities import Environment
from aml.client import get_ml_client

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "src" / "training"


def build_env() -> Environment:
    return Environment(
        name="pytorch-cpu",
        description="PyTorch 2.4 CPU + MLflow",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(TRAINING_DIR / "conda.yaml"),
    )


def main() -> None:
    ml = get_ml_client()
    compute = os.environ.get("AZUREML_COMPUTE_NAME", "cpu-cluster")

    job = command(
        code=str(TRAINING_DIR),
        command=(
            "python train.py "
            "--data-dir ${{inputs.data_dir}} "
            "--output-dir ${{outputs.model_dir}} "
            "--epochs ${{inputs.epochs}} "
            "--batch-size ${{inputs.batch_size}} "
            "--lr ${{inputs.lr}}"
        ),
        inputs={
            "data_dir": "./data",
            "epochs": 5,
            "batch_size": 128,
            "lr": 1e-3,
        },
        outputs={"model_dir": Output(type="uri_folder", mode="rw_mount")},
        environment=build_env(),
        compute=compute,
        experiment_name="cifar10-smallcnn",
        display_name="cifar10-smallcnn-baseline",
    )

    submitted = ml.jobs.create_or_update(job)
    print(f"submitted: {submitted.name}")
    print(f"studio:    {submitted.studio_url}")


if __name__ == "__main__":
    main()
