"""Register a model from a completed job's outputs.

Run: poetry run python -m aml.register_model <job_name>
"""
import sys
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from aml.client import get_ml_client


def main(job_name: str, model_name: str = "cifar10-smallcnn") -> None:
    ml = get_ml_client()
    model = Model(
        path=f"azureml://jobs/{job_name}/outputs/model_dir",
        name=model_name,
        type=AssetTypes.CUSTOM_MODEL,
        description=f"From job {job_name}",
    )
    registered = ml.models.create_or_update(model)
    print(f"registered: {registered.name}:{registered.version}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: register_model.py <job_name>")
    main(sys.argv[1])
