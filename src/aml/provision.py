"""One-shot: create or update the AzureML compute cluster.

Run: poetry run python -m aml.provision
"""
import os
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError
from aml.client import get_ml_client


def ensure_cluster(name: str, size: str = "Standard_DS3_v2", max_nodes: int = 2) -> None:
    ml = get_ml_client()
    try:
        existing = ml.compute.get(name)
        print(f"compute '{name}' already exists ({existing.size}, max={existing.max_instances})")
        return
    except ResourceNotFoundError:
        pass

    cluster = AmlCompute(
        name=name,
        type="amlcompute",
        size=size,
        min_instances=0,
        max_instances=max_nodes,
        idle_time_before_scale_down=180,
        tier="Dedicated",
    )
    ml.compute.begin_create_or_update(cluster).result()
    print(f"created compute '{name}' ({size}, 0→{max_nodes})")


if __name__ == "__main__":
    ensure_cluster(os.environ.get("AZUREML_COMPUTE_NAME", "cpu-cluster"))
