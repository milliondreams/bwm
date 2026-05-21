import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def get_ml_client() -> MLClient:
    sub = os.environ["AZURE_SUBSCRIPTION_ID"]
    rg = os.environ["AZURE_RESOURCE_GROUP"]
    ws = os.environ["AZUREML_WORKSPACE_NAME"]
    return MLClient(DefaultAzureCredential(), sub, rg, ws)
