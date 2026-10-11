import json

import mlflow
import torch
from mlflow import MlflowClient
from torch import nn
from torchvision import models

NOM = "plant-disease-classifier"

classes = json.load(open("models/classes.json"))
model = models.mobilenet_v3_small()
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
model.eval()

mlflow.set_experiment("plant-disease")
with mlflow.start_run(run_name="enregistrement-registre"):
    mlflow.log_param("architecture", "mobilenet_v3_small")
    mlflow.log_param("nb_classes", len(classes))
    mlflow.log_dict({"classes": classes}, "classes.json")
    mlflow.pytorch.log_model(model, name="model", registered_model_name=NOM)

client = MlflowClient()
version = client.get_latest_versions(NOM)[0].version
client.set_registered_model_alias(NOM, "production", version)
print(f"Modèle {NOM} enregistré : version {version}, alias 'production'")