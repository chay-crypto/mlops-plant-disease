import csv
import io
import json
import os
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from scipy.stats import ks_2samp
from torch import nn
from torchvision import models, transforms

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
LOG_FILE = Path(os.getenv("LOG_FILE", "logs/predictions.jsonl"))
REF_FILE = Path(os.getenv("REF_FILE", "monitoring/reference.csv"))
FEATURES = ["brightness", "contrast", "red", "green", "blue", "confidence"]

# Format de la variable : "utilisateur:cle,utilisateur2:cle2"
API_KEYS = {}
for pair in os.getenv("API_KEYS", "").split(","):
    if ":" in pair:
        user, key = pair.split(":", 1)
        API_KEYS[key.strip()] = user.strip()

classes = json.loads((MODEL_DIR / "classes.json").read_text())
model = models.mobilenet_v3_small()
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

usage = Counter()
app = FastAPI(title="Plant Disease API")


def get_user(x_api_key: str = Header(default="")):
    user = API_KEYS.get(x_api_key)
    if user is None:
        raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide")
    return user


def image_features(image):
    arr = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "red": float(arr[:, :, 0].mean()),
        "green": float(arr[:, :, 1].mean()),
        "blue": float(arr[:, :, 2].mean()),
    }


def log_prediction(user, label, features):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(), "user": user, "label": label, **features}
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(classes), "instance": socket.gethostname()}


@app.post("/predict")
async def predict(file: UploadFile = File(...), user: str = Depends(get_user)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier image invalide")

    with torch.no_grad():
        probs = torch.softmax(model(tf(image).unsqueeze(0)), dim=1)[0]
    top = torch.topk(probs, 3)
    usage[user] += 1

    features = image_features(image)
    features["confidence"] = float(top.values[0])
    log_prediction(user, classes[int(top.indices[0])], features)

    return {
        "user": user,
        "predictions": [
            {"label": classes[i], "probability": round(p, 4)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ],
    }


@app.get("/usage")
def get_usage(user: str = Depends(get_user)):
    n = 0
    if LOG_FILE.exists():
        with LOG_FILE.open() as f:
            n = sum(1 for line in f if json.loads(line)["user"] == user)
    return {"user": user, "requests": n}


@app.get("/drift")
def drift(last: int = 200, user: str = Depends(get_user)):
    if not REF_FILE.exists():
        raise HTTPException(status_code=503, detail="Fichier de référence absent")
    with REF_FILE.open() as f:
        reference = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]

    current = []
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()[-last:]
        current = [json.loads(line) for line in lines]
    if len(current) < 30:
        return {"status": "pas assez de données (30 minimum)", "n_current": len(current)}

    report = {}
    for feat in FEATURES:
        ref_values = [r[feat] for r in reference]
        cur_values = [c[feat] for c in current]
        p_value = float(ks_2samp(ref_values, cur_values).pvalue)
        report[feat] = {
            "reference_mean": round(float(np.mean(ref_values)), 3),
            "current_mean": round(float(np.mean(cur_values)), 3),
            "p_value": round(p_value, 4),
            "drift": p_value < 0.01,
        }

    n_drift = sum(r["drift"] for r in report.values())
    return {
        "n_reference": len(reference),
        "n_current": len(current),
        "drift_detected": n_drift >= 2,
        "features": report,
    }