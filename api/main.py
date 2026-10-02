import io
import json
import os
from collections import Counter
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from torch import nn
from torchvision import models, transforms

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

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


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(classes)}


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
    return {
        "user": user,
        "predictions": [
            {"label": classes[i], "probability": round(p, 4)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ],
    }


@app.get("/usage")
def get_usage(user: str = Depends(get_user)):
    return {"user": user, "requests": usage[user]}