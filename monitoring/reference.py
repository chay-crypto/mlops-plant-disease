import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

DATA = Path("data/plantvillage/raw/color")
OUT = Path("monitoring/reference.csv")
PER_CLASS = 10
FEATURES = ["brightness", "contrast", "red", "green", "blue", "confidence"]

classes = json.load(open("models/classes.json"))
model = models.mobilenet_v3_small()
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

rows = []
for folder in sorted(p for p in DATA.iterdir() if p.is_dir()):
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in images[-PER_CLASS:]:
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        with torch.no_grad():
            probs = torch.softmax(model(tf(image).unsqueeze(0)), dim=1)[0]
        rows.append({
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "red": float(arr[:, :, 0].mean()),
            "green": float(arr[:, :, 1].mean()),
            "blue": float(arr[:, :, 2].mean()),
            "confidence": float(probs.max()),
        })

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FEATURES)
    writer.writeheader()
    writer.writerows(rows)

print(f"{len(rows)} images de référence enregistrées dans {OUT}")