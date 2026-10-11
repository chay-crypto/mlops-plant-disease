import io
import json
import random
import time
from collections import defaultdict

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DATA = "data/plantvillage/raw/color"
engines = torch.backends.quantized.supported_engines
torch.backends.quantized.engine = "fbgemm" if "fbgemm" in engines else "qnnpack"

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Même découpage train / validation que train.py (seed 42)
ds = datasets.ImageFolder(DATA, transform=tf)
random.seed(42)
by_class = defaultdict(list)
for i, (_, y) in enumerate(ds.samples):
    by_class[y].append(i)
val_idx = []
for idxs in by_class.values():
    random.shuffle(idxs)
    idxs = idxs[:200]
    val_idx += idxs[int(0.8 * len(idxs)):]
val_dl = DataLoader(Subset(ds, val_idx), batch_size=64)

classes = json.load(open("models/classes.json"))


def load_fp32():
    m = models.mobilenet_v3_small()
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(classes))
    m.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
    return m.eval()


def size_mb(m):
    buffer = io.BytesIO()
    torch.save(m.state_dict(), buffer)
    return buffer.tell() / 1e6


def accuracy(m, dtype):
    correct = 0
    with torch.no_grad():
        for x, y in val_dl:
            correct += (m(x.to(dtype)).argmax(1) == y).sum().item()
    return correct / len(val_idx)


def latency_ms(m, dtype, n=100):
    x = torch.randn(1, 3, 224, 224, dtype=dtype)
    with torch.no_grad():
        for _ in range(10):
            m(x)
        start = time.perf_counter()
        for _ in range(n):
            m(x)
    return (time.perf_counter() - start) / n * 1000


versions = {
    "FP32": (load_fp32(), torch.float32),
    "INT8 dynamique": (torch.ao.quantization.quantize_dynamic(load_fp32(), {nn.Linear}, dtype=torch.qint8), torch.float32),
    "FP16": (load_fp32().half(), torch.float16),
}

mlflow.set_experiment("plant-disease")
print(f"Validation : {len(val_idx)} images\n")
print(f"{'Version':<16}{'Taille (Mo)':>12}{'Latence (ms)':>14}{'Précision':>11}")

for name, (m, dtype) in versions.items():
    try:
        size, lat, acc = size_mb(m), latency_ms(m, dtype), accuracy(m, dtype)
    except Exception as e:
        print(f"{name:<16} impossible sur cette machine ({type(e).__name__})")
        continue
    print(f"{name:<16}{size:>12.2f}{lat:>14.1f}{acc * 100:>10.1f} %")
    with mlflow.start_run(run_name=f"quantization-{name}"):
        mlflow.log_param("version", name)
        mlflow.log_metrics({"size_mb": size, "latency_ms": lat, "val_accuracy": acc})