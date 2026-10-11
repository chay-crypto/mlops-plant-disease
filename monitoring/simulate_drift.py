import io
import random
import sys
from pathlib import Path

import requests
from PIL import Image, ImageEnhance

API = sys.argv[1]
KEY = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "normal"
N = 50

images = [p for p in Path("data/plantvillage/raw/color").glob("*/*")
          if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
random.seed(0 if MODE == "normal" else 1)

for path in random.sample(images, N):
    image = Image.open(path).convert("RGB")
    if MODE == "sombre":
        # Simule des photos prises en plein champ, mal éclairées
        image = ImageEnhance.Brightness(image).enhance(0.35)
        image = ImageEnhance.Contrast(image).enhance(0.6)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    requests.post(
        f"{API}/predict",
        headers={"X-API-Key": KEY},
        files={"file": ("image.jpg", buffer.getvalue(), "image/jpeg")},
        timeout=30,
    )

print(f"{N} images '{MODE}' envoyées à {API}")