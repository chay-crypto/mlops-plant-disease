import os
import random
from pathlib import Path

from locust import HttpUser, between, task

API_KEY = os.getenv("API_KEY", "cle-chaima-123")
IMAGES = sorted(Path("data/plantvillage/raw/color").glob("*/*"))[::500]


class ApiUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def predict(self):
        image = random.choice(IMAGES)
        with open(image, "rb") as f:
            self.client.post(
                "/predict",
                headers={"X-API-Key": API_KEY},
                files={"file": f},
            )