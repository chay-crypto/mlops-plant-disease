import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/plantvillage/raw/color")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-class", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Appareil :", device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data, transform=tf)

    # Sous-échantillon équilibré : N images max par classe, 80 % train / 20 % validation
    random.seed(42)
    by_class = defaultdict(list)
    for i, (_, y) in enumerate(ds.samples):
        by_class[y].append(i)
    train_idx, val_idx = [], []
    for idxs in by_class.values():
        random.shuffle(idxs)
        idxs = idxs[: args.per_class]
        cut = int(0.8 * len(idxs))
        train_idx += idxs[:cut]
        val_idx += idxs[cut:]

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size)
    print(f"{len(ds.classes)} classes, {len(train_idx)} images train, {len(val_idx)} images validation")

    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(ds.classes))
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_experiment("plant-disease")
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total += loss.item() * len(y)

            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()

            train_loss = total / len(train_idx)
            val_acc = correct / len(val_idx)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={train_loss:.3f}  val_acc={val_acc:.3f}")

        Path("models").mkdir(exist_ok=True)
        torch.save(model.to("cpu").state_dict(), "models/model.pt")
        Path("models/classes.json").write_text(json.dumps(ds.classes))
        mlflow.log_artifact("models/model.pt")
        mlflow.log_artifact("models/classes.json")
        print("Modèle enregistré dans models/")


if __name__ == "__main__":
    main()