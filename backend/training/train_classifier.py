"""Real supervised training run on a public benchmark dataset — PneumoniaMNIST
(chest X-ray, normal vs. pneumonia; https://medmnist.com), chosen because it's
small enough (~4MB) to download and train on CPU in minutes, unlike LIDC-IDRI
(~125GB) or BraTS (~20GB) which the plan's Section 7 targets for a real
CT/MRI-specific model but need GPU compute and hours-to-days of training.

This gives the app one genuinely trained, genuinely evaluated deep learning
model instead of only the classical-CV heuristic in pipeline/detect.py — with
real accuracy/AUC/sensitivity/specificity reported below, not asserted.

Run from the project root:
    backend/.venv/Scripts/python.exe -m backend.training.train_classifier
"""
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import medmnist
from medmnist import INFO

from backend.pipeline.classifier_arch import SmallCNN

DATASET_KEY = "pneumoniamnist"
IMAGE_SIZE = 28
EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3
SEED = 42

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "pneumonia_cnn.pt"
METRICS_PATH = CHECKPOINT_DIR / "metrics.json"


def to_tensor(pil_img) -> torch.Tensor:
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def build_loaders():
    info = INFO[DATASET_KEY]
    DataClass = getattr(medmnist, info["python_class"])

    train_ds = DataClass(split="train", download=True, size=IMAGE_SIZE, transform=to_tensor)
    val_ds = DataClass(split="val", download=True, size=IMAGE_SIZE, transform=to_tensor)
    test_ds = DataClass(split="test", download=True, size=IMAGE_SIZE, transform=to_tensor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, info


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.numpy())
        all_labels.append(labels.squeeze(1).numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs)
    accuracy = float((preds == labels).mean())
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "auc": round(float(auc), 4),
        "accuracy": round(accuracy, 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "n": int(len(labels)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    torch.manual_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, info = build_loaders()

    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            labels = labels.float().squeeze(1)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        val_metrics = evaluate(model, val_loader)
        print(f"epoch {epoch}/{EPOCHS}  train_loss={running_loss / len(train_loader.dataset):.4f}  "
              f"val_auc={val_metrics['auc']:.4f}  val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    print("test metrics:", test_metrics)

    torch.save(model.state_dict(), CHECKPOINT_PATH)

    metrics_record = {
        "dataset": DATASET_KEY,
        "dataset_description": info["description"],
        "label_map": info["label"],
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "trained_at": dt.datetime.utcnow().isoformat() + "Z",
        "val_auc_at_selection": round(best_val_auc, 4),
        "test_metrics": test_metrics,
        "notes": (
            "Trained on a public 2D chest X-ray benchmark for demonstration of a real "
            "training/evaluation pipeline. Not trained on CT/MRI, not hospital data, "
            "not clinically validated. See README for what a real CT/MRI-specific "
            "model would require."
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics_record, indent=2))
    print(f"saved checkpoint to {CHECKPOINT_PATH}")
    print(f"saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
