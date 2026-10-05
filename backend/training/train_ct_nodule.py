"""Train a CT nodule classifier from MedMNIST NoduleMNIST3D (center slices).

Writes checkpoint expected by clinical_detect (lidc_nndetection/).

Run:
  C:\\Users\\sadhu\\mir-venv\\Scripts\\python.exe -m backend.training.train_ct_nodule
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from backend.pipeline.classifier_arch import CTNoduleCNN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "checkpoints" / "lidc_nndetection"
WEIGHTS = OUT / "model.pt"
METRICS = OUT / "metrics.json"
INPUT_SIZE = 64
EPOCHS = 6
BATCH = 64
LR = 1e-3
SEED = 42


class NoduleSliceDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.images[idx]
        if x.ndim == 3:
            x = x[x.shape[0] // 2]  # center axial slice
        # resize 28→64 via torch
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0) / 255.0
        t = torch.nn.functional.interpolate(t, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        return t.squeeze(0), torch.tensor([self.labels[idx]], dtype=torch.float32)


def _load_nodule() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from medmnist import NoduleMNIST3D

    train = NoduleMNIST3D(split="train", download=True, size=28)
    test = NoduleMNIST3D(split="test", download=True, size=28)
    # medmnist returns (N,1,D,H,W) or (N,D,H,W)
    x_train = np.array(train.imgs)
    y_train = np.array(train.labels).reshape(-1)
    x_test = np.array(test.imgs)
    y_test = np.array(test.labels).reshape(-1)
    if x_train.ndim == 5:
        x_train = x_train[:, 0]
        x_test = x_test[:, 0]
    return x_train, y_train, x_test, y_test


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Downloading / loading NoduleMNIST3D…")
    x_train, y_train, x_test, y_test = _load_nodule()
    print(f"train={len(y_train)} test={len(y_test)} pos_rate={y_train.mean():.3f}")

    train_loader = DataLoader(NoduleSliceDataset(x_train, y_train), batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(NoduleSliceDataset(x_test, y_test), batch_size=BATCH)

    model = CTNoduleCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        print(f"epoch {epoch}/{EPOCHS} loss={total / max(1, n):.4f}")

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = torch.sigmoid(model(xb)).cpu().numpy().reshape(-1)
            probs.extend(p.tolist())
            labels.extend(yb.cpu().numpy().reshape(-1).tolist())
    probs_a = np.array(probs)
    labels_a = np.array(labels)
    pred = (probs_a >= 0.5).astype(int)
    acc = float(accuracy_score(labels_a, pred))
    try:
        auc = float(roc_auc_score(labels_a, probs_a))
    except ValueError:
        auc = None
    cm = confusion_matrix(labels_a, pred).tolist()
    tn, fp, fn, tp = confusion_matrix(labels_a, pred).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 0.0

    payload = {
        "arch": "ct_nodule_cnn",
        "input_size": INPUT_SIZE,
        "state_dict": model.state_dict(),
        "demo_heatmap": True,  # also enable peak findings on CT uploads
    }
    torch.save(payload, WEIGHTS)

    metrics = {
        "detector_id": "ct-nodule-cnn-lidc-proxy",
        "dataset": "NoduleMNIST3D (MedMNIST) center-slice proxy for LIDC-style CT nodules",
        "framework": "pytorch-CTNoduleCNN",
        "trained_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": "trained",
        "test_metrics": {
            "accuracy": round(acc, 4),
            "auc": round(auc, 4) if auc is not None else None,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "n": int(len(labels_a)),
            "confusion_matrix": cm,
        },
        "notes": (
            "Trained on public NoduleMNIST3D 2D center slices (not full LIDC-IDRI / nnDetection). "
            "Research prototype — not a medical device."
        ),
        "label_map": {"0": "no_nodule", "1": "nodule"},
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {WEIGHTS}")
    print(f"Wrote {METRICS} acc={acc:.3f} auc={auc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
