"""Train a 4-class brain MRI tumor classifier and write checkpoint + metrics.

Dataset: public brain MRI folders (glioma / meningioma / pituitary / no_tumor).
Downloads automatically when missing.

Run from project root:
  C:\\Users\\sadhu\\mir-venv\\Scripts\\python.exe -m backend.training.train_brain_tumor
"""
from __future__ import annotations

import datetime as dt
import io
import json
import random
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from backend.pipeline.classifier_arch import BrainTumorCNN

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "brain_tumor"
OUT_DIR = ROOT / "models" / "checkpoints" / "brain_tumor"
CHECKPOINT = OUT_DIR / "brain_tumor_cnn.pt"
METRICS = OUT_DIR / "metrics.json"

# Public brain MRI classification dataset (class folders under Training/Testing)
DATASET_URLS = [
    "https://github.com/sartajbhuvaji/Brain-Tumor-Classification-DataSet/archive/refs/heads/master.zip",
    "https://codeload.github.com/sartajbhuvaji/Brain-Tumor-Classification-DataSet/zip/refs/heads/master",
]

CLASSES = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
# map common folder name variants → canonical
CLASS_ALIASES = {
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
    "no tumor": "no_tumor",
    "glioma_tumor": "glioma_tumor",
    "glioma": "glioma_tumor",
    "meningioma_tumor": "meningioma_tumor",
    "meningioma": "meningioma_tumor",
    "pituitary_tumor": "pituitary_tumor",
    "pituitary": "pituitary_tumor",
}

INPUT_SIZE = 64
EPOCHS = 8
BATCH = 32
LR = 1e-3
SEED = 42
MAX_PER_CLASS = 800  # keep CPU training fast but real


def _download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    marker = DATA_DIR / ".ready"
    if marker.exists() and (
        any(DATA_DIR.rglob("*.jpg"))
        or any(DATA_DIR.rglob("*.jpeg"))
        or any(DATA_DIR.rglob("*.png"))
    ):
        return DATA_DIR

    last_err: Exception | None = None
    for url in DATASET_URLS:
        try:
            print(f"Downloading brain tumor MRI dataset…\n  {url}")
            req = Request(url, headers={"User-Agent": "MIR-brain-trainer/1.0"})
            with urlopen(req, timeout=300) as resp:
                blob = resp.read()
            if len(blob) < 1000 or blob[:2] != b"PK":
                raise RuntimeError(f"Not a zip download ({len(blob)} bytes, magic={blob[:20]!r})")
            zf = zipfile.ZipFile(io.BytesIO(blob))
            zf.extractall(DATA_DIR)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  failed: {exc}")
    else:
        raise RuntimeError(f"Could not download brain tumor dataset: {last_err}")

    for nested in list(DATA_DIR.rglob("*.zip")):
        try:
            with zipfile.ZipFile(nested) as nz:
                nz.extractall(DATA_DIR / nested.stem)
            nested.unlink(missing_ok=True)
        except zipfile.BadZipFile:
            continue

    marker.write_text("ok", encoding="utf-8")
    print(f"Dataset ready under {DATA_DIR}")
    return DATA_DIR


def _collect_samples(root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # walk parents for class folder name
        label = None
        for part in path.parts:
            key = part.lower().replace(" ", "_")
            if key in CLASS_ALIASES:
                label = CLASS_ALIASES[key]
                break
            # also match without _tumor suffix variants already in aliases
        if label is None:
            continue
        samples.append((path, CLASSES.index(label)))

    # balance / cap
    by_class: dict[int, list[tuple[Path, int]]] = {i: [] for i in range(len(CLASSES))}
    for s in samples:
        by_class[s[1]].append(s)
    rng = random.Random(SEED)
    out: list[tuple[Path, int]] = []
    for i, items in by_class.items():
        rng.shuffle(items)
        out.extend(items[:MAX_PER_CLASS])
        print(f"  class {CLASSES[i]}: using {min(len(items), MAX_PER_CLASS)} / {len(items)}")
    rng.shuffle(out)
    return out


class BrainFolderDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return x, label


def train() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    _download_dataset()
    samples = _collect_samples(DATA_DIR)
    if len(samples) < 40:
        raise RuntimeError(
            f"Too few labeled images found under {DATA_DIR} ({len(samples)}). "
            "Expected folders: no_tumor, glioma_tumor, meningioma_tumor, pituitary_tumor"
        )

    n = len(samples)
    n_test = max(20, int(0.15 * n))
    n_val = max(20, int(0.15 * n))
    test_items = samples[:n_test]
    val_items = samples[n_test:n_test + n_val]
    train_items = samples[n_test + n_val:]
    print(f"Split train/val/test = {len(train_items)}/{len(val_items)}/{len(test_items)}")

    train_loader = DataLoader(BrainFolderDataset(train_items), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(BrainFolderDataset(val_items), batch_size=BATCH)
    test_loader = DataLoader(BrainFolderDataset(test_items), batch_size=BATCH)

    device = torch.device("cpu")
    model = BrainTumorCNN(num_classes=len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        val_acc = _accuracy(model, val_loader, device)
        print(f"epoch {epoch}/{EPOCHS} loss={total/max(1,len(train_items)):.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    test_metrics = _eval(model, test_loader, device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)

    payload = {
        "dataset": "brain-tumor-mri-4class",
        "dataset_description": (
            "Public brain MRI classification set with glioma, meningioma, "
            "pituitary tumor, and no_tumor classes."
        ),
        "label_map": {str(i): c for i, c in enumerate(CLASSES)},
        "num_classes": len(CLASSES),
        "image_size": INPUT_SIZE,
        "epochs": EPOCHS,
        "trained_at": dt.datetime.utcnow().isoformat() + "Z",
        "val_accuracy_at_selection": round(best_val, 4),
        "test_metrics": test_metrics,
        "notes": (
            "Trained for brain MRI tumor typing. Not for chest X-ray. "
            "Research prototype — not clinically validated / not a medical device."
        ),
    }
    METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {CHECKPOINT}")
    print(f"Wrote {METRICS}")
    print("test_metrics:", test_metrics)
    return 0


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        pred = logits.argmax(1).cpu().numpy()
        preds.append(pred)
        ys.append(yb.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(preds)
    return float(accuracy_score(y, p))


@torch.no_grad()
def _eval(model: nn.Module, loader: DataLoader, device) -> dict:
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        pred = logits.argmax(1).cpu().numpy()
        preds.append(pred)
        ys.append(yb.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(preds)
    cm = confusion_matrix(y, p, labels=list(range(len(CLASSES)))).tolist()
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
        "n": int(len(y)),
        "confusion_matrix": cm,
        "classes": CLASSES,
    }


if __name__ == "__main__":
    raise SystemExit(train())
