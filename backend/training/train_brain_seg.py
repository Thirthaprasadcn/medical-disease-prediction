"""Train a MiniU-Net for brain tumor regions using MRI images + pseudo-masks.

Uses the already-downloaded brain-tumor classification set. Pseudo-masks are
built from bright abnormal tissue inside the brain silhouette (weak labels).

Writes checkpoint expected by clinical_detect (brats_nnunet/).

Run:
  C:\\Users\\sadhu\\mir-venv\\Scripts\\python.exe -m backend.training.train_brain_seg
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.pipeline.classifier_arch import MiniUNet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "brain_tumor"
OUT = ROOT / "models" / "checkpoints" / "brats_nnunet"
WEIGHTS = OUT / "model.pt"
METRICS = OUT / "metrics.json"
INPUT_SIZE = 64
EPOCHS = 8
BATCH = 32
LR = 1e-3
SEED = 42
MAX_IMAGES = 2400


def _pseudo_mask(gray: np.ndarray) -> np.ndarray:
    """Weak tumor mask: bright blobs inside brain tissue (not background)."""
    g = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    brain = g > 20
    if brain.sum() < 30:
        return np.zeros_like(g, dtype=np.float32)
    vals = g[brain]
    thr = float(np.percentile(vals, 88))
    mask = (g >= thr) & brain
    # drop tiny speckles
    mask_u8 = mask.astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(g, dtype=np.float32)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 12:
            out[labels == i] = 1.0
    return out


def _collect_paths() -> list[Path]:
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(DATA.rglob(ext))
    # prefer tumor folders
    tumorish = [p for p in paths if "no_tumor" not in str(p).lower() and "notumor" not in str(p).lower()]
    use = tumorish if len(tumorish) > 50 else paths
    random.shuffle(use)
    return use[:MAX_IMAGES]


class SegDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        raw = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raw = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        img = cv2.resize(raw, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        mask = _pseudo_mask(raw)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy(mask).unsqueeze(0)
        return x, y


def _dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    p = (torch.sigmoid(pred) > 0.5).float()
    t = (target > 0.5).float()
    inter = (p * t).sum()
    return float((2 * inter + eps) / (p.sum() + t.sum() + eps))


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    if not DATA.exists() or not any(DATA.rglob("*.jpg")):
        print("Brain MRI dataset missing — run train_brain_tumor first.")
        return 2

    paths = _collect_paths()
    if len(paths) < 40:
        print(f"Not enough images under {DATA}")
        return 2
    split = int(0.85 * len(paths))
    train_paths, test_paths = paths[:split], paths[split:]
    print(f"train={len(train_paths)} test={len(test_paths)}")

    train_loader = DataLoader(SegDataset(train_paths), batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(SegDataset(test_paths), batch_size=BATCH)

    model = MiniUNet()
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
    dices = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            for i in range(len(xb)):
                dices.append(_dice(logits[i : i + 1], yb[i : i + 1]))
    mean_dice = float(np.mean(dices)) if dices else 0.0

    payload = {
        "arch": "mini_unet_brats_proxy",
        "input_size": INPUT_SIZE,
        "state_dict": model.state_dict(),
        "demo_heatmap": False,
    }
    torch.save(payload, WEIGHTS)

    metrics = {
        "detector_id": "mini-unet-brats-proxy",
        "dataset": "brain-tumor-mri + weak pseudo-masks (BraTS-style region head)",
        "framework": "pytorch-MiniUNet",
        "trained_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": "trained",
        "test_metrics": {
            "dice": round(mean_dice, 4),
            "sensitivity": None,
            "specificity": None,
            "n": len(test_paths),
        },
        "notes": (
            "Trained on public brain MRI images with weak intensity pseudo-masks "
            "(not official BraTS multi-modal volumes / nnU-Net). Research prototype."
        ),
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {WEIGHTS}")
    print(f"Wrote {METRICS} dice={mean_dice:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
