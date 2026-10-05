"""Inference for the trained brain-tumor CNN (Figshare / 4-class MRI)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .classifier_arch import BrainTumorCNN

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "brain_tumor"
CHECKPOINT_PATH = CHECKPOINT_DIR / "brain_tumor_cnn.pt"
METRICS_PATH = CHECKPOINT_DIR / "metrics.json"
INPUT_SIZE = 64

DEFAULT_LABELS = {
    "0": "no_tumor",
    "1": "glioma_tumor",
    "2": "meningioma_tumor",
    "3": "pituitary_tumor",
}


@lru_cache(maxsize=1)
def _load_model() -> BrainTumorCNN | None:
    if not CHECKPOINT_PATH.exists() or not METRICS_PATH.exists():
        return None
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8-sig"))
    n = int(metrics.get("num_classes", 4))
    model = BrainTumorCNN(num_classes=n)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@lru_cache(maxsize=1)
def _load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8-sig"))


def is_available() -> bool:
    return CHECKPOINT_PATH.exists() and METRICS_PATH.exists()


def classify_brain(original_gray: np.ndarray) -> dict:
    model = _load_model()
    metrics = _load_metrics()
    if model is None or metrics is None:
        return {
            "available": False,
            "notes": "Brain-tumor checkpoint missing. Run: python -m backend.training.train_brain_tumor",
        }

    small = cv2.resize(original_gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(small.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    label_map = metrics.get("label_map") or DEFAULT_LABELS
    idx = int(probs.argmax())
    pred = label_map.get(str(idx), DEFAULT_LABELS.get(str(idx), str(idx)))
    confidence = float(probs[idx])

    # human-readable disease phrasing
    display = {
        "no_tumor": "no tumor detected",
        "glioma_tumor": "glioma (brain tumor)",
        "meningioma_tumor": "meningioma (brain tumor)",
        "pituitary_tumor": "pituitary tumor",
    }.get(pred, pred.replace("_", " "))

    class_probs = {
        label_map.get(str(i), str(i)): round(float(probs[i]), 4)
        for i in range(len(probs))
    }

    return {
        "available": True,
        "predicted_label": display,
        "predicted_class": pred,
        "probability": round(confidence, 4),
        "class_probabilities": class_probs,
        "dataset_trained_on": metrics.get("dataset", "brain-tumor-mri"),
        "reported_test_metrics": metrics.get("test_metrics"),
        "notes": metrics.get(
            "notes",
            "Brain MRI multi-class model. Research prototype — not for diagnostic use.",
        ),
        "anatomy": "brain",
    }
