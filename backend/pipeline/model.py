"""Inference wrapper for the trained CNN checkpoint produced by
backend/training/train_classifier.py.

This is a genuinely trained, genuinely evaluated model (see metrics.json for
real test-set AUC/sensitivity/specificity) — a step up from the classical-CV
heuristic in detect.py, but trained on a public 2D chest X-ray benchmark
(PneumoniaMNIST), not on CT/MRI or hospital data. It answers "does this image
look more like the pneumonia class or the normal class in that benchmark's
distribution" — treat its output on an arbitrary uploaded scan as illustrative
of a real trained-model integration, not as a validated clinical signal.
"""
from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import torch

from .classifier_arch import SmallCNN

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "pneumonia_cnn.pt"
METRICS_PATH = CHECKPOINT_DIR / "metrics.json"
INPUT_SIZE = 28


@lru_cache(maxsize=1)
def _load_model() -> SmallCNN | None:
    if not CHECKPOINT_PATH.exists():
        return None
    model = SmallCNN()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


@lru_cache(maxsize=1)
def _load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


def is_available() -> bool:
    return CHECKPOINT_PATH.exists() and METRICS_PATH.exists()


def classify(original_gray: np.ndarray) -> dict:
    """original_gray: preprocessed (resized/padded) grayscale image, e.g. 512x512 uint8."""
    model = _load_model()
    metrics = _load_metrics()
    if model is None or metrics is None:
        return {"available": False}

    small = cv2.resize(original_gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(small.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor).squeeze().item()
        probability = float(torch.sigmoid(torch.tensor(logit)))

    label_map = metrics["label_map"]
    predicted_class = "1" if probability >= 0.5 else "0"

    return {
        "available": True,
        "predicted_label": label_map[predicted_class],
        "probability": round(probability, 4),
        "dataset_trained_on": metrics["dataset"],
        "reported_test_metrics": metrics["test_metrics"],
        "notes": metrics["notes"],
    }
