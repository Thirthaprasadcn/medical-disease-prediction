"""Catalog of disease models shown in the UI (not just the registry active entry)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.pipeline.brain_model import CHECKPOINT_PATH as BRAIN_CKPT
from backend.pipeline.brain_model import METRICS_PATH as BRAIN_METRICS
from backend.pipeline.brain_model import is_available as brain_available
from backend.pipeline.clinical_detect import brats_available, clinical_model_info, lidc_available
from backend.pipeline.model import CHECKPOINT_PATH as PNEU_CKPT
from backend.pipeline.model import METRICS_PATH as PNEU_METRICS
from backend.pipeline.model import is_available as pneumonia_available


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def list_disease_models() -> list[dict[str, Any]]:
    """Return all disease/modality models the product routes across."""
    brain_m = _read_json(BRAIN_METRICS) or {}
    pneu_m = _read_json(PNEU_METRICS) or {}
    brats = clinical_model_info("brats")
    lidc = clinical_model_info("lidc")
    lidc_m = _read_json(
        Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "lidc_nndetection" / "metrics.json"
    ) or {}
    brats_m = _read_json(
        Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "brats_nnunet" / "metrics.json"
    ) or {}

    lidc_ready = lidc_available() and lidc_m.get("status") == "trained"
    brats_ready = brats_available() and brats_m.get("status") == "trained"

    return [
        {
            "id": "brain_tumor_cnn",
            "name": "Brain tumor MRI",
            "scan_type": "brain",
            "modality": "MR",
            "diseases": ["glioma", "meningioma", "pituitary", "no tumor"],
            "dataset": brain_m.get("dataset", "brain-tumor-mri-4class"),
            "status": "ready" if brain_available() else "missing",
            "accuracy": (brain_m.get("test_metrics") or {}).get("accuracy"),
            "checkpoint": BRAIN_CKPT.exists(),
        },
        {
            "id": "pneumonia_cnn",
            "name": "Chest X-ray pneumonia",
            "scan_type": "chest",
            "modality": "DX",
            "diseases": ["pneumonia", "normal"],
            "dataset": pneu_m.get("dataset", "pneumoniamnist"),
            "status": "ready" if pneumonia_available() else "missing",
            "accuracy": (pneu_m.get("test_metrics") or {}).get("accuracy"),
            "checkpoint": PNEU_CKPT.exists(),
        },
        {
            "id": "lidc_ct",
            "name": "Chest CT nodules",
            "scan_type": "ct",
            "modality": "CT",
            "diseases": ["nodule", "no nodule"],
            "dataset": lidc_m.get("dataset", "NoduleMNIST3D"),
            "status": "ready" if lidc_ready else ("scaffold" if not lidc_available() else "ready"),
            "accuracy": (lidc_m.get("test_metrics") or {}).get("accuracy")
            or (lidc.get("metrics") or {}).get("accuracy"),
            "checkpoint": lidc_available(),
            "notes": lidc_m.get("notes"),
        },
        {
            "id": "brats_mri",
            "name": "Brain MRI segmentation",
            "scan_type": "brain",
            "modality": "MR",
            "diseases": ["tumor region segmentation"],
            "dataset": brats_m.get("dataset", "BraTS-proxy"),
            "status": "ready" if brats_ready else ("scaffold" if not brats_available() else "ready"),
            "accuracy": (brats_m.get("test_metrics") or {}).get("dice"),
            "metric_label": "dice",
            "checkpoint": brats_available(),
            "notes": brats_m.get("notes"),
        },
    ]
