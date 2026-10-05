"""Route inference by anatomy + modality + trained disease models.

CRITICAL: never run PneumoniaMNIST (chest) on brain MRI uploads.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from backend.pipeline.anatomy import infer_anatomy
from backend.pipeline.brain_model import classify_brain, is_available as brain_available
from backend.pipeline.clinical_detect import (
    classify_ct_nodule,
    clinical_model_info,
    detect_brats_findings,
    detect_lidc_findings,
)
from backend.pipeline.detect import Finding, detect_findings
from backend.pipeline.model import classify as classify_pneumonia
from backend.pipeline.model import is_available as pneumonia_available
from backend.models.registry.registry import get_active_entry


def resolve_mode(requested: str | None = None) -> str:
    if requested in {"demo", "clinical"}:
        return requested
    active = get_active_entry()
    if active and active.get("status") == "approved" and active.get("mode") == "clinical":
        return "clinical"
    return "demo"


def _brain_path(
    enhanced: np.ndarray,
    original: np.ndarray,
    *,
    modality: str,
    anatomy_info: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    clinical = detect_brats_findings(enhanced, original)
    info = clinical_model_info("brats")
    if clinical is not None:
        findings = clinical
        detector = info["detector"]
        det_type = info["type"]
        det_mode = "clinical"
    else:
        findings = detect_findings(enhanced, original)
        detector = "classical-cv-blob-log-v0"
        det_type = "brain MRI path — region finder + trained tumor classifier"
        det_mode = "brain"

    brain_clf = classify_brain(original)
    if brain_clf.get("available"):
        classification = brain_clf
    else:
        classification = {
            "available": False,
            "predicted_label": "brain-mri-pending-model",
            "probability": None,
            "dataset_trained_on": "brain-tumor-mri",
            "notes": brain_clf.get(
                "notes",
                "Train the brain model: python -m backend.training.train_brain_tumor",
            ),
            "anatomy": "brain",
        }

    return {
        "findings": findings,
        "classification": classification,
        "model_info": {
            "detector": detector,
            "type": det_type,
            "mode": det_mode,
            "modality": modality,
            "anatomy": "brain",
            "anatomy_reason": anatomy_info.get("reason"),
            "brain_model_available": brain_available(),
            "clinical_available": info.get("available"),
            "disease_model": "brain-tumor-mri-4class",
        },
        "modality": modality if modality not in {None, "", "OT"} else "MR",
    }


def _chest_xray_path(
    enhanced: np.ndarray,
    original: np.ndarray,
    *,
    modality: str,
    anatomy_info: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    findings = detect_findings(enhanced, original)
    if pneumonia_available():
        classification = classify_pneumonia(original)
    else:
        classification = {
            "available": False,
            "predicted_label": "chest-xray-pending-model",
            "probability": None,
            "dataset_trained_on": "pneumoniamnist",
            "notes": "Pneumonia checkpoint missing. Run train_classifier.",
        }
    classification = {**classification, "anatomy": "chest"}
    return {
        "findings": findings,
        "classification": classification,
        "model_info": {
            "detector": "classical-cv-blob-log-v0",
            "type": "chest X-ray path — heuristic regions + PneumoniaMNIST classifier",
            "mode": mode,
            "modality": modality,
            "anatomy": "chest",
            "anatomy_reason": anatomy_info.get("reason"),
            "disease_model": "pneumoniamnist",
        },
        "modality": modality if modality not in {None, "", "OT"} else "DX",
    }


def _chest_ct_path(
    enhanced: np.ndarray,
    original: np.ndarray,
    *,
    modality: str,
    anatomy_info: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    clinical = detect_lidc_findings(enhanced, original)
    info = clinical_model_info("lidc")
    if clinical is not None:
        findings = clinical
        model_info = {
            "detector": info["detector"],
            "type": info["type"],
            "mode": "clinical",
            "modality": modality,
            "anatomy": "chest",
            "anatomy_reason": anatomy_info.get("reason"),
            "disease_model": "LIDC-IDRI",
        }
    else:
        findings = detect_findings(enhanced, original)
        model_info = {
            "detector": "classical-cv-blob-log-v0",
            "type": "chest CT path — heuristic regions (LIDC checkpoint not loaded)",
            "mode": "clinical-fallback",
            "modality": modality,
            "anatomy": "chest",
            "anatomy_reason": anatomy_info.get("reason"),
            "disease_model": "LIDC-IDRI",
        }
    ct_clf = classify_ct_nodule(original)
    if ct_clf.get("available"):
        classification = ct_clf
        model_info = {
            **model_info,
            "disease_model": ct_clf.get("dataset_trained_on") or "NoduleMNIST3D",
            "mode": "clinical" if clinical is not None else model_info.get("mode"),
        }
    else:
        classification = {
            "available": False,
            "predicted_label": "chest-ct-screening",
            "probability": None,
            "dataset_trained_on": "LIDC-IDRI",
            "notes": "Chest CT path. PneumoniaMNIST is not used for CT.",
            "anatomy": "chest",
        }
    return {
        "findings": findings,
        "classification": classification,
        "model_info": model_info,
        "modality": "CT",
    }


def _arbitrate_ambiguous(
    enhanced: np.ndarray,
    original: np.ndarray,
    *,
    anatomy_info: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """When heuristics disagree, prefer the disease model that fits the image."""
    scores = anatomy_info.get("scores") or {}
    brain_s = float(scores.get("brain") or 0)
    chest_s = float(scores.get("chest") or 0)

    brain_clf = classify_brain(original) if brain_available() else {"available": False}
    if brain_clf.get("available"):
        pred = str(brain_clf.get("predicted_class") or "")
        prob = float(brain_clf.get("probability") or 0)
        tumor = pred in {"glioma_tumor", "meningioma_tumor", "pituitary_tumor"}
        # Strong signal: tumor class on a not-strongly-chest image
        if tumor and prob >= 0.35 and chest_s < brain_s + 0.25:
            out = _brain_path(
                enhanced, original, modality="MR", anatomy_info=anatomy_info, mode=mode
            )
            out["model_info"]["anatomy_reason"] = (
                f"model arbitration → brain tumor CNN ({pred}, p={prob:.2f}); "
                f"{anatomy_info.get('reason')}"
            )
            out["model_info"]["anatomy"] = "brain"
            return out
        # Confident brain prediction (including no_tumor) when MRI-like
        if prob >= 0.5 and brain_s >= 0.4:
            out = _brain_path(
                enhanced, original, modality="MR", anatomy_info=anatomy_info, mode=mode
            )
            out["model_info"]["anatomy_reason"] = (
                f"model arbitration → brain CNN (p={prob:.2f}); "
                f"{anatomy_info.get('reason')}"
            )
            return out

    # Only use pneumonia when chest is clearly ahead
    if chest_s >= brain_s + 0.15 and chest_s >= 0.55 and pneumonia_available():
        out = _chest_xray_path(
            enhanced, original, modality="DX", anatomy_info=anatomy_info, mode=mode
        )
        out["model_info"]["anatomy_reason"] = (
            f"model arbitration → chest X-ray; {anatomy_info.get('reason')}"
        )
        return out

    # Safety default: withhold pneumonia rather than mislabel a brain study
    findings = detect_findings(enhanced, original)
    if brain_clf.get("available") and brain_s >= 0.35:
        out = _brain_path(
            enhanced, original, modality="MR", anatomy_info=anatomy_info, mode=mode
        )
        out["model_info"]["anatomy_reason"] = (
            "safety default → brain path (pneumonia withheld on ambiguous scan)"
        )
        return out

    return {
        "findings": findings,
        "classification": {
            "available": False,
            "predicted_label": "anatomy-unrecognized",
            "probability": None,
            "notes": (
                "Could not safely choose a disease model. "
                "Set Scan type to Brain MRI, Chest X-ray, or Chest CT on the Upload page."
            ),
            "anatomy": "unknown",
        },
        "model_info": {
            "detector": "classical-cv-blob-log-v0",
            "type": "unrecognized anatomy — disease classifier withheld for safety",
            "mode": mode,
            "modality": anatomy_info.get("modality") or "OT",
            "anatomy": "unknown",
            "anatomy_reason": anatomy_info.get("reason"),
            "disease_model": None,
        },
        "modality": anatomy_info.get("modality") or "OT",
    }


def run_inference(
    enhanced: np.ndarray,
    original: np.ndarray,
    *,
    modality: str | None = None,
    mode: str | None = None,
    filename: str | None = None,
    scan_type: str | None = None,
) -> dict[str, Any]:
    mode = resolve_mode(mode)
    anatomy_info = infer_anatomy(
        original,
        filename=filename,
        dicom_modality=modality,
        scan_type=scan_type,
    )
    anatomy = anatomy_info["anatomy"]
    modality = (anatomy_info.get("modality") or modality or "OT").upper()

    # -------- BRAIN PATH (MRI / head) --------
    if anatomy == "brain" or modality in {"MR", "MRI"}:
        # Guard: if user/DICOM said brain, always brain — never pneumonia
        packed = _brain_path(
            enhanced, original, modality=modality, anatomy_info=anatomy_info, mode=mode
        )

    # -------- CHEST CT --------
    elif anatomy == "chest" and modality == "CT":
        packed = _chest_ct_path(
            enhanced, original, modality=modality, anatomy_info=anatomy_info, mode=mode
        )

    # -------- CHEST X-RAY --------
    elif anatomy == "chest" and modality in {"DX", "CR"}:
        # Extra safety: if heuristics called it chest but brain score is close,
        # re-check with brain model before committing to PneumoniaMNIST
        scores = anatomy_info.get("scores") or {}
        brain_s = float(scores.get("brain") or 0)
        chest_s = float(scores.get("chest") or 0)
        user_forced = (scan_type or "").lower() in {"chest", "ct"}
        # filename/DICOM chest is trusted; pure image "chest" with close scores → re-check
        image_only_chest = "image appearance" in (anatomy_info.get("reason") or "")
        if not user_forced and image_only_chest and brain_s + 0.05 >= chest_s - 0.1:
            packed = _arbitrate_ambiguous(
                enhanced, original, anatomy_info=anatomy_info, mode=mode
            )
        else:
            packed = _chest_xray_path(
                enhanced, original, modality=modality, anatomy_info=anatomy_info, mode=mode
            )

    # -------- UNKNOWN / AMBIGUOUS --------
    else:
        packed = _arbitrate_ambiguous(
            enhanced, original, anatomy_info=anatomy_info, mode=mode
        )

    return {
        "findings": packed["findings"],
        "classification": packed["classification"],
        "model_info": packed["model_info"],
        "mode": mode,
        "modality": packed.get("modality") or modality,
        "anatomy": anatomy_info,
    }
