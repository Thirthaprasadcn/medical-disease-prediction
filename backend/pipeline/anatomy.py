"""Infer anatomy / modality from filename + image appearance + optional user hint.

PNG uploads have no DICOM modality, so without careful routing brain MRIs were
incorrectly sent to the PneumoniaMNIST chest-X-ray classifier.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

BRAIN_KEYWORDS = (
    "brain", "mri", "tumor", "tumour", "glioma", "meningioma", "pituitary",
    "brats", "flair", "t1ce", "t1", "t2", "axial", "cerebr", "neuro", "head",
)
CHEST_KEYWORDS = (
    "chest", "xray", "x-ray", "pneumonia", "lung", "thorax", "cxr", "pa_",
    "radiograph",
)
CT_KEYWORDS = ("ct", "lidc", "nodule", "lung-ct", "chest-ct", "computed")


def _keyword_hit(name: str, words: tuple[str, ...]) -> bool:
    n = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    return any(w.replace("-", "").replace("_", "") in n for w in words)


def score_brain_mri(gray: np.ndarray) -> float:
    """Heuristic 0..1 score that the image looks like a brain MRI slice."""
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 0.0

    # MRI brains usually sit on near-black backgrounds (corners especially)
    dark_frac = float((gray < 22).mean())
    corner = np.concatenate(
        [
            gray[: h // 8, : w // 8].ravel(),
            gray[: h // 8, -w // 8 :].ravel(),
            gray[-h // 8 :, : w // 8].ravel(),
            gray[-h // 8 :, -w // 8 :].ravel(),
        ]
    )
    corner_dark = float((corner < 22).mean())

    ys, xs = np.where(gray > 35)
    if len(xs) < 50:
        return float(min(1.0, 0.25 * dark_frac + 0.35 * corner_dark))

    cy, cx = float(ys.mean()), float(xs.mean())
    center_dist = np.hypot(cy - h / 2, cx - w / 2) / (0.5 * max(h, w))
    compactness = 1.0 - min(1.0, center_dist)

    mask = (gray > 35).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    fill_ratio = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        peri = float(cv2.arcLength(c, True))
        if peri > 0:
            circularity = float(min(1.0, 4 * np.pi * area / (peri * peri)))
        fill_ratio = area / float(h * w)

    # brain MRI tissue usually occupies a moderate central fraction, not full frame
    fill_score = 1.0 - abs(fill_ratio - 0.35) / 0.35
    fill_score = float(min(1.0, max(0.0, fill_score)))

    score = (
        0.22 * dark_frac
        + 0.28 * corner_dark
        + 0.22 * compactness
        + 0.18 * circularity
        + 0.10 * fill_score
    )
    return float(min(1.0, max(0.0, score)))


def score_chest_xray(gray: np.ndarray) -> float:
    """Heuristic 0..1 score for chest radiograph-like images."""
    h, w = gray.shape[:2]
    dark_frac = float((gray < 18).mean())
    mid_frac = float(((gray > 40) & (gray < 200)).mean())
    # chest films usually fill more of the frame; corners are less pure-black than MRI
    corner = np.concatenate(
        [
            gray[: h // 8, : w // 8].ravel(),
            gray[: h // 8, -w // 8 :].ravel(),
            gray[-h // 8 :, : w // 8].ravel(),
            gray[-h // 8 :, -w // 8 :].ravel(),
        ]
    )
    corner_bright = float((corner > 30).mean())
    return float(
        min(
            1.0,
            max(
                0.0,
                0.40 * mid_frac + 0.25 * (1.0 - dark_frac) + 0.35 * corner_bright,
            ),
        )
    )


def infer_anatomy(
    gray: np.ndarray,
    filename: str | None = None,
    dicom_modality: str | None = None,
    scan_type: str | None = None,
) -> dict[str, Any]:
    """Return anatomy routing decision for the uploaded study.

    Priority: explicit scan_type > DICOM modality > filename keywords > image heuristics.
    """
    name = filename or ""
    dcm = (dicom_modality or "").upper().strip()
    st = (scan_type or "auto").lower().strip()

    brain_s = score_brain_mri(gray)
    chest_s = score_chest_xray(gray)
    scores = {"brain": round(brain_s, 4), "chest": round(chest_s, 4)}

    # 1) User override (Upload page "Scan type")
    if st == "brain":
        return {
            "anatomy": "brain",
            "modality": "MR",
            "confidence": 0.99,
            "reason": "user selected Brain MRI / tumor",
            "scores": scores,
            "scan_type": st,
        }
    if st == "chest":
        return {
            "anatomy": "chest",
            "modality": "DX",
            "confidence": 0.99,
            "reason": "user selected Chest X-ray",
            "scores": scores,
            "scan_type": st,
        }
    if st == "ct":
        return {
            "anatomy": "chest",
            "modality": "CT",
            "confidence": 0.99,
            "reason": "user selected Chest CT",
            "scores": scores,
            "scan_type": st,
        }

    brain_kw = _keyword_hit(name, BRAIN_KEYWORDS)
    chest_kw = _keyword_hit(name, CHEST_KEYWORDS)
    ct_kw = _keyword_hit(name, CT_KEYWORDS)

    # 2) DICOM modality
    if dcm in {"MR", "MRI"}:
        return {
            "anatomy": "brain",
            "modality": "MR",
            "confidence": 0.95,
            "reason": "DICOM modality MR",
            "scores": scores,
            "scan_type": st,
        }
    if dcm == "CT":
        if brain_kw or brain_s >= chest_s:
            return {
                "anatomy": "brain",
                "modality": "CT",
                "confidence": 0.8,
                "reason": "DICOM CT + brain appearance/filename",
                "scores": scores,
                "scan_type": st,
            }
        return {
            "anatomy": "chest",
            "modality": "CT",
            "confidence": 0.75,
            "reason": "DICOM modality CT",
            "scores": scores,
            "scan_type": st,
        }
    if dcm in {"DX", "CR"}:
        return {
            "anatomy": "chest",
            "modality": dcm,
            "confidence": 0.9,
            "reason": f"DICOM modality {dcm}",
            "scores": scores,
            "scan_type": st,
        }

    # 3) Filename keywords
    if brain_kw and not chest_kw:
        return {
            "anatomy": "brain",
            "modality": "MR",
            "confidence": 0.92,
            "reason": "filename indicates brain/MRI/tumor",
            "scores": scores,
            "scan_type": st,
        }
    if chest_kw and not brain_kw:
        return {
            "anatomy": "chest",
            "modality": "DX",
            "confidence": 0.88,
            "reason": "filename indicates chest/X-ray",
            "scores": scores,
            "scan_type": st,
        }
    if ct_kw and not brain_kw:
        return {
            "anatomy": "chest",
            "modality": "CT",
            "confidence": 0.75,
            "reason": "filename indicates CT",
            "scores": scores,
            "scan_type": st,
        }

    # 4) Image heuristics — require a clear margin before choosing chest
    #    (prevents brain MRI slices with mid-gray tissue from becoming "chest")
    if brain_s >= 0.48 and brain_s + 0.02 >= chest_s:
        return {
            "anatomy": "brain",
            "modality": "MR",
            "confidence": round(brain_s, 3),
            "reason": "image appearance matches brain MRI",
            "scores": scores,
            "scan_type": st,
        }
    if chest_s >= brain_s + 0.18 and chest_s >= 0.62:
        return {
            "anatomy": "chest",
            "modality": "DX",
            "confidence": round(chest_s, 3),
            "reason": "image appearance matches chest radiograph",
            "scores": scores,
            "scan_type": st,
        }

    # Ambiguous: lean brain when MRI-like corners/darkness present
    if brain_s >= 0.42:
        return {
            "anatomy": "brain",
            "modality": "MR",
            "confidence": round(brain_s, 3),
            "reason": "ambiguous upload; leaning brain MRI (safer than pneumonia path)",
            "scores": scores,
            "scan_type": st,
        }

    return {
        "anatomy": "unknown",
        "modality": dcm or "OT",
        "confidence": 0.3,
        "reason": "could not confidently infer anatomy",
        "scores": scores,
        "scan_type": st,
    }
