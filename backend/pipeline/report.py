"""Turns detected findings into a structured report + an annotated overlay image."""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import cv2
import numpy as np

from .detect import Finding

# Quadrant naming assumes a 512x512 canvas from preprocess.TARGET_SIZE
GRID = 512

DEFAULT_DISCLAIMER = (
    "Research prototype output. Not a medical device, not clinically validated, "
    "and not for diagnostic use. All findings require review and sign-off by a "
    "licensed radiologist before any clinical decision is made."
)


def _quadrant(x: int, y: int) -> str:
    horiz = "left" if x < GRID / 2 else "right"
    vert = "upper" if y < GRID / 2 else "lower"
    return f"{vert} {horiz}"


def _severity_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "high salience"
    if confidence >= 0.45:
        return "moderate salience"
    return "low salience"


def build_narrative(finding: Finding, *, anatomy: str | None = None) -> str:
    quadrant = _quadrant(finding.x, finding.y)
    severity = _severity_label(finding.confidence)
    tone = "denser than" if finding.polarity == "hyperintense" else "less dense than"
    organ = "brain parenchyma" if anatomy == "brain" else "surrounding tissue"
    disease_hint = (
        "Possible focal lesion candidate on brain MRI. "
        if anatomy == "brain"
        else ""
    )
    return (
        f"Finding #{finding.id}: a focal region in the {quadrant} field, "
        f"approximately {finding.radius_px * 2:.0f}px in diameter, {tone} the "
        f"{organ} ({severity}, score {finding.confidence:.2f}). "
        f"{disease_hint}"
        f"Flagged for radiologist review — not a confirmed diagnosis."
    )


def draw_overlay(original: np.ndarray, findings: list[Finding]) -> np.ndarray:
    color_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    for f in findings:
        color = (0, 165, 255) if f.confidence >= 0.75 else (0, 220, 220) if f.confidence >= 0.45 else (200, 200, 0)
        r = max(8, int(f.radius_px))
        cv2.circle(color_img, (f.x, f.y), r, color, 2)
        label = f"#{f.id} {f.confidence:.2f}"
        cv2.putText(
            color_img,
            label,
            (f.x - r, max(12, f.y - r - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return color_img


def build_report(
    scan_id: str,
    filename: str,
    findings: list[Finding],
    classification: dict | None = None,
    *,
    model_info: dict[str, Any] | None = None,
    modality: str | None = None,
    dicom_metadata: dict[str, Any] | None = None,
) -> dict:
    anatomy = None
    if isinstance(model_info, dict):
        anatomy = model_info.get("anatomy")
        if anatomy == "brain-likely":
            anatomy = "brain"

    clf = classification or {}
    findings_json = []
    for f in findings:
        findings_json.append(
            {
                "id": f.id,
                "location": {"x": f.x, "y": f.y, "quadrant": _quadrant(f.x, f.y)},
                "size_px_diameter": round(f.radius_px * 2, 1),
                "polarity": f.polarity,
                "confidence": f.confidence,
                "severity_label": _severity_label(f.confidence),
                "narrative": build_narrative(f, anatomy=anatomy),
            }
        )

    if clf.get("available") and clf.get("predicted_label"):
        disease = clf["predicted_label"]
        prob = clf.get("probability")
        prob_txt = f" (confidence {prob})" if prob is not None else ""
        if findings_json:
            summary = (
                f"Image-level assessment: {disease}{prob_txt}. "
                f"{len(findings_json)} candidate region(s) flagged for review "
                f"(highest salience {max(x['confidence'] for x in findings_json):.2f})."
            )
        else:
            summary = f"Image-level assessment: {disease}{prob_txt}. No focal regions flagged."
    elif findings_json:
        summary = (
            f"{len(findings_json)} candidate region(s) flagged for review. "
            f"Highest salience: {max(x['confidence'] for x in findings_json):.2f}."
        )
    else:
        summary = "No candidate regions of interest flagged by the detector."

    info = model_info or {
        "detector": "classical-cv-blob-log-v0",
        "type": "heuristic prototype — NOT a trained/validated clinical model",
    }

    report: dict[str, Any] = {
        "scan_id": scan_id,
        "filename": filename,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "model_info": info,
        "summary": summary,
        "findings": findings_json,
        "image_level_classification": classification or {"available": False},
        "disclaimer": DEFAULT_DISCLAIMER,
        "modality": modality or (dicom_metadata or {}).get("modality") or "OT",
        "review_status": "draft",
    }
    if dicom_metadata:
        # Strip bulky / PHI-heavy fields for default JSON; keep identifiers used by FHIR
        report["dicom_metadata"] = {
            k: v
            for k, v in dicom_metadata.items()
            if k
            in {
                "modality",
                "study_instance_uid",
                "series_instance_uid",
                "sop_instance_uid",
                "study_date",
                "study_description",
                "series_description",
                "rows",
                "columns",
                "slice_thickness",
                "window",
                "instance_count",
                "source",
                "patient_id",
                "patient_name",
                "manufacturer",
            }
        }
    return report


def new_scan_id() -> str:
    return uuid.uuid4().hex[:12]
