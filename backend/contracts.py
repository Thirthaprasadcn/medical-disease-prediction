"""Frozen API contracts for findings, classification, and reports.

Model swaps (Phase 2+) must preserve these shapes so the frontend and
report builders do not break. Treat changes here as a versioned breaking change.
"""
from __future__ import annotations

from typing import Any, TypedDict


class LocationContract(TypedDict):
    x: int
    y: int
    quadrant: str


class FindingContract(TypedDict):
    id: int
    location: LocationContract
    size_px_diameter: float
    polarity: str
    confidence: float
    severity_label: str
    narrative: str


class ClassificationContract(TypedDict, total=False):
    available: bool
    predicted_label: str
    probability: float
    dataset_trained_on: str
    reported_test_metrics: dict[str, Any]
    notes: str


class ModelInfoContract(TypedDict):
    detector: str
    type: str


class ReportContract(TypedDict, total=False):
    scan_id: str
    filename: str
    generated_at: str
    model_info: ModelInfoContract
    summary: str
    findings: list[FindingContract]
    image_level_classification: ClassificationContract
    disclaimer: str
    modality: str
    dicom_metadata: dict[str, Any]
    review_status: str


REQUIRED_FINDING_KEYS = {
    "id",
    "location",
    "size_px_diameter",
    "polarity",
    "confidence",
    "severity_label",
    "narrative",
}
REQUIRED_REPORT_KEYS = {
    "scan_id",
    "filename",
    "generated_at",
    "model_info",
    "summary",
    "findings",
    "image_level_classification",
    "disclaimer",
}


def validate_finding(finding: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FINDING_KEYS - set(finding)
    if missing:
        errors.append(f"finding missing keys: {sorted(missing)}")
    loc = finding.get("location")
    if not isinstance(loc, dict) or not {"x", "y", "quadrant"} <= set(loc):
        errors.append("finding.location must include x, y, quadrant")
    return errors


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_REPORT_KEYS - set(report)
    if missing:
        errors.append(f"report missing keys: {sorted(missing)}")
    for f in report.get("findings") or []:
        errors.extend(validate_finding(f))
    clf = report.get("image_level_classification")
    if not isinstance(clf, dict) or "available" not in clf:
        errors.append("image_level_classification.available is required")
    return errors
