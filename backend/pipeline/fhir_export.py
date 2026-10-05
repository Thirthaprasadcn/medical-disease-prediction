"""Phase 4: map internal report JSON → FHIR R4 DiagnosticReport + Observations."""
from __future__ import annotations

import datetime as dt
from typing import Any
from uuid import uuid4


def _ref(resource_type: str, resource_id: str) -> dict[str, str]:
    return {"reference": f"{resource_type}/{resource_id}"}


def finding_to_observation(scan_id: str, finding: dict[str, Any]) -> dict[str, Any]:
    obs_id = f"{scan_id}-finding-{finding['id']}"
    return {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "preliminary",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "365853007",
                    "display": "Imaging finding",
                }
            ],
            "text": f"Finding #{finding['id']}",
        },
        "valueString": finding.get("narrative"),
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": "A",
                        "display": "Abnormal",
                    }
                ],
                "text": finding.get("severity_label"),
            }
        ],
        "component": [
            {
                "code": {"text": "location_quadrant"},
                "valueString": finding.get("location", {}).get("quadrant"),
            },
            {
                "code": {"text": "size_px_diameter"},
                "valueQuantity": {
                    "value": finding.get("size_px_diameter"),
                    "unit": "px",
                    "system": "http://unitsofmeasure.org",
                    "code": "[px]",
                },
            },
            {
                "code": {"text": "confidence"},
                "valueQuantity": {
                    "value": finding.get("confidence"),
                    "unit": "probability",
                },
            },
            {
                "code": {"text": "polarity"},
                "valueString": finding.get("polarity"),
            },
        ],
        "note": [{"text": "Research prototype observation — not for clinical diagnosis."}],
    }


def report_to_fhir_bundle(report: dict[str, Any]) -> dict[str, Any]:
    scan_id = report["scan_id"]
    report_id = f"dr-{scan_id}"
    observations = [finding_to_observation(scan_id, f) for f in report.get("findings") or []]

    dicom = report.get("dicom_metadata") or {}
    subject = {
        "display": dicom.get("patient_name") or "anonymous-demo-patient",
    }
    if dicom.get("patient_id"):
        subject["identifier"] = {"value": dicom["patient_id"]}

    diagnostic_report = {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "meta": {
            "profile": [
                "http://hl7.org/fhir/StructureDefinition/DiagnosticReport"
            ]
        },
        "status": "preliminary",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "18748-4",
                    "display": "Diagnostic imaging study",
                }
            ],
            "text": "AI-assisted imaging report (prototype)",
        },
        "subject": subject,
        "effectiveDateTime": report.get("generated_at") or (dt.datetime.utcnow().isoformat() + "Z"),
        "issued": report.get("generated_at") or (dt.datetime.utcnow().isoformat() + "Z"),
        "performer": [{"display": "Medical Imaging Report Prototype"}],
        "result": [_ref("Observation", obs["id"]) for obs in observations],
        "conclusion": report.get("summary"),
        "presentedForm": [
            {
                "contentType": "application/json",
                "title": f"internal-report-{scan_id}.json",
            }
        ],
        "extension": [
            {
                "url": "https://medical-imaging-report.local/fhir/StructureDefinition/model-info",
                "valueString": json_dumps_safe(report.get("model_info")),
            },
            {
                "url": "https://medical-imaging-report.local/fhir/StructureDefinition/disclaimer",
                "valueString": report.get("disclaimer"),
            },
        ],
    }

    clf = report.get("image_level_classification") or {}
    if clf.get("available"):
        clf_obs = {
            "resourceType": "Observation",
            "id": f"{scan_id}-classification",
            "status": "preliminary",
            "code": {"text": "Image-level classification"},
            "valueString": str(clf.get("predicted_label")),
            "component": [
                {
                    "code": {"text": "probability"},
                    "valueString": str(clf.get("probability")),
                },
                {
                    "code": {"text": "dataset_trained_on"},
                    "valueString": str(clf.get("dataset_trained_on")),
                },
            ],
            "note": [{"text": str(clf.get("notes") or "")}],
        }
        observations.append(clf_obs)
        diagnostic_report["result"].append(_ref("Observation", clf_obs["id"]))

    entries = [
        {"fullUrl": f"urn:uuid:{uuid4()}", "resource": diagnostic_report},
        *[{"fullUrl": f"urn:uuid:{uuid4()}", "resource": obs} for obs in observations],
    ]
    return {
        "resourceType": "Bundle",
        "id": f"bundle-{scan_id}",
        "type": "collection",
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "entry": entries,
    }


def json_dumps_safe(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)
