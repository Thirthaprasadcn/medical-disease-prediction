from backend.pipeline.fhir_export import report_to_fhir_bundle


def test_fhir_bundle_structure():
    report = {
        "scan_id": "abc123",
        "filename": "x.png",
        "generated_at": "2026-01-01T00:00:00Z",
        "summary": "1 finding",
        "disclaimer": "prototype",
        "model_info": {"detector": "test", "type": "test"},
        "image_level_classification": {"available": False},
        "findings": [
            {
                "id": 1,
                "location": {"x": 10, "y": 10, "quadrant": "upper left"},
                "size_px_diameter": 12.0,
                "polarity": "hyperintense",
                "confidence": 0.8,
                "severity_label": "high salience",
                "narrative": "test finding",
            }
        ],
    }
    bundle = report_to_fhir_bundle(report)
    assert bundle["resourceType"] == "Bundle"
    types = {e["resource"]["resourceType"] for e in bundle["entry"]}
    assert "DiagnosticReport" in types
    assert "Observation" in types
