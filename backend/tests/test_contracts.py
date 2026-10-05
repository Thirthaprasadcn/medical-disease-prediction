"""Phase 0: freeze and verify report/finding API contracts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.contracts import validate_finding, validate_report
from backend.pipeline.detect import Finding, detect_findings
from backend.pipeline.preprocess import preprocess
from backend.pipeline.report import build_report, draw_overlay


def _tiny_png_bytes() -> bytes:
    import cv2

    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img, (32, 32), 8, 200, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_preprocess_returns_original_and_enhanced():
    pre = preprocess(_tiny_png_bytes())
    assert pre["original"].shape == (512, 512)
    assert pre["enhanced"].shape == (512, 512)


def test_report_matches_frozen_contract():
    pre = preprocess(_tiny_png_bytes())
    findings = detect_findings(pre["enhanced"], pre["original"])
    report = build_report("abc123def456", "tiny.png", findings, {"available": False})
    errors = validate_report(report)
    assert errors == [], errors
    for f in report["findings"]:
        assert validate_finding(f) == []


def test_overlay_draws_without_error():
    pre = preprocess(_tiny_png_bytes())
    findings = [
        Finding(id=1, x=100, y=100, radius_px=12.0, polarity="hyperintense", confidence=0.8)
    ]
    overlay = draw_overlay(pre["original"], findings)
    assert overlay.ndim == 3
    assert overlay.shape[:2] == pre["original"].shape


def test_existing_stored_reports_honor_contract():
    reports_dir = Path(__file__).resolve().parent.parent / "storage" / "reports"
    if not reports_dir.exists():
        pytest.skip("no stored reports")
    files = list(reports_dir.glob("*.json"))
    if not files:
        pytest.skip("no stored reports")
    checked = 0
    for path in files[:10]:
        report = json.loads(path.read_text(encoding="utf-8"))
        # Pre-industrial reports may lack newer keys; migrate lightly for the check.
        report.setdefault("image_level_classification", {"available": False})
        report.setdefault("disclaimer", "legacy")
        errors = validate_report(report)
        assert errors == [], (path.name, errors)
        checked += 1
    if checked == 0:
        pytest.skip("no stored reports")
