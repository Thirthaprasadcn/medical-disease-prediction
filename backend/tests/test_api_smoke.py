import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app


def test_upload_fhir_review_registry_smoke():
    client = TestClient(app)
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img, (32, 32), 10, 200, -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    files = {"file": ("smoke.png", io.BytesIO(buf.tobytes()), "image/png")}
    r = client.post("/api/scans", files=files, data={"mode": "demo"})
    assert r.status_code == 200, r.text
    report = r.json()
    sid = report["scan_id"]
    assert "findings" in report

    fhir = client.get(f"/api/scans/{sid}/fhir")
    assert fhir.status_code == 200
    assert fhir.json()["resourceType"] == "Bundle"

    if report["findings"]:
        rev = client.post(
            f"/api/scans/{sid}/findings/{report['findings'][0]['id']}/review",
            json={"decision": "accept", "reviewer": "qa"},
        )
        assert rev.status_code == 200
        assert rev.json()["decision"] == "accept"

    reg = client.get("/api/registry")
    assert reg.status_code == 200
    assert reg.json().get("active") is not None
