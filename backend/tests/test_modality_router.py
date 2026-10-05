import numpy as np

from backend.pipeline.modality_router import run_inference


def test_demo_mode_returns_findings_contract_fields():
    img = np.zeros((512, 512), dtype=np.uint8)
    img[200:240, 200:240] = 220
    out = run_inference(img, img, modality="DX", mode="demo", scan_type="chest", filename="chest.png")
    assert out["mode"] == "demo"
    assert "findings" in out
    assert out["model_info"]["anatomy"] == "chest"
    assert out["model_info"]["mode"] == "demo"


def test_clinical_ct_fallback_without_checkpoint():
    img = np.zeros((512, 512), dtype=np.uint8)
    out = run_inference(img, img, modality="CT", mode="clinical", scan_type="ct", filename="chest_ct.png")
    assert out["modality"] == "CT"
    assert out["model_info"]["mode"] in {"clinical", "clinical-fallback"}
    assert out["model_info"]["anatomy"] == "chest"
