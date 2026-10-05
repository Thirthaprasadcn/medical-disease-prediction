import numpy as np

from backend.pipeline.anatomy import infer_anatomy
from backend.pipeline.modality_router import run_inference


def _fake_brain() -> np.ndarray:
    img = np.zeros((512, 512), dtype=np.uint8)
    yy, xx = np.ogrid[:512, :512]
    mask = (yy - 256) ** 2 + (xx - 256) ** 2 <= 160**2
    img[mask] = 140
    img[200:260, 220:280] = 220
    return img


def test_brain_filename_routes_away_from_pneumonia():
    img = _fake_brain()
    info = infer_anatomy(img, filename="brain_tumor_mri.png")
    assert info["anatomy"] == "brain"

    out = run_inference(img, img, modality="OT", mode="clinical", filename="brain_tumor_mri.png")
    dataset = (out["classification"] or {}).get("dataset_trained_on", "")
    assert "pneumonia" not in dataset.lower()
    assert out["model_info"].get("anatomy") in {"brain", "brain-likely"}


def test_scan_type_brain_forces_brain_model():
    # flat mid-gray image that heuristics might call chest
    img = np.full((512, 512), 120, dtype=np.uint8)
    out = run_inference(
        img, img, modality="OT", mode="clinical", filename="image(1).jpg", scan_type="brain"
    )
    dataset = (out["classification"] or {}).get("dataset_trained_on", "")
    assert "pneumonia" not in dataset.lower()
    assert out["model_info"].get("anatomy") == "brain"
    assert out["modality"] in {"MR", "MRI"}


def test_chest_still_can_use_pneumonia_path():
    img = np.full((512, 512), 120, dtype=np.uint8)
    img[50:450, 40:470] = 150
    out = run_inference(
        img, img, modality="DX", mode="demo", filename="chest_xray.png", scan_type="chest"
    )
    assert out["model_info"].get("anatomy") == "chest"
