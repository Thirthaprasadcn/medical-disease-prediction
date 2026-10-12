"""Export 50 sample images for each of the four disease paths into datasets/.

1) Chest X-ray      → PneumoniaMNIST (MedMNIST)
2) Brain MRI tumor  → local Brain Tumor MRI dataset
3) Chest CT nodules → NoduleMNIST3D center slices (public proxy for LIDC-IDRI style CT)
4) Brain MRI seg    → brain MRI + pseudo-masks (BraTS-style region samples)

Full LIDC-IDRI (~100GB+, TCIA) and full BraTS (registration) are not bulk-mirrored here.
These folders give you ready local samples that match each project path.

Run:
  C:\\Users\\sadhu\\mir-venv\\Scripts\\python.exe -m scripts.export_sample_datasets
  or
  python scripts/export_sample_datasets.py
"""
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets"
BRAIN_SRC = ROOT / "backend" / "data" / "brain_tumor"
N_EACH = 50
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_readme(folder: Path, text: str) -> None:
    (folder / "README.txt").write_text(text.strip() + "\n", encoding="utf-8")


def _save_gray(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def export_pneumonia() -> int:
    from medmnist import PneumoniaMNIST

    folder = _ensure(OUT / "01_chest_xray_pneumoniamnist")
    images = _ensure(folder / "images")
    ds = PneumoniaMNIST(split="test", download=True, size=28)
    # imgs: (N, 28, 28) or (N, 1, 28, 28)
    imgs = np.array(ds.imgs)
    labels = np.array(ds.labels).reshape(-1)
    if imgs.ndim == 4:
        imgs = imgs[:, 0]
    label_names = {0: "normal", 1: "pneumonia"}

    # balanced ~25 each when possible
    idx0 = list(np.where(labels == 0)[0])
    idx1 = list(np.where(labels == 1)[0])
    random.shuffle(idx0)
    random.shuffle(idx1)
    half = N_EACH // 2
    chosen = idx0[:half] + idx1[: N_EACH - half]
    random.shuffle(chosen)

    rows = []
    for i, idx in enumerate(chosen[:N_EACH], start=1):
        lab = int(labels[idx])
        name = f"cxr_{i:02d}_{label_names[lab]}.png"
        # upscale to 224 for easier viewing while keeping content
        big = cv2.resize(imgs[idx], (224, 224), interpolation=cv2.INTER_NEAREST)
        _save_gray(images / name, big)
        rows.append({"file": f"images/{name}", "label": label_names[lab], "source": "PneumoniaMNIST", "index": int(idx)})

    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "source", "index"])
        w.writeheader()
        w.writerows(rows)

    _write_readme(
        folder,
        f"""Chest X-ray — PneumoniaMNIST
Samples: {len(rows)}
Labels: normal, pneumonia
Original MedMNIST size: 28x28 (saved here upscaled to 224x224 for viewing)
Used in project for: pneumonia vs normal classification (pneumonia_cnn)
""",
    )
    return len(rows)


def export_brain_tumor() -> int:
    folder = _ensure(OUT / "02_brain_mri_tumor")
    images = _ensure(folder / "images")
    if not BRAIN_SRC.exists():
        raise FileNotFoundError(
            f"Brain dataset missing at {BRAIN_SRC}. Run: python -m backend.training.train_brain_tumor"
        )

    class_map = {
        "no_tumor": "no_tumor",
        "notumor": "no_tumor",
        "glioma_tumor": "glioma_tumor",
        "glioma": "glioma_tumor",
        "meningioma_tumor": "meningioma_tumor",
        "meningioma": "meningioma_tumor",
        "pituitary_tumor": "pituitary_tumor",
        "pituitary": "pituitary_tumor",
    }
    by_class: dict[str, list[Path]] = {c: [] for c in ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]}
    for p in BRAIN_SRC.rglob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parent = p.parent.name.lower().replace(" ", "_")
        canon = class_map.get(parent)
        if canon and len(by_class[canon]) < 40:
            by_class[canon].append(p)

    per = max(1, N_EACH // 4)
    chosen: list[tuple[str, Path]] = []
    for cls, paths in by_class.items():
        random.shuffle(paths)
        for p in paths[:per]:
            chosen.append((cls, p))
    # fill to 50
    leftovers = []
    for cls, paths in by_class.items():
        for p in paths:
            if (cls, p) not in chosen:
                leftovers.append((cls, p))
    random.shuffle(leftovers)
    while len(chosen) < N_EACH and leftovers:
        chosen.append(leftovers.pop())
    chosen = chosen[:N_EACH]
    random.shuffle(chosen)

    rows = []
    for i, (cls, src) in enumerate(chosen, start=1):
        name = f"brain_{i:02d}_{cls}{src.suffix.lower()}"
        dst = images / name
        shutil.copy2(src, dst)
        rows.append({"file": f"images/{name}", "label": cls, "source": "Brain-Tumor-Classification-DataSet", "path": str(src.relative_to(BRAIN_SRC))})

    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "source", "path"])
        w.writeheader()
        w.writerows(rows)

    _write_readme(
        folder,
        f"""Brain MRI — Brain Tumor MRI dataset
Samples: {len(rows)}
Classes: no_tumor, glioma_tumor, meningioma_tumor, pituitary_tumor
Used in project for: 4-class tumor classification (brain_tumor_cnn)
""",
    )
    return len(rows)


def export_ct_nodules() -> int:
    """Public CT nodule slices (NoduleMNIST3D) as LIDC-style samples."""
    from medmnist import NoduleMNIST3D

    folder = _ensure(OUT / "03_chest_ct_lidc_style")
    images = _ensure(folder / "images")
    ds = NoduleMNIST3D(split="test", download=True, size=28)
    imgs = np.array(ds.imgs)
    labels = np.array(ds.labels).reshape(-1)
    if imgs.ndim == 5:
        imgs = imgs[:, 0]
    # imgs: (N, D, H, W)
    label_names = {0: "no_nodule", 1: "nodule"}
    idx0 = list(np.where(labels == 0)[0])
    idx1 = list(np.where(labels == 1)[0])
    random.shuffle(idx0)
    random.shuffle(idx1)
    half = N_EACH // 2
    chosen = idx0[:half] + idx1[: N_EACH - half]
    random.shuffle(chosen)

    rows = []
    for i, idx in enumerate(chosen[:N_EACH], start=1):
        vol = imgs[idx]
        if vol.ndim == 3:
            sl = vol[vol.shape[0] // 2]
        else:
            sl = vol
        lab = int(labels[idx])
        name = f"ct_{i:02d}_{label_names[lab]}.png"
        big = cv2.resize(sl, (224, 224), interpolation=cv2.INTER_NEAREST)
        _save_gray(images / name, big)
        rows.append(
            {
                "file": f"images/{name}",
                "label": label_names[lab],
                "source": "NoduleMNIST3D_center_slice_LIDC_style_proxy",
                "index": int(idx),
            }
        )

    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "source", "index"])
        w.writeheader()
        w.writerows(rows)

    _write_readme(
        folder,
        f"""Chest CT nodules — LIDC-style samples
Samples: {len(rows)}
Source used here: NoduleMNIST3D (MedMNIST) center slices — public proxy for LIDC-IDRI style CT nodules.
Full LIDC-IDRI is ~100GB+ from TCIA and is not copied into this repo.
Labels: no_nodule, nodule
Saved viewing size: 224x224 (from 28x28 center slice)
Used in project for: lung nodule detection / classification (ct_nodule_cnn)
""",
    )
    return len(rows)


def _pseudo_mask(gray: np.ndarray, size: int = 224) -> np.ndarray:
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    brain = g > 20
    out = np.zeros_like(g, dtype=np.uint8)
    if brain.sum() < 30:
        return out
    thr = float(np.percentile(g[brain], 88))
    mask = ((g >= thr) & brain).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            out[labels == i] = 255
    return out


def export_brats_style() -> int:
    folder = _ensure(OUT / "04_brain_mri_segmentation_brats_style")
    images = _ensure(folder / "images")
    masks = _ensure(folder / "masks")
    if not BRAIN_SRC.exists():
        raise FileNotFoundError(f"Brain dataset missing at {BRAIN_SRC}")

    # prefer tumor class images for segmentation demos
    tumor_paths = [
        p
        for p in BRAIN_SRC.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "no_tumor" not in str(p).lower()
        and "notumor" not in str(p).lower()
    ]
    if len(tumor_paths) < N_EACH:
        tumor_paths = [p for p in BRAIN_SRC.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(tumor_paths)
    chosen = tumor_paths[:N_EACH]

    rows = []
    for i, src in enumerate(chosen, start=1):
        raw = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        img = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_AREA)
        msk = _pseudo_mask(raw, 224)
        iname = f"brats_style_{i:02d}.png"
        mname = f"brats_style_{i:02d}_mask.png"
        _save_gray(images / iname, img)
        _save_gray(masks / mname, msk)
        rows.append(
            {
                "image": f"images/{iname}",
                "mask": f"masks/{mname}",
                "source": "brain_mri_plus_pseudo_mask_BraTS_style",
                "note": "weak intensity mask — not official BraTS multi-modal volume",
            }
        )

    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "mask", "source", "note"])
        w.writeheader()
        w.writerows(rows)

    _write_readme(
        folder,
        f"""Brain MRI segmentation — BraTS-style samples
Samples: {len(rows)} image+mask pairs
Official BraTS requires challenge registration and large multi-modal NIfTI volumes; not bulk-copied here.
These samples: brain MRI slices + pseudo tumor-region masks for local demos of the segmentation path.
Used in project for: tumor-region segmentation (MiniUNet / brats path)
""",
    )
    return len(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    print("Exporting PneumoniaMNIST (50)…")
    counts["pneumonia"] = export_pneumonia()
    print("Exporting Brain Tumor MRI (50)…")
    counts["brain_tumor"] = export_brain_tumor()
    print("Exporting CT nodules LIDC-style (50)…")
    counts["ct_nodules"] = export_ct_nodules()
    print("Exporting BraTS-style seg pairs (50)…")
    counts["brats_style"] = export_brats_style()

    summary = OUT / "README.md"
    summary.write_text(
        f"""# Local sample datasets (50 each)

| Folder | PPT dataset | Purpose | Count |
|--------|-------------|---------|-------|
| `01_chest_xray_pneumoniamnist` | PneumoniaMNIST | Pneumonia vs Normal | {counts['pneumonia']} |
| `02_brain_mri_tumor` | Brain Tumor MRI dataset | 4-class tumor classification | {counts['brain_tumor']} |
| `03_chest_ct_lidc_style` | LIDC-IDRI (proxy: NoduleMNIST3D) | Lung nodule detection | {counts['ct_nodules']} |
| `04_brain_mri_segmentation_brats_style` | BraTS (style samples + masks) | Tumor-region segmentation | {counts['brats_style']} |

**Note:** Full **LIDC-IDRI** and official **BraTS** archives are very large and need TCIA/challenge access.
This export gives you **50 ready samples per disease path** inside the project for demos, PPT evidence, and testing.

Each folder has `images/`, `labels.csv`, and `README.txt`.
""",
        encoding="utf-8",
    )
    print("Done:", counts)
    print(f"Location: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
