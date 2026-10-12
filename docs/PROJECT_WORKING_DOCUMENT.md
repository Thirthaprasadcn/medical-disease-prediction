# Medical Imaging Report — How the Project Works (Start to End)

**Project:** Medical Disease Prediction / Medical Imaging Report (MIR)  
**Repository:** https://github.com/Thirthaprasadcn/medical-disease-prediction  
**Author:** Thirthaprasad C N  

**Important:** This system is a **research / academic prototype**. It is **not** a medical device, not clinically validated, and must not be used for real patient diagnosis.

---

## 1. What this project does

MIR is an end-to-end AI imaging console that:

1. Accepts a medical image (PNG, JPG, DICOM, or DICOM ZIP)
2. Preprocesses it to a standard size
3. Detects anatomy / modality (Brain MRI, Chest X-ray, or Chest CT)
4. Runs the **correct trained disease model** for that path
5. Finds candidate regions on the image (overlay)
6. Builds a structured report
7. Lets a reviewer accept/reject findings and sign off
8. Optionally exports **FHIR R4** and JSON

User journey in the website:

**Home → Upload → Output (report)**

---

## 2. High-level architecture

```
Browser (frontend)
  Home / Upload / Report pages + 3D WebGL stage
           │
           ▼ HTTP (FastAPI)
backend/main.py
           │
           ├─ preprocess.py          → 512×512 grayscale
           ├─ anatomy.py             → brain / chest / unknown
           ├─ modality_router.py     → chooses disease path
           ├─ detect.py              → region findings (LoG blobs)
           ├─ clinical_detect.py     → CT nodule + brain seg adapters
           ├─ model.py               → pneumonia CNN
           ├─ brain_model.py         → brain tumor CNN
           ├─ report.py              → JSON report + overlay PNG
           ├─ fhir_export.py         → FHIR bundle
           ├─ db/reviews.py          → accept/reject / sign-off
           └─ pacs/                  → DICOM SCP + study queue (scaffold)
```

**Tech stack**

| Layer | Technology |
|--------|------------|
| Backend API | FastAPI + Uvicorn |
| Images | OpenCV, NumPy, scikit-image |
| Deep learning | PyTorch |
| Frontend | Jinja HTML + CSS + JS + Three.js |
| Data / review | JSON reports + SQLite reviews |
| Standards | DICOM (pydicom), FHIR R4 |

---

## 3. Image sizes (pixels) — important

### 3.1 Pipeline working canvas (detection & report)

| Stage | Size (pixels) | Channels | Notes |
|--------|----------------|----------|--------|
| Uploaded file | Any (e.g. 256–3000+) | Color or gray | Decoded to grayscale |
| After preprocess | **512 × 512** | 1 (uint8 gray) | Aspect ratio kept; padded on black canvas |
| Enhanced image | **512 × 512** | 1 | Denoise + CLAHE for detection |
| Overlay / report grid | **512 × 512** | RGB overlay | Findings drawn on original resized image |

Defined in `backend/pipeline/preprocess.py` as `TARGET_SIZE = 512`.

**How resize works**

1. Decode upload → grayscale  
2. Scale so the **longer side = 512**  
3. Place on a **512×512** black canvas (letterbox / pad)  
4. Create `enhanced` = bilateral denoise + CLAHE  

So every study is normalized to **512×512 pixels** before routing and region detection.

### 3.2 Model input sizes (after second resize from 512×512)

| Model | Input size (pixels) | From working canvas |
|--------|----------------------|---------------------|
| Pneumonia CNN | **28 × 28** | 512→28 |
| Brain tumor CNN | **64 × 64** | 512→64 |
| CT nodule CNN | **64 × 64** | 512→64 |
| Brain Mini-UNet (seg) | **64 × 64** | 512→64 |

All model tensors are normalized to **0–1 float** (`pixel / 255`).

---

## 4. Models used (detailed)

### 4.1 Brain tumor MRI classifier

| Item | Detail |
|------|--------|
| ID | `brain_tumor_cnn` |
| File | `backend/models/checkpoints/brain_tumor/brain_tumor_cnn.pt` |
| Architecture | `BrainTumorCNN` (3 conv blocks + FC) |
| Input | 1 × **64 × 64** |
| Classes (4) | `no_tumor`, `glioma_tumor`, `meningioma_tumor`, `pituitary_tumor` |
| Dataset | Public brain MRI tumor classification set |
| Approx. test accuracy | **~82.8%** |
| Used when | Scan type = Brain, or anatomy = brain / MR |

**Disease output examples:** “glioma (brain tumor)”, “meningioma (brain tumor)”, “pituitary tumor”, “no tumor detected”.

### 4.2 Chest X-ray pneumonia classifier

| Item | Detail |
|------|--------|
| ID | `pneumonia_cnn` |
| File | `backend/models/checkpoints/pneumonia_cnn.pt` |
| Architecture | `SmallCNN` (binary logit) |
| Input | 1 × **28 × 28** |
| Classes (2) | `normal`, `pneumonia` |
| Dataset | **PneumoniaMNIST** (MedMNIST; pediatric CXR benchmark) |
| Approx. test accuracy | **~86.7%** (AUC ~0.96) |
| Used when | Scan type = Chest X-ray, or anatomy = chest + DX/CR |

**Critical rule:** This model must **not** run on brain MRI. The router enforces that.

### 4.3 Chest CT nodule classifier

| Item | Detail |
|------|--------|
| ID | `lidc_ct` / `ct-nodule-cnn-lidc-proxy` |
| File | `backend/models/checkpoints/lidc_nndetection/model.pt` |
| Architecture | `CTNoduleCNN` |
| Input | 1 × **64 × 64** |
| Classes (2) | `no_nodule`, `nodule` |
| Dataset | **NoduleMNIST3D** center slices (proxy for LIDC-style CT) |
| Approx. test accuracy | **~86.5%** (AUC ~0.93) |
| Used when | Scan type = Chest CT, or modality = CT on chest path |

Also produces **findings** (candidate nodule locations) via peak / heatmap logic on the 512×512 enhanced image.

### 4.4 Brain MRI region segmentation

| Item | Detail |
|------|--------|
| ID | `brats_mri` / `mini-unet-brats-proxy` |
| File | `backend/models/checkpoints/brats_nnunet/model.pt` |
| Architecture | `MiniUNet` |
| Input / output | 1 × **64 × 64** → mask **64 × 64** (upscaled to 512 for findings) |
| Task | Tumor / abnormal region segmentation |
| Training | Brain MRI images + weak intensity pseudo-masks |
| Approx. Dice | **~77%** |
| Used when | Brain path clinical region adapter is available |

### 4.5 Classical region detector (all paths as fallback / companion)

| Item | Detail |
|------|--------|
| Module | `backend/pipeline/detect.py` |
| Method | Laplacian of Gaussian (LoG) blob detection |
| Works on | **512 × 512** enhanced image |
| Max findings | 8 |
| Output | Circles with confidence, polarity, location quadrant |

---

## 5. Start-to-end workflow (step by step)

### Step 1 — Start the server

```powershell
.\scripts\run.ps1
# or
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open: http://127.0.0.1:8000  

Pages:

1. **Home** — product intro + disease model roster  
2. **Upload** — file + scan type + options  
3. **Output** — report after analysis  

### Step 2 — User uploads a study (Upload page)

User provides:

| Field | Meaning |
|--------|---------|
| File | PNG / JPG / `.dcm` / DICOM ZIP (max 100 MB) |
| **Scan type** | `auto` / `brain` / `chest` / `ct` |
| Mode | `clinical` or `demo` |
| CT window | Auto / lung / soft / bone (DICOM CT) |
| Priority | routine / urgent / STAT |
| Reviewer | name for audit |
| Clinical note | optional indication |

Frontend posts to `POST /api/scans` with multipart form data (`scan_type`, `mode`, etc.).

### Step 3 — Ingest

**A) PNG / JPG**

- Bytes → OpenCV decode → grayscale  

**B) DICOM / ZIP**

- `dicom_ingest.py` reads with **pydicom**  
- Applies CT windowing if needed  
- Converts a 2D slice to PNG bytes  
- Metadata (modality, study UID, etc.) stored on the report  

### Step 4 — Preprocess (always → 512×512)

```
raw grayscale
   → resize_keep_aspect(target=512)
   → original_512
   → denoise + CLAHE
   → enhanced_512
```

Both arrays are **512×512 uint8**.

### Step 5 — Anatomy / modality inference

`anatomy.py` + user `scan_type`:

**Priority order**

1. User scan type (Brain / Chest / CT) — strongest  
2. DICOM modality (`MR`, `CT`, `DX`, `CR`)  
3. Filename keywords (`brain`, `tumor`, `pneumonia`, `chest`, …)  
4. Image heuristics (dark circular MRI-like vs chest-like fill)  
5. Safety arbitration (prefer brain over pneumonia when ambiguous)

Outputs something like:

```json
{
  "anatomy": "brain",
  "modality": "MR",
  "confidence": 0.9,
  "reason": "user selected Brain MRI / tumor",
  "scores": {"brain": 0.61, "chest": 0.48}
}
```

### Step 6 — Modality router chooses the disease path

`modality_router.py`:

| Path | Disease model | Region / seg |
|------|----------------|--------------|
| Brain / MR | Brain tumor CNN (**64×64**) | BraTS Mini-UNet findings if ready, else LoG |
| Chest X-ray / DX | Pneumonia CNN (**28×28**) | LoG findings |
| Chest CT | CT nodule CNN (**64×64**) | LIDC adapter findings + LoG fallback |

**Safety:** PneumoniaMNIST is **never** applied on unrecognized/brain-like images by default.

### Step 7 — Inference details

For each path:

1. Resize 512×512 → model input size  
2. Forward pass (PyTorch)  
3. Softmax / sigmoid → label + probability  
4. Collect class probabilities (brain 4-class)  
5. Build **findings** list (x, y, radius, confidence) in **512×512** coordinates  

### Step 8 — Report generation

`report.py` builds a JSON report including:

- `scan_id`, filename, timestamps  
- `modality`, model info, anatomy reason  
- `summary` / impression text  
- `image_level_classification` (disease assessment)  
- `findings[]` with narratives  
- disclaimer (research only)  
- optional DICOM metadata + workflow (priority, reviewer, scan_type)  

Also writes:

- Original PNG (`512×512`)  
- Overlay PNG (circles on image)  

Stored under `backend/storage/` (local runtime; not committed to Git).

### Step 9 — Output page (clinician view)

User is redirected to `/report/{scan_id}`:

- Side-by-side **Original** and **AI overlay**  
- Impression + **Disease assessment**  
- Modality, anatomy, disease model, routing reason  
- Findings list with **Accept / Reject**  
- **Sign off**, FHIR export, JSON export  

Reviews persist in SQLite via `db/reviews.py`.

### Step 10 — Optional exports & integrations

| Feature | Endpoint / module |
|---------|-------------------|
| Report JSON | `GET /api/scans/{id}` |
| FHIR R4 Bundle | `GET /api/scans/{id}/fhir` |
| Feedback export | `GET /api/feedback/export` |
| Model registry | `/api/registry*` |
| PACS SCP | `POST /api/pacs/scp/start` (lab/demo) |

---

## 6. End-to-end example (Brain MRI PNG)

1. User selects **Scan type = Brain MRI / tumor**  
2. Uploads `tumor.png` (any pixel size, e.g. 512×512 or larger)  
3. Server resizes to **512×512**  
4. Router forces brain path (`MR`)  
5. Brain tumor CNN sees **64×64** crop/resize → e.g. `meningioma_tumor` @ 0.63  
6. Segmentation / LoG adds finding circles on **512×512**  
7. Output page shows disease assessment from **brain-tumor-mri-4class**, not PneumoniaMNIST  

---

## 7. End-to-end example (Chest X-ray)

1. Scan type = **Chest X-ray / pneumonia**  
2. Preprocess → **512×512**  
3. Pneumonia CNN → **28×28** → `pneumonia` or `normal`  
4. Findings from LoG on 512 canvas  
5. Disease model shown: **pneumoniamnist**  

---

## 8. End-to-end example (Chest CT)

1. Scan type = **Chest CT / nodules** (or DICOM `CT`)  
2. Preprocess → **512×512** (windowing applied if DICOM)  
3. CT nodule CNN → **64×64** → nodule / no nodule  
4. Findings from clinical CT adapter  
5. Pneumonia model is **not** used  

---

## 9. Pixel size summary (quick reference)

```
Upload (H×W, any)
        │
        ▼
   512 × 512   ← pipeline canvas (detect, overlay, report grid)
        │
        ├──────────────► 28 × 28   Pneumonia CNN
        ├──────────────► 64 × 64   Brain tumor CNN
        ├──────────────► 64 × 64   CT nodule CNN
        └──────────────► 64 × 64   Brain Mini-UNet mask
                              │
                              ▼
                     mask scaled back to 512 × 512 for findings
```

---

## 10. How to retrain models

From project root (with venv + `PYTHONPATH` set):

```bash
python -m backend.training.train_classifier      # pneumonia 28×28
python -m backend.training.train_brain_tumor     # brain 64×64, 4-class
python -m backend.training.train_ct_nodule       # CT nodule 64×64
python -m backend.training.train_brain_seg       # Mini-UNet 64×64
```

Checkpoints land in `backend/models/checkpoints/…` with `metrics.json`.

---

## 11. Limitations (honest)

- Not cleared by CDSCO / FDA; research use only  
- Public benchmark / proxy datasets ≠ hospital distribution  
- Brain tumor subtype can be wrong even when path is correct (~83% accuracy)  
- CT / BraTS paths are **lightweight proxies**, not full nnDetection / nnU-Net hospital training  
- 2D slice workflow — not full 3D volumetric clinical inference  

---

## 12. Folder map (for readers of the code)

| Path | Role |
|------|------|
| `backend/main.py` | API + page routes |
| `backend/pipeline/preprocess.py` | **512×512** normalize |
| `backend/pipeline/anatomy.py` | Anatomy scoring |
| `backend/pipeline/modality_router.py` | Path selection |
| `backend/pipeline/model.py` | Pneumonia **28×28** |
| `backend/pipeline/brain_model.py` | Brain tumor **64×64** |
| `backend/pipeline/clinical_detect.py` | CT + brain seg |
| `backend/pipeline/detect.py` | LoG findings on 512 |
| `backend/pipeline/report.py` | Report + overlay |
| `frontend/templates/` | Home / Upload / Report |
| `docs/regulatory/` | QMS / risk / pathway notes |

---

## 13. One-sentence summary

**Any uploaded scan is normalized to 512×512 pixels, routed to the correct disease AI (28×28 pneumonia, 64×64 brain/CT models), analyzed for disease and regions, then shown as a reviewable report with optional FHIR export.**
