# Medical Imaging Report — Industrial Pipeline

AI-assisted diagnostic reporting pipeline: **Ingestion → Preprocessing → Inference → Report → Clinician Review**.

This repository implements the end-to-end industrial plan from `Medical_Imaging_Report_Presentation.pptx`: a working local system with DICOM ingestion, clinical model adapters (LIDC/nnDetection, BraTS/nnU-Net), model registry + validation gate, FHIR export, persisted clinician review, PACS SCP scaffolding, and a regulatory/QMS documentation track.

**Not a medical device.** No CDSCO/FDA clearance. Every report carries a research disclaimer.

## Quick start

On Windows + OneDrive, prefer a venv **outside** the synced folder (native libs can hang when cloud-only):

```powershell
# recommended
.\scripts\run.ps1
```

Or manually:

```bash
python -m venv C:\Users\sadhu\mir-venv
C:\Users\sadhu\mir-venv\Scripts\activate
pip install -r backend\requirements.txt
set PYTHONPATH=%CD%
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

Upload PNG/JPG, `.dcm`, or a ZIP of DICOM instances. Choose **demo** or **clinical** mode.

## Architecture

```
backend/
  main.py                     FastAPI app (upload, FHIR, review, registry, PACS)
  contracts.py                Frozen findings/report API contracts
  pipeline/
    preprocess.py             resize, denoise, CLAHE
    detect.py                 heuristic LoG detector (demo / fallback)
    clinical_detect.py        LIDC/nnDetection + BraTS/nnU-Net adapters
    modality_router.py        demo vs clinical routing by modality
    dicom_ingest.py           pydicom ingestion + windowing
    model.py                  PneumoniaMNIST demo CNN
    report.py                 JSON report + overlay
    fhir_export.py            FHIR R4 DiagnosticReport bundle
  models/registry/            model registry, gate, audit log
  db/reviews.py               SQLite accept/reject + sign-off
  pacs/                       study queue + C-STORE SCP scaffolding
  training/                   PneumoniaMNIST trainer + LIDC/BraTS wrappers
  tests/                      pytest suite
docs/regulatory/              ISO 13485 / IEC 62304 / ISO 14971 / CDSCO-FDA drafts
frontend/                     upload, report, persisted review UI
```

## API surface

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/scans` | Upload PNG/JPG/DICOM/ZIP (`mode`, `window` form fields) |
| GET | `/api/scans/{id}` | Report JSON + reviews |
| GET | `/api/scans/{id}/fhir` | FHIR R4 Bundle |
| POST | `/api/scans/{id}/findings/{fid}/review` | Persist accept/reject |
| POST | `/api/scans/{id}/signoff` | Report sign-off |
| GET | `/api/feedback/export` | Retraining feedback export |
| GET/POST | `/api/registry*` | List / register / promote / rollback models |
| GET | `/api/pacs/queue` | Study queue |
| POST | `/api/pacs/scp/start` | Start DICOM C-STORE SCP |

## Clinical model training (GPU hosts)

```bash
# Writes checkpoint folder + metrics stub; pass --train when data + toolkit exist
python -m backend.training.train_nndetection_lidc
python -m backend.training.train_nnunet_brats

# Demo CNN (CPU, minutes)
python -m backend.training.train_classifier
```

Place approved weights at:

- `backend/models/checkpoints/lidc_nndetection/model.pt` + `metrics.json`
- `backend/models/checkpoints/brats_nnunet/model.pt` + `metrics.json`

Then register/promote through `/api/registry/promote` (validation gate enforced).

## PACS SCP

```bash
set MIR_PACS_SCP=1
set MIR_AE_TITLE=MIR_SCP
set MIR_PACS_PORT=11112
uvicorn main:app --app-dir ..
```

Or `POST /api/pacs/scp/start`. Lab/demo only — require TLS, AE allow-lists, and institutional agreements before PHI.

## Tests

```bash
cd backend
.venv\Scripts\python.exe -m pytest tests -q
```

## Regulatory track

See [`docs/regulatory/`](docs/regulatory/) for intended use, ISO 13485 QMS outline, IEC 62304 lifecycle, ISO 14971 risk starter, and CDSCO/FDA pathway checklist.
