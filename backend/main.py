from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from backend.contracts import validate_report
from backend.db.reviews import (
    export_feedback,
    export_feedback_jsonl,
    get_report_reviews,
    init_db,
    set_signoff,
    upsert_finding_review,
)
from backend.models.registry.registry import (
    ensure_demo_entry,
    get_active_entry,
    list_entries,
    promote_entry,
    register_entry,
    rollback_active,
)
from backend.pacs.scp_server import scp_status, start_scp_background
from backend.pacs.study_queue import enqueue_study, list_studies, mark_processed
from backend.pipeline.dicom_ingest import (
    DicomIngestError,
    ingest_dicom_bytes,
    ingest_dicom_zip,
    is_dicom_bytes,
    save_dicom_bytes,
    slice_to_png_bytes,
)
from backend.pipeline.fhir_export import report_to_fhir_bundle
from backend.pipeline.modality_router import run_inference
from backend.pipeline.disease_models import list_disease_models
from backend.pipeline.preprocess import preprocess
from backend.pipeline.report import build_report, draw_overlay, new_scan_id

BASE_DIR = Path(__file__).resolve().parent
STORAGE = BASE_DIR / "storage"
UPLOADS = STORAGE / "uploads"
OVERLAYS = STORAGE / "overlays"
REPORTS = STORAGE / "reports"
DICOM_DIR = STORAGE / "dicom"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

for d in (UPLOADS, OVERLAYS, REPORTS, DICOM_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_DICOM_TYPES = {
    "application/dicom",
    "application/octet-stream",
    "application/zip",
    "application/x-zip-compressed",
}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    ensure_demo_entry()
    if os.environ.get("MIR_PACS_SCP") == "1":
        start_scp_background()
    yield


app = FastAPI(title="Medical Imaging Report — Industrial Pipeline", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))


def _persist_report(
    *,
    scan_id: str,
    filename: str,
    original,
    overlay,
    findings,
    classification,
    model_info,
    modality: str | None,
    dicom_metadata: dict[str, Any] | None,
    workflow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cv2.imwrite(str(UPLOADS / f"{scan_id}.png"), original)
    cv2.imwrite(str(OVERLAYS / f"{scan_id}.png"), overlay)
    report = build_report(
        scan_id,
        filename,
        findings,
        classification,
        model_info=model_info,
        modality=modality,
        dicom_metadata=dicom_metadata,
    )
    if workflow:
        report["workflow"] = workflow
    errors = validate_report(report)
    if errors:
        raise HTTPException(500, f"Report contract violation: {errors}")
    (REPORTS / f"{scan_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _analyze_image_bytes(
    image_bytes: bytes,
    filename: str,
    *,
    modality: str | None = None,
    mode: str | None = None,
    dicom_metadata: dict[str, Any] | None = None,
    workflow: dict[str, Any] | None = None,
    scan_type: str | None = None,
) -> dict[str, Any]:
    try:
        pre = preprocess(image_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    inferred = run_inference(
        pre["enhanced"],
        pre["original"],
        modality=modality or (dicom_metadata or {}).get("modality"),
        mode=mode,
        filename=filename,
        scan_type=scan_type or (workflow or {}).get("scan_type") or "auto",
    )
    overlay = draw_overlay(pre["original"], inferred["findings"])
    scan_id = new_scan_id()
    if workflow is None:
        workflow = {}
    workflow = {
        **workflow,
        "anatomy": inferred.get("anatomy"),
    }
    return _persist_report(
        scan_id=scan_id,
        filename=filename,
        original=pre["original"],
        overlay=overlay,
        findings=inferred["findings"],
        classification=inferred["classification"],
        model_info=inferred["model_info"],
        modality=inferred["modality"],
        dicom_metadata=dicom_metadata,
        workflow=workflow,
    )


def _recent_reports(limit: int = 12) -> list[dict]:
    scans = sorted(REPORTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    recent = []
    for p in scans[:limit]:
        data = json.loads(p.read_text(encoding="utf-8"))
        recent.append(
            {
                "scan_id": data["scan_id"],
                "filename": data["filename"],
                "generated_at": data["generated_at"],
                "modality": data.get("modality"),
                "summary": data.get("summary"),
                "findings_count": len(data.get("findings") or []),
                "review_status": data.get("review_status", "draft"),
                "mode": (data.get("model_info") or {}).get("mode"),
            }
        )
    return recent


@app.get("/")
def home(request: Request):
    """Page 1 — cinematic industrial landing."""
    recent = _recent_reports(6)
    active = get_active_entry()
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "recent": recent,
            "active_model": active,
            "disease_models": list_disease_models(),
            "scp": scp_status(),
            "report_count": len(list(REPORTS.glob("*.json"))),
            "page": "home",
        },
    )


@app.get("/upload")
def upload_page(request: Request):
    """Page 2 — study ingest console."""
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "active_model": get_active_entry(),
            "disease_models": list_disease_models(),
            "scp": scp_status(),
            "queue": list_studies(12),
            "page": "upload",
        },
    )


@app.post("/api/scans")
async def upload_scan(
    file: UploadFile = File(...),
    mode: str | None = Form(default=None),
    scan_type: str | None = Form(default="auto"),
    window: str = Form(default="default"),
    priority: str | None = Form(default="routine"),
    reviewer: str | None = Form(default=None),
    clinical_note: str | None = Form(default=None),
):
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File too large (max 100MB).")
    if not image_bytes:
        raise HTTPException(400, "Empty file.")

    filename = file.filename or "unknown"
    content_type = (file.content_type or "").lower()
    lower_name = filename.lower()
    workflow = {
        "priority": (priority or "routine").lower(),
        "reviewer": (reviewer or "").strip() or "anonymous",
        "clinical_note": (clinical_note or "").strip() or None,
        "window": window,
        "requested_mode": mode,
        "scan_type": (scan_type or "auto").lower(),
    }

    # Optional user override for anatomy routing (fixes brain→pneumonia mistakes)
    scan_type_norm = (scan_type or "auto").lower()
    modality_override = None
    filename_hint = filename
    if scan_type_norm == "brain":
        modality_override = "MR"
        filename_hint = f"brain_mri_{filename}"
    elif scan_type_norm == "chest":
        modality_override = "DX"
        filename_hint = f"chest_xray_{filename}"
    elif scan_type_norm == "ct":
        modality_override = "CT"
        filename_hint = f"chest_ct_{filename}"

    # DICOM path (single instance or zip)
    is_zip = lower_name.endswith(".zip") or content_type in {
        "application/zip",
        "application/x-zip-compressed",
    }
    is_dicom = (
        lower_name.endswith((".dcm", ".dicom", ".ima"))
        or content_type == "application/dicom"
        or is_dicom_bytes(image_bytes)
    )

    if is_zip or is_dicom:
        try:
            if is_zip:
                result = ingest_dicom_zip(image_bytes, window_name=window)
            else:
                result = ingest_dicom_bytes(image_bytes, window_name=window)
        except DicomIngestError as exc:
            raise HTTPException(400, str(exc)) from exc

        png_bytes = slice_to_png_bytes(result["slice"])
        report = _analyze_image_bytes(
            png_bytes,
            filename_hint,
            modality=modality_override or result["metadata"].get("modality"),
            mode=mode,
            dicom_metadata=result["metadata"],
            workflow=workflow,
            scan_type=scan_type_norm,
        )
        if is_zip:
            (DICOM_DIR / f"{report['scan_id']}.zip").write_bytes(image_bytes)
        else:
            save_dicom_bytes(DICOM_DIR / f"{report['scan_id']}.dcm", image_bytes)

        q = enqueue_study(
            source="upload_dicom",
            filename=filename,
            modality=result["metadata"].get("modality"),
            study_uid=result["metadata"].get("study_instance_uid"),
            patient_id=result["metadata"].get("patient_id"),
            dicom_path=str(DICOM_DIR / f"{report['scan_id']}.dcm"),
        )
        mark_processed(q["queue_id"], report["scan_id"])
        return JSONResponse(report)

    if content_type not in ALLOWED_IMAGE_TYPES and not lower_name.endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(
            400,
            f"Unsupported file type '{content_type}'. Upload PNG/JPG or DICOM (.dcm/.zip).",
        )

    report = _analyze_image_bytes(
        image_bytes,
        filename_hint,
        mode=mode,
        modality=modality_override or "OT",
        workflow=workflow,
        scan_type=scan_type_norm,
    )
    return JSONResponse(report)


@app.get("/api/scans/{scan_id}")
def get_report(scan_id: str):
    path = REPORTS / f"{scan_id}.json"
    if not path.exists():
        raise HTTPException(404, "Scan not found")
    report = json.loads(path.read_text(encoding="utf-8"))
    reviews = get_report_reviews(scan_id)
    report["reviews"] = reviews
    report["review_status"] = reviews.get("signoff", {}).get("status", "draft")
    return JSONResponse(report)


@app.get("/api/scans/{scan_id}/fhir")
def get_fhir(scan_id: str):
    path = REPORTS / f"{scan_id}.json"
    if not path.exists():
        raise HTTPException(404, "Scan not found")
    report = json.loads(path.read_text(encoding="utf-8"))
    return JSONResponse(report_to_fhir_bundle(report))


@app.get("/api/scans/{scan_id}/overlay")
def get_overlay(scan_id: str):
    path = OVERLAYS / f"{scan_id}.png"
    if not path.exists():
        raise HTTPException(404, "Overlay not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/scans/{scan_id}/original")
def get_original(scan_id: str):
    path = UPLOADS / f"{scan_id}.png"
    if not path.exists():
        raise HTTPException(404, "Original not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/scans/{scan_id}/findings/{finding_id}/review")
async def review_finding(scan_id: str, finding_id: int, request: Request):
    path = REPORTS / f"{scan_id}.json"
    if not path.exists():
        raise HTTPException(404, "Scan not found")
    body = await request.json()
    decision = body.get("decision")
    try:
        saved = upsert_finding_review(
            scan_id,
            finding_id,
            decision,
            reviewer=body.get("reviewer") or "anonymous",
            comment=body.get("comment"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(saved)


@app.post("/api/scans/{scan_id}/signoff")
async def signoff_report(scan_id: str, request: Request):
    path = REPORTS / f"{scan_id}.json"
    if not path.exists():
        raise HTTPException(404, "Scan not found")
    body = await request.json()
    try:
        saved = set_signoff(
            scan_id,
            body.get("status", "signed"),
            reviewer=body.get("reviewer") or "anonymous",
            note=body.get("note"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    report = json.loads(path.read_text(encoding="utf-8"))
    report["review_status"] = saved["status"]
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return JSONResponse(saved)


@app.get("/api/feedback/export")
def feedback_export():
    rows = export_feedback()
    path = export_feedback_jsonl()
    return JSONResponse({"count": len(rows), "path": str(path), "rows": rows})


@app.get("/api/registry")
def registry_list():
    return JSONResponse({"active": get_active_entry(), "entries": list_entries()})


@app.post("/api/registry/register")
async def registry_register(request: Request):
    body = await request.json()
    try:
        entry = register_entry(body, approved_by=body.get("approved_by"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(entry)


@app.post("/api/registry/promote")
async def registry_promote(request: Request):
    body = await request.json()
    try:
        entry = promote_entry(
            body["model_id"],
            body["version"],
            approved_by=body.get("approved_by") or "operator",
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(entry)


@app.post("/api/registry/rollback")
async def registry_rollback(request: Request):
    body = await request.json()
    try:
        entry = rollback_active(by=body.get("by") or "operator")
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(entry)


@app.get("/api/pacs/queue")
def pacs_queue():
    return JSONResponse({"studies": list_studies()})


@app.get("/api/pacs/scp")
def pacs_scp_get():
    return JSONResponse(scp_status())


@app.post("/api/pacs/scp/start")
def pacs_scp_start():
    return JSONResponse(start_scp_background())


@app.get("/report/{scan_id}")
def report_page(request: Request, scan_id: str):
    """Page 3 — output / clinician review."""
    path = REPORTS / f"{scan_id}.json"
    if not path.exists():
        raise HTTPException(404, "Scan not found")
    report = json.loads(path.read_text(encoding="utf-8"))
    reviews = get_report_reviews(scan_id)
    review_map = {str(r["finding_id"]): r for r in reviews.get("findings", [])}
    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "report": report,
            "reviews": reviews,
            "review_map": review_map,
            "page": "output",
        },
    )
