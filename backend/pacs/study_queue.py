"""Phase 6: local study queue for PACS-received / uploaded DICOM studies."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

QUEUE_PATH = Path(__file__).resolve().parent.parent / "storage" / "study_queue.json"


def _load() -> list[dict[str, Any]]:
    if not QUEUE_PATH.exists():
        return []
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def _save(items: list[dict[str, Any]]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def enqueue_study(
    *,
    source: str,
    filename: str,
    modality: str | None = None,
    study_uid: str | None = None,
    patient_id: str | None = None,
    dicom_path: str | None = None,
    ae_title: str | None = None,
) -> dict[str, Any]:
    items = _load()
    entry = {
        "queue_id": uuid4().hex[:12],
        "received_at": dt.datetime.utcnow().isoformat() + "Z",
        "source": source,
        "filename": filename,
        "modality": modality,
        "study_instance_uid": study_uid,
        "patient_id": patient_id,
        "dicom_path": dicom_path,
        "ae_title": ae_title,
        "status": "queued",
        "scan_id": None,
    }
    items.insert(0, entry)
    _save(items[:500])
    return entry


def list_studies(limit: int = 50) -> list[dict[str, Any]]:
    return _load()[:limit]


def mark_processed(queue_id: str, scan_id: str) -> dict[str, Any] | None:
    items = _load()
    for item in items:
        if item["queue_id"] == queue_id:
            item["status"] = "processed"
            item["scan_id"] = scan_id
            item["processed_at"] = dt.datetime.utcnow().isoformat() + "Z"
            _save(items)
            return item
    return None
