"""Phase 6: DICOM C-STORE SCP scaffolding via pynetdicom.

Starts an optional background listener that stores received instances under
storage/dicom/pacs/ and enqueues them for analysis. Disabled by default;
enable with MIR_PACS_SCP=1 or POST /api/pacs/scp/start.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from backend.pacs.study_queue import enqueue_study

STORAGE = Path(__file__).resolve().parent.parent / "storage" / "dicom" / "pacs"
_state: dict[str, Any] = {
    "running": False,
    "ae_title": os.environ.get("MIR_AE_TITLE", "MIR_SCP"),
    "port": int(os.environ.get("MIR_PACS_PORT", "11112")),
    "received": 0,
    "last_error": None,
    "thread": None,
}


def scp_status() -> dict[str, Any]:
    return {
        "running": _state["running"],
        "ae_title": _state["ae_title"],
        "port": _state["port"],
        "received": _state["received"],
        "last_error": _state["last_error"],
        "note": (
            "Research SCP for lab/demo use. Use TLS, AE Title allow-lists, and "
            "institutional agreements before any PHI network exposure."
        ),
    }


def _handle_store(event):
    """pynetdicom evt.EVT_C_STORE handler."""
    try:
        ds = event.dataset
        ds.file_meta = event.file_meta
        STORAGE.mkdir(parents=True, exist_ok=True)
        sop = getattr(ds, "SOPInstanceUID", "unknown")
        out = STORAGE / f"{sop}.dcm"
        ds.save_as(str(out), write_like_original=False)
        enqueue_study(
            source="pacs_cstore",
            filename=out.name,
            modality=str(getattr(ds, "Modality", "") or ""),
            study_uid=str(getattr(ds, "StudyInstanceUID", "") or ""),
            patient_id=str(getattr(ds, "PatientID", "") or ""),
            dicom_path=str(out),
            ae_title=str(event.assoc.requestor.ae_title),
        )
        _state["received"] += 1
        return 0x0000
    except Exception as exc:  # noqa: BLE001
        _state["last_error"] = str(exc)
        return 0xC210


def _run_scp() -> None:
    try:
        from pynetdicom import AE, evt, AllStoragePresentationContexts
        from pynetdicom.sop_class import Verification
    except ImportError as exc:
        _state["last_error"] = f"pynetdicom not installed: {exc}"
        _state["running"] = False
        return

    ae = AE(ae_title=_state["ae_title"])
    ae.supported_contexts = AllStoragePresentationContexts
    ae.add_supported_context(Verification)
    handlers = [(evt.EVT_C_STORE, _handle_store)]
    _state["running"] = True
    try:
        ae.start_server(("", _state["port"]), block=True, evt_handlers=handlers)
    except Exception as exc:  # noqa: BLE001
        _state["last_error"] = str(exc)
    finally:
        _state["running"] = False


def start_scp_background() -> dict[str, Any]:
    if _state["running"]:
        return scp_status()
    t = threading.Thread(target=_run_scp, name="mir-pacs-scp", daemon=True)
    _state["thread"] = t
    t.start()
    return scp_status()
