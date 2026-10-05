"""Phase 3: model registry with promote / rollback / audit log."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from .gate import validate_entry

REGISTRY_DIR = Path(__file__).resolve().parent
ENTRIES_DIR = REGISTRY_DIR / "entries"
ACTIVE_PATH = REGISTRY_DIR / "active.json"
AUDIT_PATH = REGISTRY_DIR / "audit.log"

ENTRIES_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _audit(action: str, detail: dict[str, Any]) -> None:
    line = json.dumps({"ts": _now(), "action": action, **detail}, ensure_ascii=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry_path(model_id: str, version: str) -> Path:
    safe = f"{model_id}__{version}.json".replace("/", "_")
    return ENTRIES_DIR / safe


def register_entry(entry: dict[str, Any], *, approved_by: str | None = None) -> dict[str, Any]:
    entry = dict(entry)
    entry.setdefault("status", "registered")
    entry.setdefault("registered_at", _now())
    if approved_by:
        entry["approved_by"] = approved_by
    errors = validate_entry(entry, require_files=False)
    if errors:
        raise ValueError("; ".join(errors))
    path = _entry_path(entry["model_id"], entry["version"])
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    _audit("register", {"model_id": entry["model_id"], "version": entry["version"]})
    return entry


def list_entries() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(ENTRIES_DIR.glob("*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def get_active_entry() -> dict[str, Any] | None:
    if not ACTIVE_PATH.exists():
        return None
    return json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))


def promote_entry(model_id: str, version: str, *, approved_by: str) -> dict[str, Any]:
    path = _entry_path(model_id, version)
    if not path.exists():
        raise FileNotFoundError(f"registry entry not found: {model_id}@{version}")
    entry = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_entry(entry, require_files=True)
    if errors:
        _audit("promote_rejected", {"model_id": model_id, "version": version, "errors": errors})
        raise ValueError(f"validation gate failed: {errors}")

    previous = get_active_entry()
    if previous:
        rollback_path = REGISTRY_DIR / "previous.json"
        rollback_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")

    entry["status"] = "approved"
    entry["approved_by"] = approved_by
    entry["approved_at"] = _now()
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    ACTIVE_PATH.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    _audit(
        "promote",
        {
            "model_id": model_id,
            "version": version,
            "approved_by": approved_by,
            "previous": None if not previous else f"{previous.get('model_id')}@{previous.get('version')}",
        },
    )
    return entry


def rollback_active(*, by: str) -> dict[str, Any] | None:
    previous_path = REGISTRY_DIR / "previous.json"
    if not previous_path.exists():
        raise FileNotFoundError("no previous active model to roll back to")
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    ACTIVE_PATH.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    _audit("rollback", {"by": by, "restored": f"{previous.get('model_id')}@{previous.get('version')}"})
    return previous


def ensure_demo_entry() -> dict[str, Any]:
    """Register the shipping PneumoniaMNIST demo model if missing."""
    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "pneumonia_cnn.pt"
    metrics_path = Path(__file__).resolve().parents[1] / "checkpoints" / "metrics.json"
    if not metrics_path.exists():
        return {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    entry = {
        "model_id": "pneumonia_cnn",
        "version": "v0-demo",
        "dataset": metrics.get("dataset", "pneumoniamnist"),
        "modality": "DX",
        "mode": "demo",
        "checkpoint_path": str(ckpt),
        "metrics_path": str(metrics_path),
        "checksum_sha256": file_checksum(ckpt) if ckpt.exists() else "pending",
        "test_metrics": metrics.get("test_metrics", {}),
        "status": "approved",
        "approved_by": "bootstrap",
        "notes": "Demo classifier shipped with the prototype.",
    }
    register_entry(entry, approved_by="bootstrap")
    if get_active_entry() is None:
        # Demo mode bypasses clinical gate thresholds via mode=demo
        ACTIVE_PATH.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        _audit("bootstrap_active", {"model_id": "pneumonia_cnn", "version": "v0-demo"})
    return entry
