"""Phase 3: validation gate — refuse model promotion when checks fail."""
from __future__ import annotations

from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "model_id",
    "version",
    "dataset",
    "modality",
    "mode",
    "checkpoint_path",
    "metrics_path",
    "checksum_sha256",
    "test_metrics",
}

# Minimum hold-out thresholds for clinical promotion (tunable policy)
MIN_CLINICAL = {
    "sensitivity": 0.70,
    "specificity": 0.50,
}


def validate_entry(entry: dict[str, Any], *, require_files: bool = True) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(entry)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")

    metrics = entry.get("test_metrics") or {}
    if not isinstance(metrics, dict):
        errors.append("test_metrics must be an object")
    else:
        for key in ("sensitivity", "specificity"):
            if key not in metrics:
                errors.append(f"test_metrics.{key} required")

    if entry.get("mode") == "clinical":
        for key, minimum in MIN_CLINICAL.items():
            val = (entry.get("test_metrics") or {}).get(key)
            if val is None:
                continue
            try:
                if float(val) < minimum:
                    errors.append(f"{key}={val} below gate minimum {minimum}")
            except (TypeError, ValueError):
                errors.append(f"{key} is not numeric")

    if require_files:
        for field in ("checkpoint_path", "metrics_path"):
            path = entry.get(field)
            if path and not Path(path).exists():
                # Allow relative paths from repo root / backend
                alt = Path(__file__).resolve().parents[2] / path
                if not alt.exists():
                    errors.append(f"{field} not found: {path}")

    if not entry.get("checksum_sha256"):
        errors.append("checksum_sha256 required")

    return errors
