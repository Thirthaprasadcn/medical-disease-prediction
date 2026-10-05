"""Phase 1: DICOM ingestion with pydicom.

Accepts a single DICOM instance or a zip/folder of instances belonging to one
series, applies modality-appropriate windowing, and returns a 2D uint8 slice
plus metadata that feeds the existing preprocess → inference → report chain.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
except ImportError:  # pragma: no cover
    pydicom = None  # type: ignore


# Default windows (center, width) for common clinical views
WINDOWS = {
    "CT": {"lung": ( -600, 1500), "soft": (40, 400), "bone": (300, 1500), "default": (40, 400)},
    "MR": {"default": (None, None)},  # percentile stretch
    "DX": {"default": (None, None)},
    "CR": {"default": (None, None)},
}


class DicomIngestError(ValueError):
    pass


def is_dicom_bytes(data: bytes) -> bool:
    if len(data) < 132:
        return False
    if data[128:132] == b"DICM":
        return True
    # Some files omit the preamble; try a cheap tag sniff
    return data[:2] in (b"\x00\x00", b"\xfe\xff", b"\x08\x00") or b"DICM" in data[:512]


def _require_pydicom() -> None:
    if pydicom is None:
        raise DicomIngestError("pydicom is not installed. Run: pip install pydicom")


def _read_dataset(data: bytes):
    _require_pydicom()
    try:
        return pydicom.dcmread(io.BytesIO(data), force=True)
    except Exception as exc:  # noqa: BLE001
        raise DicomIngestError(f"Failed to parse DICOM: {exc}") from exc


def _extract_metadata(ds) -> dict[str, Any]:
    def _get(tag: str, default: Any = None) -> Any:
        val = getattr(ds, tag, default)
        if val is None:
            return default
        return str(val)

    return {
        "patient_id": _get("PatientID"),
        "patient_name": _get("PatientName"),
        "study_instance_uid": _get("StudyInstanceUID"),
        "series_instance_uid": _get("SeriesInstanceUID"),
        "sop_instance_uid": _get("SOPInstanceUID"),
        "modality": _get("Modality", "OT"),
        "study_date": _get("StudyDate"),
        "study_description": _get("StudyDescription"),
        "series_description": _get("SeriesDescription"),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "slice_thickness": _get("SliceThickness"),
        "manufacturer": _get("Manufacturer"),
    }


def _window_to_uint8(pixels: np.ndarray, center: float | None, width: float | None) -> np.ndarray:
    arr = pixels.astype(np.float64)
    if center is None or width is None:
        lo, hi = np.percentile(arr, (1, 99))
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
            if hi <= lo:
                hi = lo + 1.0
    else:
        lo = center - width / 2.0
        hi = center + width / 2.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _pixels_from_dataset(ds) -> np.ndarray:
    if not hasattr(ds, "pixel_array"):
        raise DicomIngestError("DICOM has no pixel data")
    try:
        arr = apply_modality_lut(ds.pixel_array, ds)
    except Exception:  # noqa: BLE001
        arr = ds.pixel_array
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # multi-frame or RGB — take middle frame / luminance
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        else:
            arr = arr[arr.shape[0] // 2]
    if arr.ndim != 2:
        raise DicomIngestError(f"Unsupported pixel array shape: {arr.shape}")
    return arr


def dataset_to_slice(ds, window_name: str = "default") -> tuple[np.ndarray, dict[str, Any]]:
    meta = _extract_metadata(ds)
    modality = (meta.get("modality") or "OT").upper()
    pixels = _pixels_from_dataset(ds)

    windows = WINDOWS.get(modality, WINDOWS["MR"])
    center, width = windows.get(window_name, windows["default"])
    if modality == "CT" and window_name == "default":
        # Prefer lung window for chest CT demos when series description hints lung
        desc = (meta.get("series_description") or "").lower()
        if "lung" in desc or "thorax" in desc or "chest" in desc:
            center, width = WINDOWS["CT"]["lung"]

    slice_u8 = _window_to_uint8(pixels, center, width)
    meta["window"] = {"center": center, "width": width, "name": window_name}
    return slice_u8, meta


def ingest_dicom_bytes(data: bytes, window_name: str = "default") -> dict[str, Any]:
    """Parse one DICOM instance → uint8 slice + metadata."""
    ds = _read_dataset(data)
    slice_u8, meta = dataset_to_slice(ds, window_name=window_name)
    return {"slice": slice_u8, "metadata": meta, "dataset": ds}


def ingest_dicom_zip(data: bytes, window_name: str = "default") -> dict[str, Any]:
    """Parse a zip of DICOM instances; pick the middle slice of the largest series."""
    _require_pydicom()
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DicomIngestError("Not a valid ZIP of DICOM files") from exc

    by_series: dict[str, list[tuple[float, bytes]]] = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        raw = zf.read(name)
        if not is_dicom_bytes(raw) and not name.lower().endswith((".dcm", ".dicom", ".ima")):
            continue
        try:
            ds = _read_dataset(raw)
        except DicomIngestError:
            continue
        series = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else float(getattr(ds, "InstanceNumber", 0) or 0)
        by_series.setdefault(series, []).append((z, raw))

    if not by_series:
        raise DicomIngestError("ZIP contained no readable DICOM instances")

    series_uid, instances = max(by_series.items(), key=lambda kv: len(kv[1]))
    instances.sort(key=lambda t: t[0])
    mid_raw = instances[len(instances) // 2][1]
    result = ingest_dicom_bytes(mid_raw, window_name=window_name)
    result["metadata"]["series_instance_uid"] = series_uid
    result["metadata"]["instance_count"] = len(instances)
    result["metadata"]["source"] = "dicom_zip"
    return result


def save_dicom_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def slice_to_png_bytes(slice_u8: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", slice_u8)
    if not ok:
        raise DicomIngestError("Failed to encode DICOM slice as PNG")
    return buf.tobytes()
