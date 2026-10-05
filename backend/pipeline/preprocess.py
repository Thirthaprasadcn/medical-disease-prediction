"""Scan preprocessing: load an uploaded image and normalize it for analysis.

DICOM studies are ingested in dicom_ingest.py (windowing + metadata) and then
passed here as a derived PNG/uint8 slice. Full NIfTI / 3D resampling remains
an extension for volumetric clinical models.
"""
from __future__ import annotations

import cv2
import numpy as np


TARGET_SIZE = 512


def load_grayscale(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image — expected a valid PNG/JPG file")
    return img


def resize_keep_aspect(img: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)


def preprocess(image_bytes: bytes) -> dict:
    """Returns the original (resized) grayscale image and an enhanced version used for detection."""
    raw = load_grayscale(image_bytes)
    resized = resize_keep_aspect(raw)
    enhanced = enhance_contrast(denoise(resized))
    return {"original": resized, "enhanced": enhanced}
