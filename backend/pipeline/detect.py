"""Candidate region-of-interest detection.

IMPORTANT — this is a classical computer-vision heuristic (Laplacian-of-Gaussian
blob detection + local contrast scoring), NOT a trained clinical detection model.
There is no labeled hospital training data available in this environment, so it
cannot claim any validated sensitivity/specificity. It exists to make the
end-to-end pipeline (upload -> candidate findings -> report) runnable and to
mark the exact seam where a real trained model (e.g. nnDetection / nnU-Net,
per the project plan's Section 7) plugs in later — see backend/pipeline/model.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from skimage.feature import blob_log

MIN_SIGMA = 3
MAX_SIGMA = 25
NUM_SIGMA = 12
BLOB_THRESHOLD = 0.045
CONTRAST_FLOOR = 8
MAX_FINDINGS = 8
BACKGROUND_INTENSITY_FLOOR = 12
MERGE_DISTANCE_PX = 18


@dataclass
class Finding:
    id: int
    x: int
    y: int
    radius_px: float
    polarity: str  # "hyperintense" (brighter than surroundings) or "hypointense" (darker)
    confidence: float  # heuristic salience score in [0, 1], NOT a calibrated clinical probability


def _local_contrast(img: np.ndarray, y: int, x: int, r: float) -> float:
    """Mean intensity inside the blob minus mean intensity in a surrounding ring."""
    h, w = img.shape
    r_in = max(2, int(round(r)))
    r_out = int(round(r * 2.2)) + 2

    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - y) ** 2 + (xx - x) ** 2

    inner_mask = dist2 <= r_in ** 2
    ring_mask = (dist2 > r_in ** 2) & (dist2 <= r_out ** 2)

    if not inner_mask.any() or not ring_mask.any():
        return 0.0

    inner_mean = float(img[inner_mask].mean())
    ring_mean = float(img[ring_mask].mean())
    return inner_mean - ring_mean


def _dedupe(blobs: list[tuple[int, int, float, float]]) -> list[tuple[int, int, float, float]]:
    """Greedy dedupe of (y, x, r, score) by center distance, keeping the higher-score blob."""
    blobs = sorted(blobs, key=lambda b: -b[3])
    kept: list[tuple[int, int, float, float]] = []
    for y, x, r, score in blobs:
        if all((y - ky) ** 2 + (x - kx) ** 2 > MERGE_DISTANCE_PX ** 2 for ky, kx, kr, ks in kept):
            kept.append((y, x, r, score))
    return kept


def detect_findings(enhanced: np.ndarray, original: np.ndarray) -> list[Finding]:
    norm = enhanced.astype(np.float64) / 255.0

    # blob_log only finds blobs brighter than their surroundings, so run it a
    # second time on the inverted image to also catch hypointense (darker) blobs.
    raw_blobs_bright = blob_log(
        norm, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA, num_sigma=NUM_SIGMA,
        threshold=BLOB_THRESHOLD, overlap=0.3,
    )
    raw_blobs_dark = blob_log(
        1.0 - norm, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA, num_sigma=NUM_SIGMA,
        threshold=BLOB_THRESHOLD, overlap=0.3,
    )
    raw_blobs = np.vstack([raw_blobs_bright, raw_blobs_dark]) if len(raw_blobs_dark) else raw_blobs_bright

    candidates: list[tuple[int, int, float, float, str]] = []
    for y, x, sigma in raw_blobs:
        y, x = int(y), int(x)
        if original[y, x] < BACKGROUND_INTENSITY_FLOOR:
            continue  # sits in the padded/black background, not tissue
        r = float(sigma * np.sqrt(2))
        contrast = _local_contrast(enhanced.astype(np.float64), y, x, r)
        if abs(contrast) < CONTRAST_FLOOR:
            continue  # negligible contrast vs surrounding tissue, not a real candidate
        polarity = "hyperintense" if contrast > 0 else "hypointense"
        size_factor = min(1.0, r / MAX_SIGMA)
        contrast_factor = min(1.0, abs(contrast) / 80.0)
        score = 0.6 * contrast_factor + 0.4 * size_factor
        candidates.append((y, x, r, score, polarity))

    deduped = _dedupe([(y, x, r, s) for y, x, r, s, _ in candidates])
    polarity_lookup = {(y, x, r): p for y, x, r, s, p in candidates}

    deduped.sort(key=lambda b: -b[3])
    top = deduped[:MAX_FINDINGS]

    findings = []
    for i, (y, x, r, score) in enumerate(top, start=1):
        polarity = polarity_lookup.get((y, x, r), "hyperintense")
        findings.append(Finding(
            id=i, x=x, y=y, radius_px=round(r, 1),
            polarity=polarity, confidence=round(min(0.97, max(0.05, score)), 2),
        ))
    return findings


def findings_to_dicts(findings: list[Finding]) -> list[dict]:
    return [asdict(f) for f in findings]
