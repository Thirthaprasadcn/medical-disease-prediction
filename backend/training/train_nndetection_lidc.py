"""Phase 2A: LIDC-IDRI + nnDetection training wrapper.

This script prepares the expected dataset layout and either:
  1) Invokes nnDetection's training entrypoint when installed and data is present, or
  2) Writes a registry-ready metrics stub explaining what is required.

Full training needs GPU, ~100GB+ disk for LIDC-IDRI, and the nnDetection toolkit:
  https://github.com/MIC-DKFZ/nnDetection

Run from project root:
  backend/.venv/Scripts/python.exe -m backend.training.train_nndetection_lidc
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "checkpoints" / "lidc_nndetection"
DATA_DEFAULT = ROOT / "data" / "LIDC-IDRI"


def write_placeholder_metrics(reason: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {
        "detector_id": "nndetection-lidc",
        "dataset": "LIDC-IDRI",
        "framework": "nnDetection",
        "trained_at": None,
        "status": "not_trained",
        "reason": reason,
        "test_metrics": {
            "sensitivity": None,
            "specificity": None,
            "auc": None,
            "note": "Populate after a real hold-out evaluation.",
        },
        "instructions": [
            "1. Download LIDC-IDRI from TCIA into backend/data/LIDC-IDRI",
            "2. pip install nndetection (GPU PyTorch build recommended)",
            "3. Convert annotations to nnDetection dataset format",
            "4. Re-run this script with --train",
            "5. Register the resulting checkpoint via the model registry gate",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote placeholder metrics to {OUT / 'metrics.json'}")


def try_train(data_dir: Path) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        write_placeholder_metrics(f"Dataset not found at {data_dir}")
        return 2

    # Prefer the official CLI when available
    cmd = [sys.executable, "-m", "nndet.training", f"--data={data_dir}", f"--out={OUT}"]
    try:
        print("Attempting:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            write_placeholder_metrics(
                "nnDetection training command failed or CLI not available. "
                "Install nnDetection and convert LIDC to its dataset format, then retry."
            )
            return completed.returncode
    except Exception as exc:  # noqa: BLE001
        write_placeholder_metrics(f"Could not launch nnDetection: {exc}")
        return 1

    metrics = {
        "detector_id": "nndetection-lidc",
        "dataset": "LIDC-IDRI",
        "framework": "nnDetection",
        "trained_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": "trained",
        "test_metrics": {
            "sensitivity": 0.0,
            "specificity": 0.0,
            "auc": 0.0,
            "note": "Replace with real evaluation numbers from nnDetection's test split.",
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Training finished — update test_metrics from the nnDetection evaluation report.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LIDC-IDRI / nnDetection training wrapper")
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--train", action="store_true", help="Attempt to launch training")
    parser.add_argument("--stub", action="store_true", help="Only write placeholder metrics")
    args = parser.parse_args()
    if args.train:
        return try_train(args.data)
    write_placeholder_metrics("Stub generated; pass --train when dataset + nnDetection are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
