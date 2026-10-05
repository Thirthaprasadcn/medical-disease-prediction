"""Phase 2B: BraTS + nnU-Net training wrapper.

Full training needs GPU and the BraTS dataset (~20GB+):
  https://www.med.upenn.edu/cbica/brats/
  https://github.com/MIC-DKFZ/nnUNet

Run from project root:
  backend/.venv/Scripts/python.exe -m backend.training.train_nnunet_brats
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "checkpoints" / "brats_nnunet"
DATA_DEFAULT = ROOT / "data" / "BraTS"


def write_placeholder_metrics(reason: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {
        "detector_id": "nnunet-brats",
        "dataset": "BraTS",
        "framework": "nnU-Net",
        "trained_at": None,
        "status": "not_trained",
        "reason": reason,
        "test_metrics": {
            "sensitivity": None,
            "specificity": None,
            "dice": None,
            "note": "Populate after a real hold-out evaluation.",
        },
        "instructions": [
            "1. Download BraTS into backend/data/BraTS",
            "2. pip install nnunetv2 (GPU PyTorch build recommended)",
            "3. nnUNetv2_plan_and_preprocess -d <dataset_id>",
            "4. Re-run this script with --train",
            "5. Promote checkpoint through the model registry validation gate",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote placeholder metrics to {OUT / 'metrics.json'}")


def try_train(data_dir: Path) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        write_placeholder_metrics(f"Dataset not found at {data_dir}")
        return 2

    cmd = [sys.executable, "-m", "nnunetv2.run.run_training", "Dataset999_BraTS", "3d_fullres", "0"]
    try:
        print("Attempting:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            write_placeholder_metrics(
                "nnU-Net training command failed or CLI not available. "
                "Install nnunetv2 and configure Dataset999_BraTS, then retry."
            )
            return completed.returncode
    except Exception as exc:  # noqa: BLE001
        write_placeholder_metrics(f"Could not launch nnU-Net: {exc}")
        return 1

    metrics = {
        "detector_id": "nnunet-brats",
        "dataset": "BraTS",
        "framework": "nnU-Net",
        "trained_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": "trained",
        "test_metrics": {
            "sensitivity": 0.0,
            "specificity": 0.0,
            "dice": 0.0,
            "note": "Replace with real evaluation numbers from nnU-Net.",
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Training finished — update test_metrics from the nnU-Net evaluation report.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BraTS / nnU-Net training wrapper")
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.train:
        return try_train(args.data)
    write_placeholder_metrics("Stub generated; pass --train when dataset + nnU-Net are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
