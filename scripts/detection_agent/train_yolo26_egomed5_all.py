#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
from ultralytics import YOLO


DATA_YAML = Path(f"{REPO_ROOT}/data/yolo_det/EgoMed5_All/dataset.yaml")
PROJECT = Path(f"{REPO_ROOT}/runs/yolo26_det")

MODEL = "yolo26m.pt"

RUN_NAME = "EgoMed5_All_yolo26m_imgsz1024"

EPOCHS = 150
IMGSZ = 1024
BATCH = 4
DEVICE = 1
WORKERS = 8


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {DATA_YAML}")

    print("=" * 100)
    print("Training universal YOLO26m detector on EgoMed5_All")
    print(f"Data yaml: {DATA_YAML}")
    print(f"Run name:  {RUN_NAME}")
    print("=" * 100)

    model = YOLO(MODEL)

    results = model.train(
        task="detect",
        data=str(DATA_YAML),

        # Basic training settings
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,

        # Optimizer settings
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        warmup_epochs=3,
        patience=50,

        # Geometry augmentation
        # Important: no left-right flip because of left/right anatomical classes.
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,

        # Composition augmentation
        # Conservative setting for medical image structure.
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        # Color / brightness augmentation
        hsv_h=0.0,
        hsv_s=0.05,
        hsv_v=0.1,
    )

    best_weight = PROJECT / RUN_NAME / "weights" / "best.pt"
    last_weight = PROJECT / RUN_NAME / "weights" / "last.pt"

    print("\n[Done] EgoMed5_All")
    print(f"Best weight: {best_weight}")
    print(f"Last weight: {last_weight}")

    return results


if __name__ == "__main__":
    main()
