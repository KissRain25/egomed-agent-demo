#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
from ultralytics import YOLO

DATA_ROOT = Path(f"{REPO_ROOT}/data/yolo_cls_tool_selection")
PROJECT = Path(f"{REPO_ROOT}/runs/tool_selection_cls")

MODEL = "yolo26m-cls.pt"

EPOCHS = 100
IMGSZ = 320
BATCH = 64
DEVICE = 2
WORKERS = 8

RUN_NAME = f"tool_selection_yolo26m_cls_imgsz{IMGSZ}"

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Classification data not found: {DATA_ROOT}")

    print("=" * 100)
    print("Training Detection Agent tool-selection classifier")
    print(f"Data root: {DATA_ROOT}")
    print(f"Model:     {MODEL}")
    print(f"Project:   {PROJECT}")
    print(f"Run name:  {RUN_NAME}")
    print("=" * 100)

    model = YOLO(MODEL)

    model.train(
        task="classify",
        data=str(DATA_ROOT),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        warmup_epochs=3,
        patience=30,
        hsv_h=0.0,
        hsv_s=0.05,
        hsv_v=0.1,
        degrees=0.0,
        translate=0.05,
        scale=0.1,
        fliplr=0.0,
        flipud=0.0,
    )

    best = PROJECT / RUN_NAME / "weights" / "best.pt"
    print("\n[DONE]")
    print(f"Best weight: {best}")

    if best.exists():
        print("\nEvaluating best.pt on test split...")
        model = YOLO(str(best))
        model.val(
            task="classify",
            data=str(DATA_ROOT),
            split="test",
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
        )

if __name__ == "__main__":
    main()
