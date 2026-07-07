from pathlib import Path

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
from ultralytics import YOLO


DATA_ROOT = Path(f"{REPO_ROOT}/data/yolo_det")
PROJECT = Path(f"{REPO_ROOT}/runs/yolo26_det")

DATASETS = [
    "Amos",
    # "CAMUS",
    # "ACDC",
    # "Montgomery-County-CXR-Set",
    # "PolypGen2021_MultiCenterData_v3",
]

MODEL = "yolo26m.pt"

EPOCHS = 150
IMGSZ = 1024
BATCH = 4
DEVICE = 1
WORKERS = 8


def train_one_dataset(dataset_name):
    data_yaml = DATA_ROOT / dataset_name / "dataset.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")

    run_name = f"{dataset_name}_yolo26m_imgsz{IMGSZ}"

    print("=" * 100)
    print(f"Training YOLO26m detector")
    print(f"Dataset: {dataset_name}")
    print(f"Data yaml: {data_yaml}")
    print(f"Run name: {run_name}")
    print("=" * 100)

    model = YOLO(MODEL)

    results = model.train(
        task="detect",
        data=str(data_yaml),

        # Basic training settings
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT),
        name=run_name,
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

    best_weight = PROJECT / run_name / "weights" / "best.pt"
    last_weight = PROJECT / run_name / "weights" / "last.pt"

    print(f"\n[Done] {dataset_name}")
    print(f"Best weight: {best_weight}")
    print(f"Last weight: {last_weight}")

    return results


def main():
    for dataset_name in DATASETS:
        train_one_dataset(dataset_name)

    print("\nAll YOLO26m detection models finished training.")


if __name__ == "__main__":
    main()