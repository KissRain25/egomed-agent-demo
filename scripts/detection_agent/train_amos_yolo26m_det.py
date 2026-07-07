# train_amos_yolo26m_det.py
from pathlib import Path

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
import shutil
import heapq


DATA_ROOT = Path(f"{REPO_ROOT}/data/yolo_det")
PROJECT = Path(f"{REPO_ROOT}/runs/yolo26_det")

DATASETS = [
    "Amos",
]

MODEL = "yolo26m.pt"

EPOCHS = 300
IMGSZ = 1024
BATCH = 4
DEVICE = 2
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

    weights_dir = PROJECT / run_name / "weights"
    top3_dir = PROJECT / run_name / "weights_top3"
    top3_dir.mkdir(parents=True, exist_ok=True)

    top3_heap = []

    def on_fit_epoch_end(trainer):
        metrics = trainer.metrics
        map_score = metrics.get("metrics/mAP50-95(B)", 0.0)
        epoch = trainer.epoch + 1

        current_weight = weights_dir / "last.pt"
        if not current_weight.exists():
            return

        save_path = top3_dir / f"epoch{epoch}_map{map_score:.4f}.pt"
        shutil.copy(current_weight, save_path)

        heapq.heappush(top3_heap, (map_score, epoch, save_path))

        if len(top3_heap) > 3:
            worst_score, worst_epoch, worst_path = heapq.heappop(top3_heap)
            if worst_path.exists():
                worst_path.unlink()
            print(f"[Top3] Removed epoch{worst_epoch} (mAP={worst_score:.4f})")

        print(f"[Top3] Epoch {epoch}: mAP50-95={map_score:.4f} | "
              f"Current top3: {sorted([(e, f'{s:.4f}') for s, e, _ in top3_heap], reverse=True)}")

    model = YOLO(MODEL)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    results = model.train(
        task="detect",
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT),
        name=run_name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        warmup_epochs=3,
        patience=50,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.05,
        hsv_v=0.1,
    )

    print(f"\n[Done] {dataset_name}")
    print(f"Top3 weights saved to: {top3_dir}")
    print("Final top3:")
    for score, epoch, path in sorted(top3_heap, reverse=True):
        print(f"  Epoch {epoch}: mAP50-95={score:.4f} -> {path.name}")

    return results


def main():
    for dataset_name in DATASETS:
        train_one_dataset(dataset_name)
    print("\nAll YOLO26m detection models finished training.")


if __name__ == "__main__":
    main()
