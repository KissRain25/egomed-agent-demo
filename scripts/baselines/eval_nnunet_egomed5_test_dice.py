import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(f"{REPO_ROOT}")

DATASET_ID = 501
DATASET_NAME = "Dataset501_EgoMed5"
CONFIG = "2d"
FOLD = 0
GPU_ID = "0"

NNUNET_RAW = BASE_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = BASE_DIR / "nnUNet_results"

RAW_DATASET_DIR = NNUNET_RAW / DATASET_NAME
IMAGES_TS = RAW_DATASET_DIR / "imagesTs"
LABELS_TS = RAW_DATASET_DIR / "labelsTs"
MAPPING_JSON = RAW_DATASET_DIR / "global_label_mapping.json"

RUN_ROOT = BASE_DIR / "runs" / "eval_nnunet" / "Dataset501_EgoMed5_2d_fold0"
PRED_DIR = RUN_ROOT / "predictions"
EXCEL_OUTPUT = RUN_ROOT / "Dataset501_EgoMed5_nnunet_2d_fold0_test_dice.xlsx"

# 用 best checkpoint 还是 final checkpoint
CHECKPOINT_NAME = "checkpoint_best.pth"
# CHECKPOINT_NAME = "checkpoint_final.pth"

OVERWRITE_PRED = True

# Amos 只统计这 5 个目标；其他数据集统计全部目标
FOCUS_TARGETS_BY_DATASET = {
    "Amos": [
        "liver",
        "right kidney",
        "left kidney",
        "spleen",
        "stomach",
    ],
}


# ============================================================
# Utilities
# ============================================================

def run_cmd(cmd, env=None):
    print("\n" + "=" * 100)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True, env=env)


def dice_score(pred_mask, gt_mask, eps=1e-6):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    inter = np.logical_and(pred_mask, gt_mask).sum()
    return float((2.0 * inter + eps) / (pred_sum + gt_sum + eps))


def safe_dataset_name(name):
    return str(name).replace("-", "_").replace(".", "_")


def load_global_mapping(mapping_json):
    with open(mapping_json, "r") as f:
        data = json.load(f)

    global_labels = {
        name: int(label_id)
        for name, label_id in data["global_labels"].items()
    }

    dataset_gray_to_global = {
        dataset: {
            int(gray): int(global_id)
            for gray, global_id in mapping.items()
        }
        for dataset, mapping in data["dataset_gray_to_global"].items()
    }

    dataset_gray_to_name = {
        dataset: {
            int(gray): class_name
            for gray, class_name in mapping.items()
        }
        for dataset, mapping in data["dataset_gray_to_name"].items()
    }

    return global_labels, dataset_gray_to_global, dataset_gray_to_name


def build_target_infos(dataset_gray_to_global, dataset_gray_to_name):
    infos = []

    for dataset, gray_to_global in dataset_gray_to_global.items():
        focus_targets = FOCUS_TARGETS_BY_DATASET.get(dataset, None)

        for gray_value, global_label_id in sorted(gray_to_global.items(), key=lambda x: x[0]):
            class_name = dataset_gray_to_name[dataset][gray_value]

            if focus_targets is not None and class_name not in focus_targets:
                continue

            infos.append({
                "dataset": dataset,
                "target_class": class_name,
                "label_gray_value": int(gray_value),
                "global_label_id": int(global_label_id),
            })

    return infos


def parse_identifier(identifier, dataset_names):
    """
    Example identifier:
        Amos_263_263_0000
        Montgomery_County_CXR_Set_12_12_0000

    Return:
        dataset, case_id, frame_stem
    """
    safe_to_raw = {
        safe_dataset_name(d): d
        for d in dataset_names
    }

    # 长名字优先，避免前缀误匹配
    for safe_name in sorted(safe_to_raw.keys(), key=len, reverse=True):
        prefix = safe_name + "_"
        if identifier.startswith(prefix):
            dataset = safe_to_raw[safe_name]
            rest = identifier[len(prefix):]

            if "_" in rest:
                case_id, frame_stem = rest.split("_", 1)
            else:
                case_id = rest
                frame_stem = ""

            return dataset, case_id, frame_stem

    return "unknown", "unknown", identifier


def read_label(path):
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if label is None:
        raise FileNotFoundError(f"Cannot read label: {path}")

    if label.ndim == 3:
        label = label[:, :, 0]

    return label.astype(np.int32)


# ============================================================
# Prediction
# ============================================================

def run_nnunet_predict():
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    if OVERWRITE_PRED and PRED_DIR.exists():
        print(f"[Remove old predictions] {PRED_DIR}")
        shutil.rmtree(PRED_DIR)

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(IMAGES_TS),
        "-o", str(PRED_DIR),
        "-d", str(DATASET_ID),
        "-c", CONFIG,
        "-f", str(FOLD),
        "-chk", CHECKPOINT_NAME,
    ]

    run_cmd(cmd, env=env)


# ============================================================
# Dice evaluation
# ============================================================

def evaluate_dice():
    if not LABELS_TS.exists():
        raise FileNotFoundError(f"labelsTs not found: {LABELS_TS}")

    if not PRED_DIR.exists():
        raise FileNotFoundError(f"Prediction dir not found: {PRED_DIR}")

    if not MAPPING_JSON.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {MAPPING_JSON}")

    global_labels, dataset_gray_to_global, dataset_gray_to_name = load_global_mapping(MAPPING_JSON)
    target_infos = build_target_infos(dataset_gray_to_global, dataset_gray_to_name)

    dataset_names = list(dataset_gray_to_global.keys())

    label_files = sorted(LABELS_TS.glob("*.png"))

    print(f"\nNumber of test labels: {len(label_files)}")

    frame_rows = []

    for gt_path in tqdm(label_files, desc="Computing Dice"):
        identifier = gt_path.stem
        pred_path = PRED_DIR / f"{identifier}.png"

        if not pred_path.exists():
            print(f"[Warning] Missing prediction: {pred_path}")
            continue

        gt_label = read_label(gt_path)
        pred_label = read_label(pred_path)

        if pred_label.shape != gt_label.shape:
            pred_label = cv2.resize(
                pred_label.astype(np.int32),
                (gt_label.shape[1], gt_label.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        dataset, case_id, frame_stem = parse_identifier(identifier, dataset_names)

        for info in target_infos:
            if info["dataset"] != dataset:
                continue

            global_label_id = info["global_label_id"]

            gt_mask = gt_label == global_label_id

            # 和你前面评估保持一致：只统计 GT 中该目标真实存在的帧
            if gt_mask.sum() == 0:
                continue

            pred_mask = pred_label == global_label_id

            d = dice_score(pred_mask, gt_mask)
            inter = int(np.logical_and(pred_mask, gt_mask).sum())

            frame_rows.append({
                "dataset": dataset,
                "case_id": case_id,
                "frame_identifier": identifier,
                "frame_stem": frame_stem,
                "target_class": info["target_class"],
                "label_gray_value": info["label_gray_value"],
                "global_label_id": global_label_id,
                "frame_dice": d,
                "gt_area": int(gt_mask.sum()),
                "pred_area": int(pred_mask.sum()),
                "intersection": inter,
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
            })

    frame_df = pd.DataFrame(frame_rows)

    if len(frame_df) == 0:
        case_df = pd.DataFrame()
        class_summary = pd.DataFrame()
    else:
        case_df = (
            frame_df.groupby(
                [
                    "dataset",
                    "case_id",
                    "target_class",
                    "label_gray_value",
                    "global_label_id",
                ],
                dropna=False,
            )
            .agg(
                mean_dice=("frame_dice", "mean"),
                std_dice=("frame_dice", "std"),
                min_dice=("frame_dice", "min"),
                max_dice=("frame_dice", "max"),
                valid_gt_frames=("frame_dice", "count"),
                mean_gt_area=("gt_area", "mean"),
                mean_pred_area=("pred_area", "mean"),
            )
            .reset_index()
        )

        class_summary = (
            case_df.groupby(
                [
                    "dataset",
                    "target_class",
                    "label_gray_value",
                    "global_label_id",
                ],
                dropna=False,
            )
            .agg(
                mean_dice=("mean_dice", "mean"),
                std_dice=("mean_dice", "std"),
                min_dice=("mean_dice", "min"),
                max_dice=("mean_dice", "max"),
                num_cases=("case_id", "count"),
                total_valid_gt_frames=("valid_gt_frames", "sum"),
                mean_valid_gt_frames=("valid_gt_frames", "mean"),
                mean_gt_area=("mean_gt_area", "mean"),
                mean_pred_area=("mean_pred_area", "mean"),
            )
            .reset_index()
            .sort_values(["dataset", "global_label_id"])
        )

    with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
        frame_df.to_excel(writer, sheet_name="frame_class_dice", index=False)
        case_df.to_excel(writer, sheet_name="case_class_dice", index=False)
        class_summary.to_excel(writer, sheet_name="class_summary", index=False)

    print("\nDone.")
    print(f"Saved Excel: {EXCEL_OUTPUT}")
    print(f"Prediction dir: {PRED_DIR}")

    if len(class_summary) > 0:
        print("\nDataset-level mean Dice:")
        print(
            class_summary[
                [
                    "dataset",
                    "target_class",
                    "global_label_id",
                    "mean_dice",
                    "std_dice",
                    "num_cases",
                    "total_valid_gt_frames",
                ]
            ].to_string(index=False)
        )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 120)
    print("nnU-Net EgoMed5 Test Dice Evaluation")
    print("=" * 120)
    print(f"Dataset: {DATASET_ID} / {DATASET_NAME}")
    print(f"Config: {CONFIG}")
    print(f"Fold: {FOLD}")
    print(f"Checkpoint: {CHECKPOINT_NAME}")
    print(f"ImagesTs: {IMAGES_TS}")
    print(f"LabelsTs: {LABELS_TS}")
    print(f"Prediction dir: {PRED_DIR}")
    print(f"Excel output: {EXCEL_OUTPUT}")
    print("=" * 120)

    run_nnunet_predict()
    evaluate_dice()


if __name__ == "__main__":
    main()