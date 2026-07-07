#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import re
import pandas as pd
from ultralytics import YOLO

DATA_ROOT = Path(f"{REPO_ROOT}/data")
WEIGHT = Path(f"{REPO_ROOT}/runs/tool_selection_cls/tool_selection_yolo26m_cls_imgsz320/weights/best.pt")

OUT_DIR = Path(f"{REPO_ROOT}/runs/tool_selection_cls/first_frame_test_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "first_frame_test_predictions.csv"
SUMMARY_CSV = OUT_DIR / "first_frame_test_summary.csv"

IMGSZ = 320
DEVICE = 2

DATASET_TO_CLASS = {
    "Amos": "CT",
    "ACDC": "MRI",
    "CAMUS": "Ultrasound",
    "Montgomery-County-CXR-Set": "Xray",
    "PolypGen2021_MultiCenterData_v3": "Endoscopy",
}

DISPLAY_NAME = {
    "Amos": "AMOS",
    "ACDC": "ACDC",
    "CAMUS": "CAMUS",
    "Montgomery-County-CXR-Set": "MCC",
    "PolypGen2021_MultiCenterData_v3": "PMC",
}


def natural_key(path_or_name):
    name = Path(str(path_or_name)).name
    nums = re.findall(r"\d+", name)
    return [int(x) for x in nums] if nums else [name]


def normalize_case_id(x):
    x = str(x).strip()
    try:
        return str(int(float(x)))
    except Exception:
        return x


def read_test_cases(split_file):
    cases = []
    current = None

    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("[") and s.endswith("]"):
                current = s[1:-1].strip()
                continue

            if current == "test":
                cases.append(normalize_case_id(s))

    return cases


def collect_images(case_dir):
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        files.extend(case_dir.glob(ext))
    return sorted(files, key=natural_key)


def main():
    if not WEIGHT.exists():
        raise FileNotFoundError(f"Weight not found: {WEIGHT}")

    print("=" * 100)
    print("Tool selection evaluation on first frame of each test case")
    print("=" * 100)
    print(f"Weight: {WEIGHT}")
    print(f"IMGSZ:  {IMGSZ}")
    print("=" * 100)

    model = YOLO(str(WEIGHT))
    names = model.names
    print("Model names:", names)

    rows = []

    for dataset_name, gt_class in DATASET_TO_CLASS.items():
        img_root = DATA_ROOT / dataset_name / "img"
        split_file = DATA_ROOT / "yolo_det" / dataset_name / "split_cases.txt"

        if not img_root.exists():
            raise FileNotFoundError(f"Image root not found: {img_root}")
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        test_cases = read_test_cases(split_file)

        print("\n" + "-" * 100)
        print(f"Dataset: {dataset_name} -> GT tool: {gt_class}")
        print(f"Num test cases: {len(test_cases)}")
        print("-" * 100)

        for case_id in test_cases:
            case_dir = img_root / str(case_id)
            imgs = collect_images(case_dir)

            if len(imgs) == 0:
                rows.append({
                    "dataset": DISPLAY_NAME.get(dataset_name, dataset_name),
                    "dataset_raw": dataset_name,
                    "case_id": case_id,
                    "first_frame": "",
                    "gt_tool": gt_class,
                    "pred_tool": "",
                    "conf": 0.0,
                    "correct": False,
                    "error": f"no image found in {case_dir}",
                })
                continue

            first_frame = imgs[0]

            try:
                result = model.predict(
                    source=str(first_frame),
                    imgsz=IMGSZ,
                    device=DEVICE,
                    verbose=False,
                )[0]

                probs = result.probs
                top1_idx = int(probs.top1)
                top1_conf = float(probs.top1conf)
                pred_tool = str(names[top1_idx])

                correct = pred_tool == gt_class

                rows.append({
                    "dataset": DISPLAY_NAME.get(dataset_name, dataset_name),
                    "dataset_raw": dataset_name,
                    "case_id": case_id,
                    "first_frame": str(first_frame),
                    "gt_tool": gt_class,
                    "pred_tool": pred_tool,
                    "conf": top1_conf,
                    "correct": bool(correct),
                    "error": "",
                })

            except Exception as e:
                rows.append({
                    "dataset": DISPLAY_NAME.get(dataset_name, dataset_name),
                    "dataset_raw": dataset_name,
                    "case_id": case_id,
                    "first_frame": str(first_frame),
                    "gt_tool": gt_class,
                    "pred_tool": "",
                    "conf": 0.0,
                    "correct": False,
                    "error": repr(e),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    summary = (
        df.groupby(["dataset", "gt_tool"], dropna=False)
        .agg(
            num_cases=("case_id", "count"),
            accuracy=("correct", "mean"),
            mean_conf=("conf", "mean"),
            num_errors=("error", lambda x: int((x.astype(str) != "").sum())),
        )
        .reset_index()
    )

    overall = pd.DataFrame([{
        "dataset": "Overall",
        "gt_tool": "-",
        "num_cases": len(df),
        "accuracy": float(df["correct"].mean()) if len(df) else 0.0,
        "mean_conf": float(df["conf"].mean()) if len(df) else 0.0,
        "num_errors": int((df["error"].astype(str) != "").sum()),
    }])

    summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"\nSaved predictions: {OUT_CSV}")
    print(f"Saved summary:     {SUMMARY_CSV}")

    wrong = df[~df["correct"]]
    if len(wrong) > 0:
        print("\nWrong cases:")
        print(wrong[["dataset", "case_id", "gt_tool", "pred_tool", "conf", "first_frame", "error"]].to_string(index=False))
    else:
        print("\nAll first-frame test cases are correctly routed.")


if __name__ == "__main__":
    main()
