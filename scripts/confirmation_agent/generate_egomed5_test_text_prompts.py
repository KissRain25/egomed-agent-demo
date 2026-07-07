import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

DATA_ROOT = Path(f"{REPO_ROOT}/data")

DATASETS = [
    "Amos",
    "CAMUS",
    "ACDC",
    "Montgomery-County-CXR-Set",
    "PolypGen2021_MultiCenterData_v3",
]

# Amos only uses these 5 targets.
FOCUS_TARGETS_BY_DATASET = {
    "Amos": [
        "liver",
        "right kidney",
        "left kidney",
        "spleen",
        "stomach",
    ],
}

# Ambiguous prompts are mapped to one fixed target class.
# They are NOT union masks.
# These prompts are activated when the mapped target first appears in the case.
AMBIGUOUS_PROMPT_MAPPING = {
    "Amos": [
        {
            "prompt": "segment the kidney",
            "target_class": "left kidney",
        },
    ],
    "Montgomery-County-CXR-Set": [
        {
            "prompt": "segment the lung",
            "target_class": "left lung",
        },
    ],
}

EXACT_PROMPT_TEMPLATE = "segment the {class_name}"

OUTPUT_DIR = DATA_ROOT / "text_prompt_eval"
OUTPUT_CSV = OUTPUT_DIR / "egomed5_test_text_prompts_online_schedule.csv"
OUTPUT_XLSX = OUTPUT_DIR / "egomed5_test_text_prompts_online_schedule.xlsx"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "egomed5_test_text_prompts_online_schedule_summary.csv"
OUTPUT_SCHEDULE_CSV = OUTPUT_DIR / "egomed5_test_prompt_schedule.csv"


# ============================================================
# Basic utilities
# ============================================================

def natural_key(path_or_name):
    name = Path(str(path_or_name)).name
    nums = re.findall(r"\d+", name)
    return [int(x) for x in nums] if nums else [name]


def normalize_case_id(case_id):
    case_id = str(case_id).strip()
    return str(int(float(case_id))) if case_id.replace(".", "", 1).isdigit() else case_id


def collect_image_files(case_img_dir):
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        files.extend(case_img_dir.glob(ext))
    return sorted(files, key=natural_key)


def collect_label_files(case_label_dir):
    files = []
    for ext in ["*.png"]:
        files.extend(case_label_dir.glob(ext))
    return sorted(files, key=natural_key)


def read_split_cases(split_file, target_split="test"):
    cases = []
    current_split = None

    with open(split_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("[") and s.endswith("]"):
                current_split = s[1:-1]
                continue

            if current_split == target_split:
                cases.append(normalize_case_id(s))

    return cases


def read_label(path):
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"Cannot read label: {path}")
    return label


def mask_to_bbox(mask):
    mask = mask.astype(bool)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_to_str(box):
    if box is None:
        return ""
    return ",".join([f"{float(x):.2f}" for x in box])


# ============================================================
# Excel label mapping utilities
# ============================================================

def parse_gray_label_mapping(mapping_str):
    mapping = {}

    if pd.isna(mapping_str):
        return mapping

    mapping_str = str(mapping_str)
    mapping_str = (
        mapping_str
        .replace("，", ",")
        .replace("；", ",")
        .replace(";", ",")
        .replace("：", ":")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", ",")
    )

    for item in mapping_str.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue

        gray_value, cls_name = item.split(":", 1)
        gray_value = int(gray_value.strip())
        cls_name = cls_name.strip()

        if gray_value == 0:
            continue

        if cls_name.lower() in ["background", "bg", "back ground"]:
            continue

        mapping[gray_value] = cls_name

    return mapping


def build_dataset_gray_mapping(excel_file):
    df = pd.read_excel(excel_file)

    if "灰度标签像素含义" not in df.columns:
        raise ValueError(f"Missing column 灰度标签像素含义 in {excel_file}")

    gray_mapping = {}

    for value in df["灰度标签像素含义"].dropna():
        one_mapping = parse_gray_label_mapping(value)

        for gray_value, cls_name in one_mapping.items():
            if gray_value in gray_mapping and gray_mapping[gray_value] != cls_name:
                print(
                    f"[Warning] {excel_file.name}: gray value {gray_value} conflict: "
                    f"{gray_mapping[gray_value]} vs {cls_name}. "
                    f"Using {gray_mapping[gray_value]}."
                )
                continue

            gray_mapping[gray_value] = cls_name

    return dict(sorted(gray_mapping.items(), key=lambda x: x[0]))


def filter_gray_mapping(dataset_name, gray_mapping):
    focus_targets = FOCUS_TARGETS_BY_DATASET.get(dataset_name, None)

    if focus_targets is None:
        return gray_mapping

    focus_set = set(focus_targets)

    filtered = {
        gray_value: class_name
        for gray_value, class_name in gray_mapping.items()
        if class_name in focus_set
    }

    missing = focus_set - set(filtered.values())
    if missing:
        raise ValueError(
            f"{dataset_name}: focus targets not found in Excel gray mapping: {missing}"
        )

    return filtered


def build_target_mappings(dataset_name, excel_file):
    full_gray_mapping = build_dataset_gray_mapping(excel_file)
    gray_to_class = filter_gray_mapping(dataset_name, full_gray_mapping)

    class_to_gray = {
        class_name: gray_value
        for gray_value, class_name in gray_to_class.items()
    }

    return gray_to_class, class_to_gray


# ============================================================
# Online prompt schedule
# ============================================================

def scan_case_first_appearance(label_files, gray_to_class):
    """
    Scan a case and find:
      - first appearance frame for each target class
      - per-frame GT info for each target class

    Returns:
      frame_infos:
        list of dict[class_name] = {
            gray_value,
            gt_exists,
            gt_bbox,
            gt_area
        }

      first_appearance:
        dict[class_name] = first_frame_idx
    """
    frame_infos = []
    first_appearance = {}

    for frame_idx, label_path in enumerate(label_files):
        label = read_label(label_path)
        frame_info = {}

        for gray_value, class_name in gray_to_class.items():
            gt_mask = label == int(gray_value)
            gt_bbox = mask_to_bbox(gt_mask)
            gt_exists = gt_bbox is not None
            gt_area = int(gt_mask.sum()) if gt_exists else 0

            frame_info[class_name] = {
                "gray_value": int(gray_value),
                "gt_exists": bool(gt_exists),
                "gt_bbox": gt_bbox,
                "gt_area": gt_area,
            }

            if gt_exists and class_name not in first_appearance:
                first_appearance[class_name] = int(frame_idx)

        frame_infos.append(frame_info)

    return frame_infos, first_appearance


def build_case_prompt_schedule(dataset_name, first_appearance, class_to_gray):
    """
    Build an online case-level prompt schedule.

    Exact prompts:
      Activated when their target class first appears.

    Ambiguous prompts:
      Activated when the predefined mapped target class first appears.
      Example:
        "segment the kidney" -> "left kidney"
    """
    prompts = []

    # Exact prompts.
    for class_name, activate_frame in sorted(first_appearance.items(), key=lambda x: (x[1], x[0])):
        gray_value = int(class_to_gray[class_name])
        prompts.append({
            "prompt": EXACT_PROMPT_TEMPLATE.format(class_name=class_name),
            "prompt_type": "exact",
            "target_mode": "single",
            "target_class": class_name,
            "target_gray_value": gray_value,
            "target_classes": class_name,
            "target_gray_values": str(gray_value),
            "prompt_activate_frame": int(activate_frame),
            "activation_reason": f"first_visible:{class_name}",
        })

    # Ambiguous prompts.
    for item in AMBIGUOUS_PROMPT_MAPPING.get(dataset_name, []):
        prompt = item["prompt"]
        target_class = item["target_class"]

        if target_class not in class_to_gray:
            print(
                f"[Warning] {dataset_name}: ambiguous target class '{target_class}' "
                f"not found in evaluated class mapping. Skip prompt '{prompt}'."
            )
            continue

        if target_class not in first_appearance:
            # Target never appears in this case, so this prompt is never activated.
            continue

        gray_value = int(class_to_gray[target_class])
        activate_frame = int(first_appearance[target_class])

        prompts.append({
            "prompt": prompt,
            "prompt_type": "ambiguous",
            "target_mode": "single",
            "target_class": target_class,
            "target_gray_value": gray_value,
            "target_classes": target_class,
            "target_gray_values": str(gray_value),
            "prompt_activate_frame": activate_frame,
            "activation_reason": f"first_visible:{target_class}",
        })

    # Stable sorting:
    # 1. activation frame
    # 2. exact before ambiguous if same frame
    # 3. prompt text
    prompt_type_rank = {
        "exact": 0,
        "ambiguous": 1,
    }

    prompts = sorted(
        prompts,
        key=lambda p: (
            p["prompt_activate_frame"],
            prompt_type_rank.get(p["prompt_type"], 99),
            p["prompt"],
        )
    )

    # Add case_prompt_id.
    for i, p in enumerate(prompts):
        p["case_prompt_id"] = i

    return prompts


def generate_rows_for_case(
    dataset_name,
    case_id,
    image_files,
    label_files,
    frame_infos,
    case_prompts,
):
    """
    For each frame, use all prompts activated up to that frame.
    If the target is absent in the current frame, gt_exists=False, gt_bbox="", gt_area=0.
    """
    rows = []

    n = min(len(image_files), len(label_files), len(frame_infos))

    for frame_idx in range(n):
        image_path = image_files[frame_idx]
        label_path = label_files[frame_idx]
        frame_info = frame_infos[frame_idx]

        active_prompts = [
            p for p in case_prompts
            if int(p["prompt_activate_frame"]) <= frame_idx
        ]

        for p in active_prompts:
            target_class = p["target_class"]

            if target_class not in frame_info:
                gt_exists = False
                gt_bbox = None
                gt_area = 0
            else:
                info = frame_info[target_class]
                gt_exists = bool(info["gt_exists"])
                gt_bbox = info["gt_bbox"]
                gt_area = int(info["gt_area"])

            rows.append({
                "dataset": dataset_name,
                "split": "test",
                "case_id": case_id,
                "frame_idx": int(frame_idx),
                "frame_name": image_path.name,
                "image_path": str(image_path),
                "label_path": str(label_path),

                "prompt": p["prompt"],
                "prompt_type": p["prompt_type"],
                "target_mode": p["target_mode"],
                "target_class": p["target_class"],
                "target_gray_value": int(p["target_gray_value"]),
                "target_classes": p["target_classes"],
                "target_gray_values": p["target_gray_values"],

                "case_prompt_id": int(p["case_prompt_id"]),
                "prompt_activate_frame": int(p["prompt_activate_frame"]),
                "activation_reason": p["activation_reason"],

                "gt_exists": bool(gt_exists),
                "gt_bbox": bbox_to_str(gt_bbox),
                "gt_area": int(gt_area),
            })

    return rows


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset_prompts(dataset_name):
    dataset_root = DATA_ROOT / dataset_name
    img_root = dataset_root / "img"
    label_root = dataset_root / "label"
    excel_file = dataset_root / f"{dataset_name}.xlsx"
    split_file = DATA_ROOT / "yolo_det" / dataset_name / "split_cases.txt"

    print("\n" + "=" * 120)
    print(f"Generating ONLINE-SCHEDULE test text prompts for dataset: {dataset_name}")
    print("=" * 120)
    print(f"Image root: {img_root}")
    print(f"Label root: {label_root}")
    print(f"Excel file: {excel_file}")
    print(f"Split file: {split_file}")
    print("=" * 120)

    if not img_root.exists():
        raise FileNotFoundError(f"Image root not found: {img_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Label root not found: {label_root}")
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    if not split_file.exists():
        raise FileNotFoundError(f"split_cases.txt not found: {split_file}")

    gray_to_class, class_to_gray = build_target_mappings(dataset_name, excel_file)

    print("\nTarget classes:")
    for gray_value, class_name in gray_to_class.items():
        print(f"  gray_value={gray_value}, class_name={class_name}")

    if dataset_name in AMBIGUOUS_PROMPT_MAPPING:
        print("\nAmbiguous prompt mapping:")
        for item in AMBIGUOUS_PROMPT_MAPPING[dataset_name]:
            print(f"  prompt='{item['prompt']}' -> target_class='{item['target_class']}'")

    test_cases = read_split_cases(split_file, target_split="test")
    print(f"\nNumber of test cases: {len(test_cases)}")
    print(f"First 10 test cases: {test_cases[:10]}")

    all_rows = []
    schedule_rows = []

    for case_idx, case_id in enumerate(test_cases):
        print("\n" + "-" * 100)
        print(f"[{dataset_name}] [{case_idx + 1}/{len(test_cases)}] case {case_id}")
        print("-" * 100)

        case_img_dir = img_root / case_id
        case_label_dir = label_root / case_id

        if not case_img_dir.exists():
            print(f"[Skip] Missing image dir: {case_img_dir}")
            continue

        if not case_label_dir.exists():
            print(f"[Skip] Missing label dir: {case_label_dir}")
            continue

        image_files = collect_image_files(case_img_dir)
        label_files = collect_label_files(case_label_dir)

        if len(image_files) == 0:
            print(f"[Skip] No images in {case_img_dir}")
            continue

        if len(label_files) == 0:
            print(f"[Skip] No labels in {case_label_dir}")
            continue

        if len(image_files) != len(label_files):
            print(
                f"[Warning] Count mismatch {dataset_name} case {case_id}: "
                f"images={len(image_files)}, labels={len(label_files)}. Using min length."
            )

        n = min(len(image_files), len(label_files))
        image_files = image_files[:n]
        label_files = label_files[:n]

        frame_infos, first_appearance = scan_case_first_appearance(
            label_files=label_files,
            gray_to_class=gray_to_class,
        )

        if len(first_appearance) == 0:
            print(f"[Warning] No target appears in this case: {dataset_name} case {case_id}")
            continue

        case_prompts = build_case_prompt_schedule(
            dataset_name=dataset_name,
            first_appearance=first_appearance,
            class_to_gray=class_to_gray,
        )

        print("Case prompt schedule:")
        for p in case_prompts:
            print(
                f"  frame={p['prompt_activate_frame']:>4}, "
                f"type={p['prompt_type']:<9}, "
                f"prompt='{p['prompt']}', "
                f"target='{p['target_class']}'"
            )

            schedule_rows.append({
                "dataset": dataset_name,
                "split": "test",
                "case_id": case_id,
                "case_prompt_id": int(p["case_prompt_id"]),
                "prompt": p["prompt"],
                "prompt_type": p["prompt_type"],
                "target_class": p["target_class"],
                "target_gray_value": int(p["target_gray_value"]),
                "target_mode": p["target_mode"],
                "target_classes": p["target_classes"],
                "target_gray_values": p["target_gray_values"],
                "prompt_activate_frame": int(p["prompt_activate_frame"]),
                "activation_reason": p["activation_reason"],
                "num_case_frames": int(n),
            })

        case_rows = generate_rows_for_case(
            dataset_name=dataset_name,
            case_id=case_id,
            image_files=image_files,
            label_files=label_files,
            frame_infos=frame_infos,
            case_prompts=case_prompts,
        )

        all_rows.extend(case_rows)

    prompt_df = pd.DataFrame(all_rows)
    schedule_df = pd.DataFrame(schedule_rows)

    return prompt_df, schedule_df


def _mean_present_area(x):
    vals = [v for v in x if v > 0]
    return float(np.mean(vals)) if vals else np.nan


def _min_present_area(x):
    vals = [v for v in x if v > 0]
    return int(np.min(vals)) if vals else 0


def build_summary(prompt_df, schedule_df):
    if len(prompt_df) == 0:
        return pd.DataFrame()

    summary = (
        prompt_df.groupby(
            ["dataset", "prompt_type", "prompt", "target_class", "target_gray_value"],
            dropna=False,
        )
        .agg(
            num_prompt_frame_samples=("prompt", "count"),
            num_gt_exists_frames=("gt_exists", "sum"),
            num_cases=("case_id", "nunique"),
            mean_gt_area_present=("gt_area", _mean_present_area),
            min_gt_area_present=("gt_area", _min_present_area),
            max_gt_area_present=("gt_area", "max"),
        )
        .reset_index()
        .sort_values(["dataset", "prompt_type", "target_gray_value", "prompt"])
    )

    if len(schedule_df) > 0:
        schedule_summary = (
            schedule_df.groupby(
                ["dataset", "prompt_type", "prompt", "target_class", "target_gray_value"],
                dropna=False,
            )
            .agg(
                num_case_prompt_activations=("case_id", "count"),
                mean_activate_frame=("prompt_activate_frame", "mean"),
                min_activate_frame=("prompt_activate_frame", "min"),
                max_activate_frame=("prompt_activate_frame", "max"),
            )
            .reset_index()
        )

        summary = summary.merge(
            schedule_summary,
            on=["dataset", "prompt_type", "prompt", "target_class", "target_gray_value"],
            how="left",
        )

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("Generate EgoMed5 TEST text prompt CSV with ONLINE prompt schedule")
    print("=" * 120)
    print(f"Datasets: {DATASETS}")
    print(f"Exact prompt template: {EXACT_PROMPT_TEMPLATE}")
    print("Rule:")
    print("  1. Scan each test case and find the first appearance frame of every target.")
    print("  2. Activate exact prompt when its target first appears.")
    print("  3. Activate ambiguous prompt when its predefined target first appears.")
    print("  4. For each subsequent frame, keep all previously activated prompts.")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Output XLSX: {OUTPUT_XLSX}")
    print("=" * 120)

    all_prompt_dfs = []
    all_schedule_dfs = []

    for dataset_name in DATASETS:
        prompt_df, schedule_df = generate_dataset_prompts(dataset_name)
        all_prompt_dfs.append(prompt_df)
        all_schedule_dfs.append(schedule_df)

        per_dataset_prompt_csv = OUTPUT_DIR / f"{dataset_name}_test_text_prompts_online_schedule.csv"
        per_dataset_schedule_csv = OUTPUT_DIR / f"{dataset_name}_test_prompt_schedule.csv"

        prompt_df.to_csv(per_dataset_prompt_csv, index=False)
        schedule_df.to_csv(per_dataset_schedule_csv, index=False)

        print(f"\n[{dataset_name}] Saved per-dataset prompt CSV:   {per_dataset_prompt_csv}")
        print(f"[{dataset_name}] Saved per-dataset schedule CSV: {per_dataset_schedule_csv}")
        print(f"[{dataset_name}] Number of prompt-frame samples: {len(prompt_df)}")
        print(f"[{dataset_name}] Number of case prompt activations: {len(schedule_df)}")

    all_prompt_df = pd.concat(all_prompt_dfs, ignore_index=True) if all_prompt_dfs else pd.DataFrame()
    all_schedule_df = pd.concat(all_schedule_dfs, ignore_index=True) if all_schedule_dfs else pd.DataFrame()
    summary_df = build_summary(all_prompt_df, all_schedule_df)

    all_prompt_df.to_csv(OUTPUT_CSV, index=False)
    all_schedule_df.to_csv(OUTPUT_SCHEDULE_CSV, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        all_prompt_df.to_excel(writer, sheet_name="prompt_frame_samples", index=False)
        all_schedule_df.to_excel(writer, sheet_name="case_prompt_schedule", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print("\n" + "=" * 120)
    print("Done.")
    print(f"Saved prompt CSV:    {OUTPUT_CSV}")
    print(f"Saved schedule CSV:  {OUTPUT_SCHEDULE_CSV}")
    print(f"Saved prompt XLSX:   {OUTPUT_XLSX}")
    print(f"Saved summary CSV:   {OUTPUT_SUMMARY_CSV}")
    print("=" * 120)

    if len(summary_df) > 0:
        print("\nSummary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
