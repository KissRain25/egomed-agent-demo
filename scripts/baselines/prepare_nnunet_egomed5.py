import json

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import shutil
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

IMG_DIR_NAME = "img"
LABEL_DIR_NAME = "label"

YOLO_DET_ROOT = DATA_ROOT / "yolo_det"

NNUNET_RAW = Path(f"{REPO_ROOT}/nnUNet_raw")
NNUNET_PREPROCESSED = Path(f"{REPO_ROOT}/nnUNet_preprocessed")

DATASET_ID = 501
DATASET_NAME = "EgoMed5"
NNUNET_DATASET_NAME = f"Dataset{DATASET_ID}_{DATASET_NAME}"

OUTPUT_DATASET_DIR = NNUNET_RAW / NNUNET_DATASET_NAME
OUTPUT_IMAGES_TR = OUTPUT_DATASET_DIR / "imagesTr"
OUTPUT_LABELS_TR = OUTPUT_DATASET_DIR / "labelsTr"
OUTPUT_IMAGES_TS = OUTPUT_DATASET_DIR / "imagesTs"
OUTPUT_LABELS_TS = OUTPUT_DATASET_DIR / "labelsTs"

SPLITS_FINAL_PATH = NNUNET_PREPROCESSED / NNUNET_DATASET_NAME / "splits_final.json"

TRAIN_VAL_FRAME_STRIDE = 10
TEST_FRAME_STRIDE = 1

OVERWRITE = True


# ============================================================
# Basic utilities
# ============================================================

def natural_key(path_or_name):
    name = Path(path_or_name).name
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


def read_split_cases(split_file):
    split_cases = {
        "train": [],
        "val": [],
        "test": [],
    }

    current_split = None

    with open(split_file, "r") as f:
        for line in f:
            s = line.strip()

            if not s:
                continue

            if s.startswith("[") and s.endswith("]"):
                current_split = s[1:-1]
                continue

            if current_split in split_cases:
                split_cases[current_split].append(s)

    return split_cases


# ============================================================
# Excel label mapping
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


def build_dataset_gray_mapping(excel_path):
    df = pd.read_excel(excel_path)

    if "灰度标签像素含义" not in df.columns:
        raise ValueError(f"{excel_path} missing column: 灰度标签像素含义")

    gray_mapping = {}

    for value in df["灰度标签像素含义"].dropna():
        one_mapping = parse_gray_label_mapping(value)

        for gray_value, cls_name in one_mapping.items():
            if gray_value in gray_mapping and gray_mapping[gray_value] != cls_name:
                print(
                    f"[Warning] {excel_path.name}: gray value {gray_value} conflict: "
                    f"{gray_mapping[gray_value]} vs {cls_name}. "
                    f"Using {gray_mapping[gray_value]}."
                )
                continue

            gray_mapping[gray_value] = cls_name

    return dict(sorted(gray_mapping.items(), key=lambda x: x[0]))


def build_global_label_mapping():
    """
    Build:
        dataset_gray_to_global[dataset_name][old_gray] = new_global_label_id
        global_labels = {
            "background": 0,
            "Amos_spleen": 1,
            ...
        }
    """
    global_labels = {
        "background": 0,
    }

    dataset_gray_to_global = {}
    dataset_gray_to_name = {}

    next_label_id = 1

    for dataset_name in DATASETS:
        excel_path = DATA_ROOT / dataset_name / f"{dataset_name}.xlsx"

        if not excel_path.exists():
            raise FileNotFoundError(f"Excel not found: {excel_path}")

        gray_mapping = build_dataset_gray_mapping(excel_path)
        dataset_gray_to_name[dataset_name] = gray_mapping
        dataset_gray_to_global[dataset_name] = {}

        for old_gray, cls_name in gray_mapping.items():
            global_name = f"{dataset_name}_{cls_name}"

            if global_name not in global_labels:
                global_labels[global_name] = next_label_id
                next_label_id += 1

            dataset_gray_to_global[dataset_name][old_gray] = global_labels[global_name]

    return global_labels, dataset_gray_to_global, dataset_gray_to_name


# ============================================================
# Image / label conversion
# ============================================================

def ensure_uint8_or_rgb_image(img):
    """
    nnU-Net 2D can read png images. Keep RGB/BGR image as 3 channels if present.
    For grayscale images, keep single channel.
    """
    if img is None:
        return None

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def remap_label_to_global(label, gray_to_global):
    """
    Input label:
        dataset-local gray label

    Output label:
        global label ids for merged nnU-Net dataset
    """
    out = np.zeros_like(label, dtype=np.uint16)

    for old_gray, global_id in gray_to_global.items():
        out[label == old_gray] = global_id

    return out


def save_image_for_nnunet(src_img_path, dst_img_path):
    """
    nnU-Net image filename must be:
        case_identifier_0000.png

    We save using cv2.imwrite.
    """
    img = cv2.imread(str(src_img_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Failed to read image: {src_img_path}")

    img = ensure_uint8_or_rgb_image(img)

    ok = cv2.imwrite(str(dst_img_path), img)

    if not ok:
        raise RuntimeError(f"Failed to write image: {dst_img_path}")


def save_label_for_nnunet(src_label_path, dst_label_path, gray_to_global):
    label = cv2.imread(str(src_label_path), cv2.IMREAD_GRAYSCALE)

    if label is None:
        raise RuntimeError(f"Failed to read label: {src_label_path}")

    global_label = remap_label_to_global(label, gray_to_global)

    ok = cv2.imwrite(str(dst_label_path), global_label)

    if not ok:
        raise RuntimeError(f"Failed to write label: {dst_label_path}")


def make_frame_identifier(dataset_name, case_id, img_file):
    """
    Example:
        dataset_name = Amos
        case_id = 249
        img_file.stem = 249_0340

    output:
        Amos_249_249_0340
    """
    safe_dataset = dataset_name.replace("-", "_").replace(".", "_")
    return f"{safe_dataset}_{case_id}_{img_file.stem}"


def process_one_case(
    dataset_name,
    case_id,
    split,
    gray_to_global,
    train_identifiers,
    val_identifiers,
    test_identifiers,
):
    dataset_dir = DATA_ROOT / dataset_name
    img_dir = dataset_dir / IMG_DIR_NAME / case_id
    label_dir = dataset_dir / LABEL_DIR_NAME / case_id

    if not img_dir.exists():
        print(f"[Missing image dir] {img_dir}")
        return 0

    if not label_dir.exists():
        print(f"[Missing label dir] {label_dir}")
        return 0

    image_files = collect_image_files(img_dir)
    label_files = collect_label_files(label_dir)

    if len(image_files) == 0:
        print(f"[No images] {img_dir}")
        return 0

    if len(label_files) == 0:
        print(f"[No labels] {label_dir}")
        return 0

    if len(image_files) != len(label_files):
        print(
            f"[Count mismatch] {dataset_name} case {case_id}: "
            f"images={len(image_files)}, labels={len(label_files)}. "
            f"Using min length."
        )

    n = min(len(image_files), len(label_files))
    pairs = list(zip(image_files[:n], label_files[:n]))

    if split in ["train", "val"]:
        pairs = pairs[::TRAIN_VAL_FRAME_STRIDE]
        out_img_dir = OUTPUT_IMAGES_TR
        out_lbl_dir = OUTPUT_LABELS_TR
    elif split == "test":
        pairs = pairs[::TEST_FRAME_STRIDE]
        out_img_dir = OUTPUT_IMAGES_TS
        out_lbl_dir = OUTPUT_LABELS_TS
    else:
        raise ValueError(f"Unsupported split: {split}")

    count = 0

    for img_file, label_file in pairs:
        case_identifier = make_frame_identifier(dataset_name, case_id, img_file)

        # nnU-Net image file must have modality suffix _0000
        out_img_path = out_img_dir / f"{case_identifier}_0000.png"
        out_lbl_path = out_lbl_dir / f"{case_identifier}.png"

        save_image_for_nnunet(img_file, out_img_path)
        save_label_for_nnunet(label_file, out_lbl_path, gray_to_global)

        if split == "train":
            train_identifiers.append(case_identifier)
        elif split == "val":
            val_identifiers.append(case_identifier)
        elif split == "test":
            test_identifiers.append(case_identifier)

        count += 1

    return count


# ============================================================
# dataset.json and splits_final.json
# ============================================================

def write_dataset_json(global_labels):
    """
    nnU-Net v2 dataset.json.

    For channel_names:
        We use RGB because many frames are jpg/color; grayscale images saved via cv2 may still be accepted,
        but if this causes issue, convert all inputs to RGB explicitly.
    """
    labels_for_json = {}
    for name, label_id in global_labels.items():
        labels_for_json[name] = int(label_id)

    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B",
        },
        "labels": labels_for_json,
        "numTraining": len(list(OUTPUT_LABELS_TR.glob("*.png"))),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }

    dataset_json_path = OUTPUT_DATASET_DIR / "dataset.json"

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"[Saved] {dataset_json_path}")


def write_splits_final(train_identifiers, val_identifiers):
    SPLITS_FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    splits = [
        {
            "train": sorted(train_identifiers),
            "val": sorted(val_identifiers),
        }
    ]

    with open(SPLITS_FINAL_PATH, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"[Saved] {SPLITS_FINAL_PATH}")


def write_global_label_mapping(global_labels, dataset_gray_to_global, dataset_gray_to_name):
    mapping_path = OUTPUT_DATASET_DIR / "global_label_mapping.json"

    data = {
        "global_labels": global_labels,
        "dataset_gray_to_global": {
            dataset_name: {
                str(k): int(v)
                for k, v in mapping.items()
            }
            for dataset_name, mapping in dataset_gray_to_global.items()
        },
        "dataset_gray_to_name": {
            dataset_name: {
                str(k): v
                for k, v in mapping.items()
            }
            for dataset_name, mapping in dataset_gray_to_name.items()
        },
    }

    with open(mapping_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[Saved] {mapping_path}")


# ============================================================
# Main
# ============================================================

def main():
    if OVERWRITE and OUTPUT_DATASET_DIR.exists():
        print(f"[Remove old dataset] {OUTPUT_DATASET_DIR}")
        shutil.rmtree(OUTPUT_DATASET_DIR)

    OUTPUT_IMAGES_TR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS_TR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES_TS.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS_TS.mkdir(parents=True, exist_ok=True)

    global_labels, dataset_gray_to_global, dataset_gray_to_name = build_global_label_mapping()

    print("\nGlobal labels:")
    for name, label_id in sorted(global_labels.items(), key=lambda x: x[1]):
        print(f"  {label_id}: {name}")

    train_identifiers = []
    val_identifiers = []
    test_identifiers = []

    total_counts = {}

    for dataset_name in DATASETS:
        split_file = YOLO_DET_ROOT / dataset_name / "split_cases.txt"

        if not split_file.exists():
            raise FileNotFoundError(f"split_cases.txt not found: {split_file}")

        split_cases = read_split_cases(split_file)
        gray_to_global = dataset_gray_to_global[dataset_name]

        print("\n" + "=" * 100)
        print(f"Processing dataset: {dataset_name}")
        print(f"Split file: {split_file}")
        print(f"Train cases: {len(split_cases['train'])}")
        print(f"Val cases:   {len(split_cases['val'])}")
        print(f"Test cases:  {len(split_cases['test'])}")
        print("=" * 100)

        total_counts[dataset_name] = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        for split in ["train", "val", "test"]:
            for case_id in tqdm(split_cases[split], desc=f"{dataset_name}-{split}"):
                case_id = normalize_case_id(case_id)

                count = process_one_case(
                    dataset_name=dataset_name,
                    case_id=case_id,
                    split=split,
                    gray_to_global=gray_to_global,
                    train_identifiers=train_identifiers,
                    val_identifiers=val_identifiers,
                    test_identifiers=test_identifiers,
                )

                total_counts[dataset_name][split] += count

        print(f"[{dataset_name}] counts: {total_counts[dataset_name]}")

    write_dataset_json(global_labels)
    write_splits_final(train_identifiers, val_identifiers)
    write_global_label_mapping(global_labels, dataset_gray_to_global, dataset_gray_to_name)

    print("\nDone preparing nnU-Net dataset.")
    print(f"Dataset dir: {OUTPUT_DATASET_DIR}")
    print(f"imagesTr: {len(list(OUTPUT_IMAGES_TR.glob('*.png')))}")
    print(f"labelsTr: {len(list(OUTPUT_LABELS_TR.glob('*.png')))}")
    print(f"imagesTs: {len(list(OUTPUT_IMAGES_TS.glob('*.png')))}")
    print(f"labelsTs: {len(list(OUTPUT_LABELS_TS.glob('*.png')))}")

    print("\nCounts by dataset:")
    for dataset_name, counts in total_counts.items():
        print(f"  {dataset_name}: {counts}")

    print("\nNext commands:")
    print("export nnUNet_raw=$EGOMED_ROOT/nnUNet_raw")
    print("export nnUNet_preprocessed=$EGOMED_ROOT/nnUNet_preprocessed")
    print("export nnUNet_results=$EGOMED_ROOT/nnUNet_results")
    print(f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print(f"CUDA_VISIBLE_DEVICES=1 nnUNetv2_train {DATASET_ID} 2d 0")


if __name__ == "__main__":
    main()