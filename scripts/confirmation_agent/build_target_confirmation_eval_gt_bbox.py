#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


DATA_ROOT = Path(f"{REPO_ROOT}/data")
INPUT_CSV = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_full.csv")

OUT_ALL = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_gt_bbox_all.csv")
OUT_VALID = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_gt_bbox_valid.csv")
OUT_STATS = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_gt_bbox_stats.csv")


def normalize_case_id(case_id):
    s = str(case_id).strip()
    return str(int(float(s))) if s.replace(".", "", 1).isdigit() else s


def normalize_text(x):
    x = "" if pd.isna(x) else str(x)
    x = x.strip().lower().replace("_", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def split_candidates(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]


def natural_key(p):
    nums = re.findall(r"\d+", Path(str(p)).name)
    return [int(x) for x in nums] if nums else [Path(str(p)).name]


def collect_files(folder, exts):
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files, key=natural_key)


def resolve_frame_paths(dataset, case_id, frame_idx):
    case_id = normalize_case_id(case_id)

    img_dir = DATA_ROOT / dataset / "img" / case_id
    label_dir = DATA_ROOT / dataset / "label" / case_id

    image_files = collect_files(img_dir, ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"])
    label_files = collect_files(label_dir, ["*.png"])

    if int(frame_idx) >= len(image_files):
        raise IndexError(f"frame_idx={frame_idx} exceeds image files in {img_dir}, n={len(image_files)}")

    if int(frame_idx) >= len(label_files):
        raise IndexError(f"frame_idx={frame_idx} exceeds label files in {label_dir}, n={len(label_files)}")

    return image_files[int(frame_idx)], label_files[int(frame_idx)]


def parse_gray_label_mapping(mapping_str):
    mapping = {}
    if pd.isna(mapping_str):
        return mapping

    s = (
        str(mapping_str)
        .replace("，", ",")
        .replace("；", ",")
        .replace(";", ",")
        .replace("：", ":")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", ",")
    )

    for item in s.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue

        gray_value, cls_name = item.split(":", 1)
        gray_value = int(gray_value.strip())
        cls_name = cls_name.strip()

        if gray_value == 0 or cls_name.lower() in ["background", "bg", "back ground"]:
            continue

        mapping[cls_name] = gray_value

    return mapping


_GRAY_CACHE = {}


def get_gray_mapping(dataset):
    if dataset in _GRAY_CACHE:
        return _GRAY_CACHE[dataset]

    xlsx = DATA_ROOT / dataset / f"{dataset}.xlsx"
    df = pd.read_excel(xlsx)

    mapping = {}
    for v in df["灰度标签像素含义"].dropna():
        mapping.update(parse_gray_label_mapping(v))

    _GRAY_CACHE[dataset] = mapping
    return mapping


def find_gray_value(mapping, target):
    target_norm = normalize_text(target)

    for k, v in mapping.items():
        if normalize_text(k) == target_norm:
            return int(v), k

    return None, None


def gt_bbox_from_label(label, gray_value):
    mask = label == int(gray_value)
    if not mask.any():
        return None, 0

    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    area = int(mask.sum())
    return bbox, area


def bbox_position_text(bbox, image_shape):
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]

    cx = (x1 + x2) / 2.0 / max(w, 1)
    cy = (y1 + y2) / 2.0 / max(h, 1)

    horiz = "left" if cx < 0.33 else ("right" if cx > 0.67 else "center")
    vert = "upper" if cy < 0.33 else ("lower" if cy > 0.67 else "middle")

    return f"{vert}-{horiz}"


def bbox_size_text(area, image_shape):
    h, w = image_shape[:2]
    ratio = area / max(h * w, 1)

    if ratio < 0.005:
        return "small"
    if ratio < 0.03:
        return "medium"
    return "large"


def canonical_match(a, b):
    return normalize_text(a) == normalize_text(b)


def infer_instruction_type(instruction):
    x = normalize_text(instruction)
    spatial_words = [
        "left side", "right side", "upper", "lower", "top", "bottom",
        "center", "middle", "near", "larger", "smaller",
        "on the left", "on the right"
    ]
    if any(w in x for w in spatial_words):
        return "spatial"
    return "semantic"


def build_one(row, idx):
    dataset = str(row["dataset"])
    case_id = normalize_case_id(row["case_id"])
    frame_idx = int(row["frame_idx"])

    image_path, label_path = resolve_frame_paths(dataset, case_id, frame_idx)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(image_path)
    if label is None:
        raise FileNotFoundError(label_path)

    mapping = get_gray_mapping(dataset)
    candidates = split_candidates(row["candidate_targets"])

    objects = []
    missing_candidates = []

    for cand in candidates:
        gray_value, canonical_name = find_gray_value(mapping, cand)

        if gray_value is None:
            missing_candidates.append({
                "category": cand,
                "reason": "class_not_in_label_mapping",
            })
            continue

        bbox, area = gt_bbox_from_label(label, gray_value)

        if bbox is None:
            missing_candidates.append({
                "category": cand,
                "reason": "not_visible_in_this_frame",
            })
            continue

        obj = {
            "id": f"o{len(objects) + 1}",
            "category": cand,
            "bbox": bbox,
            "position": bbox_position_text(bbox, image.shape),
            "size": bbox_size_text(area, image.shape),
            "area_pixels": area,
            "source": "gt_bbox",
        }
        objects.append(obj)

    intended = str(row["intended_target"])
    intended_id = ""

    for obj in objects:
        if canonical_match(obj["category"], intended):
            intended_id = obj["id"]
            break

    gt_state = normalize_text(row["gt_state"])

    is_valid = True
    invalid_reason = ""

    if not intended_id:
        is_valid = False
        invalid_reason = "intended_target_not_visible"

    elif gt_state == "ambiguous" and len(objects) < 2:
        is_valid = False
        invalid_reason = "ambiguous_but_less_than_two_visible_candidates"

    elif len(objects) == 0:
        is_valid = False
        invalid_reason = "no_visible_candidates"

    out = row.to_dict()
    out.update({
        "sample_id": str(idx),
        "image_path": str(image_path),
        "label_path": str(label_path),
        "candidate_objects": json.dumps(objects, ensure_ascii=False),
        "visible_candidate_targets": ";".join([o["category"] for o in objects]),
        "num_visible_candidates": len(objects),
        "intended_target_id": intended_id,
        "instruction_type": infer_instruction_type(row["instruction"]),
        "is_valid": bool(is_valid),
        "invalid_reason": invalid_reason,
        "missing_candidates": json.dumps(missing_candidates, ensure_ascii=False),
    })

    return out


def main():
    print("=" * 100)
    print("Build GT-bbox target confirmation eval CSV")
    print("=" * 100)
    print("Input:", INPUT_CSV)
    print("Output all:", OUT_ALL)
    print("Output valid:", OUT_VALID)

    df = pd.read_csv(INPUT_CSV)

    rows = []
    for idx, row in df.iterrows():
        try:
            rows.append(build_one(row, idx))
        except Exception as e:
            out = row.to_dict()
            out.update({
                "sample_id": str(idx),
                "candidate_objects": "[]",
                "visible_candidate_targets": "",
                "num_visible_candidates": 0,
                "intended_target_id": "",
                "instruction_type": infer_instruction_type(row.get("instruction", "")),
                "is_valid": False,
                "invalid_reason": f"exception: {repr(e)}",
                "missing_candidates": "[]",
            })
            rows.append(out)

    out_df = pd.DataFrame(rows)
    valid_df = out_df[out_df["is_valid"].astype(bool)].copy()

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_ALL, index=False)
    valid_df.to_csv(OUT_VALID, index=False)

    stats = []

    stats.append({"metric": "num_all_samples", "value": len(out_df)})
    stats.append({"metric": "num_valid_samples", "value": len(valid_df)})
    stats.append({"metric": "num_invalid_samples", "value": len(out_df) - len(valid_df)})

    for k, v in out_df["invalid_reason"].fillna("").value_counts().items():
        key = k if k else "valid"
        stats.append({"metric": f"invalid_reason::{key}", "value": int(v)})

    for k, v in valid_df["gt_state"].fillna("").value_counts().items():
        stats.append({"metric": f"valid_gt_state::{k}", "value": int(v)})

    for k, v in valid_df["instruction_type"].fillna("").value_counts().items():
        stats.append({"metric": f"valid_instruction_type::{k}", "value": int(v)})

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT_STATS, index=False)

    print("=" * 100)
    print("[DONE]")
    print("All:", OUT_ALL, out_df.shape)
    print("Valid:", OUT_VALID, valid_df.shape)
    print("Stats:", OUT_STATS)
    print("\nStats:")
    print(stats_df.to_string(index=False))

    print("\nPreview valid:")
    cols = [
        "sample_id", "dataset", "case_id", "frame_idx",
        "instruction", "candidate_objects", "intended_target",
        "intended_target_id", "gt_state", "is_valid"
    ]
    print(valid_df.head(5)[cols].to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
