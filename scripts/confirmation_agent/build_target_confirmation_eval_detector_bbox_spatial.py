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
from ultralytics import YOLO


DATA_ROOT = Path(f"{REPO_ROOT}/data")
INPUT_CSV = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_full.csv")
OUT_CSV = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_detector_bbox_spatial.csv")
OUT_DEBUG = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_detector_bbox_spatial_debug.csv")

WEIGHTS = {
    "Amos": f"{REPO_ROOT}/runs/yolo26_det/Amos_yolo26m_imgsz1024/weights/best.pt",
    "ACDC": f"{REPO_ROOT}/runs/yolo26_det/ACDC_yolo26m_imgsz1024/weights/best.pt",
    "CAMUS": f"{REPO_ROOT}/runs/yolo26_det/CAMUS_yolo26m_imgsz1024/weights/best.pt",
    "Montgomery-County-CXR-Set": f"{REPO_ROOT}/runs/yolo26_det/Montgomery-County-CXR-Set_yolo26m_imgsz1024/weights/best.pt",
    "PolypGen2021_MultiCenterData_v3": f"{REPO_ROOT}/runs/yolo26_det/PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024/weights/best.pt",
}

CONF_THRES = 0.10
IMGSZ = 1024


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

    imgs = collect_files(img_dir, ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"])
    labels = collect_files(label_dir, ["*.png"])

    if int(frame_idx) >= len(imgs):
        raise IndexError(f"frame_idx={frame_idx} exceeds images in {img_dir}, n={len(imgs)}")

    return imgs[int(frame_idx)], labels[int(frame_idx)]


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


def gt_bbox_from_label(label_path, dataset, target):
    mapping = get_gray_mapping(dataset)
    if target not in mapping:
        return None

    gray = int(mapping[target])
    lab = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if lab is None:
        return None

    mask = lab == gray
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_position_text(bbox, image_shape):
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    cx = (x1 + x2) / 2.0 / max(w, 1)
    cy = (y1 + y2) / 2.0 / max(h, 1)

    horiz = "left" if cx < 0.33 else ("right" if cx > 0.67 else "center")
    vert = "upper" if cy < 0.33 else ("lower" if cy > 0.67 else "middle")

    return f"{vert}-{horiz}"


def canonical_match(a, b):
    return normalize_text(a) == normalize_text(b)


def infer_instruction_type(instruction):
    x = normalize_text(instruction)
    spatial_words = [
        "left side", "right side", "upper", "lower", "top", "bottom",
        "center", "middle", "near", "larger", "smaller", "on the left", "on the right"
    ]
    if any(w in x for w in spatial_words):
        return "spatial"
    return "semantic"


def run_detector(model, image_path):
    results = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        conf=CONF_THRES,
        verbose=False,
    )
    res = results[0]
    names = model.names

    dets = []
    if res.boxes is None or len(res.boxes) == 0:
        return dets

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, clss):
        dets.append({
            "category": str(names[int(cls_id)]),
            "bbox": [int(round(float(v))) for v in box.tolist()],
            "confidence": float(conf),
            "source": "detector",
        })

    return dets


def build_objects_for_sample(row, models):
    dataset = str(row["dataset"])
    case_id = normalize_case_id(row["case_id"])
    frame_idx = int(row["frame_idx"])

    image_path, label_path = resolve_frame_paths(dataset, case_id, frame_idx)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    candidates = split_candidates(row["candidate_targets"])
    dets = run_detector(models[dataset], image_path)

    objects = []
    debug_items = []

    for i, cand in enumerate(candidates, 1):
        matched = [d for d in dets if canonical_match(d["category"], cand)]

        if matched:
            # choose highest confidence detection for this candidate class
            d = sorted(matched, key=lambda x: x["confidence"], reverse=True)[0]
            bbox = d["bbox"]
            conf = d["confidence"]
            source = "detector"
        else:
            # fallback to GT bbox to keep candidate object complete
            bbox = gt_bbox_from_label(label_path, dataset, cand)
            conf = 1.0 if bbox is not None else 0.0
            source = "gt_fallback" if bbox is not None else "missing"

        if bbox is None:
            # use a dummy full-image bbox only as last resort
            h, w = image.shape[:2]
            bbox = [0, 0, w - 1, h - 1]
            source = "dummy_full_image"

        obj = {
            "id": f"o{i}",
            "category": cand,
            "bbox": bbox,
            "confidence": round(float(conf), 4),
            "position": bbox_position_text(bbox, image.shape),
            "source": source,
        }
        objects.append(obj)

        debug_items.append({
            "candidate": cand,
            "source": source,
            "confidence": conf,
            "bbox": bbox,
        })

    intended = str(row["intended_target"])
    intended_id = ""
    for obj in objects:
        if canonical_match(obj["category"], intended):
            intended_id = obj["id"]
            break

    return {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "candidate_objects": json.dumps(objects, ensure_ascii=False),
        "intended_target_id": intended_id,
        "instruction_type": infer_instruction_type(row["instruction"]),
        "debug_candidates": json.dumps(debug_items, ensure_ascii=False),
    }


def main():
    print("=" * 100)
    print("Build bbox-aware target confirmation eval CSV")
    print("=" * 100)
    print("Input:", INPUT_CSV)
    print("Output:", OUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    models = {}
    for dataset, weight in WEIGHTS.items():
        print(f"[Load] {dataset}: {weight}")
        models[dataset] = YOLO(weight)

    out_rows = []
    debug_rows = []

    for idx, row in df.iterrows():
        extra = build_objects_for_sample(row, models)

        out = row.to_dict()
        out["sample_id"] = str(idx)
        out.update({
            "image_path": extra["image_path"],
            "candidate_objects": extra["candidate_objects"],
            "intended_target_id": extra["intended_target_id"],
            "instruction_type": extra["instruction_type"],
        })
        out_rows.append(out)

        debug = row.to_dict()
        debug["sample_id"] = str(idx)
        debug["debug_candidates"] = extra["debug_candidates"]
        debug_rows.append(debug)

        if (idx + 1) % 20 == 0:
            print(f"[Done] {idx + 1}/{len(df)}")

    out_df = pd.DataFrame(out_rows)
    debug_df = pd.DataFrame(debug_rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    debug_df.to_csv(OUT_DEBUG, index=False)

    print("=" * 100)
    print("[DONE]")
    print("Saved:", OUT_CSV)
    print("Debug:", OUT_DEBUG)
    print("Shape:", out_df.shape)
    print("\nPreview:")
    print(out_df.head(3)[[
        "sample_id", "dataset", "case_id", "frame_idx",
        "instruction", "candidate_objects", "intended_target", "intended_target_id", "gt_state"
    ]].to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
