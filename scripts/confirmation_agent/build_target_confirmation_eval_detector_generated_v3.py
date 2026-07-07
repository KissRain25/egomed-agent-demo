#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import json
import random
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


DATA_ROOT = Path(f"{REPO_ROOT}/data")

OUT_CSV = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_detector_generated_v3.csv")
OUT_STATS = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_detector_generated_v3_stats.csv")
OUT_DEBUG = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_detector_generated_v3_debug.csv")

WEIGHTS = {
    "Amos": f"{REPO_ROOT}/runs/yolo26_det/Amos_yolo26m_imgsz1024/weights/best.pt",
    "ACDC": f"{REPO_ROOT}/runs/yolo26_det/ACDC_yolo26m_imgsz1024/weights/best.pt",
    "CAMUS": f"{REPO_ROOT}/runs/yolo26_det/CAMUS_yolo26m_imgsz1024/weights/best.pt",
    "Montgomery-County-CXR-Set": f"{REPO_ROOT}/runs/yolo26_det/Montgomery-County-CXR-Set_yolo26m_imgsz1024/weights/best.pt",
    "PolypGen2021_MultiCenterData_v3": f"{REPO_ROOT}/runs/yolo26_det/PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024/weights/best.pt",
}

# 每个数据集最多抽多少帧。太多会跑慢，可以之后调大。
MAX_FRAMES_PER_DATASET = {
    "Amos": 80,
    "ACDC": 60,
    "CAMUS": 60,
    "Montgomery-County-CXR-Set": 60,
    "PolypGen2021_MultiCenterData_v3": 40,
}

CONF_THRES = 0.10
IMGSZ = 1024
RANDOM_SEED = 2026


def normalize_text(x):
    x = "" if pd.isna(x) else str(x)
    x = x.strip().lower().replace("_", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def natural_key(p):
    nums = re.findall(r"\d+", Path(str(p)).name)
    return [int(x) for x in nums] if nums else [Path(str(p)).name]


def collect_image_files(dataset):
    img_root = DATA_ROOT / dataset / "img"
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        files.extend(img_root.rglob(ext))
    return sorted(files, key=natural_key)


def parse_case_frame_from_image_path(dataset, image_path):
    image_path = Path(image_path)
    case_id = image_path.parent.name

    # 常见 stem: 12_0000
    nums = re.findall(r"\d+", image_path.stem)
    if len(nums) >= 2:
        frame_idx = int(nums[-1])
    elif len(nums) == 1:
        frame_idx = int(nums[0])
    else:
        frame_idx = 0

    return str(case_id), int(frame_idx)


def bbox_center(obj):
    x1, y1, x2, y2 = obj["bbox"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(obj):
    x1, y1, x2, y2 = obj["bbox"]
    return max(1, (x2 - x1 + 1) * (y2 - y1 + 1))


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
        })

    # 每类只保留最高置信度一个框
    best_by_cat = {}
    for d in dets:
        key = normalize_text(d["category"])
        if key not in best_by_cat or d["confidence"] > best_by_cat[key]["confidence"]:
            best_by_cat[key] = d

    final = []
    for i, d in enumerate(sorted(best_by_cat.values(), key=lambda x: x["category"]), 1):
        final.append({
            "id": f"o{i}",
            "category": d["category"],
            "bbox": d["bbox"],
            "confidence": round(float(d["confidence"]), 4),
        })

    return final


def find_obj_by_category(objects, category):
    cat_norm = normalize_text(category)
    for obj in objects:
        if normalize_text(obj["category"]) == cat_norm:
            return obj
    return None


def add_sample(samples, dataset, case_id, frame_idx, image_path, objects,
               instruction, intended_obj, gt_state, instruction_type,
               clarification_response=""):
    if intended_obj is None:
        return

    samples.append({
        "dataset": dataset,
        "case_id": case_id,
        "frame_idx": frame_idx,
        "image_path": str(image_path),
        "instruction": instruction,
        "intended_target": intended_obj["category"],
        "intended_target_id": intended_obj["id"],
        "gt_state": gt_state,
        "candidate_objects": json.dumps(objects, ensure_ascii=False),
        "clarification_response": clarification_response,
        "instruction_type": instruction_type,
    })


def unique_spatial_object(objects, relation):
    if len(objects) < 2:
        return None

    if relation == "left":
        sorted_objs = sorted(objects, key=lambda o: bbox_center(o)[0])
        if abs(bbox_center(sorted_objs[0])[0] - bbox_center(sorted_objs[1])[0]) < 120:
            return None
        return sorted_objs[0]

    if relation == "right":
        sorted_objs = sorted(objects, key=lambda o: bbox_center(o)[0], reverse=True)
        if abs(bbox_center(sorted_objs[0])[0] - bbox_center(sorted_objs[1])[0]) < 120:
            return None
        return sorted_objs[0]

    if relation == "upper":
        sorted_objs = sorted(objects, key=lambda o: bbox_center(o)[1])
        if abs(bbox_center(sorted_objs[0])[1] - bbox_center(sorted_objs[1])[1]) < 120:
            return None
        return sorted_objs[0]

    if relation == "lower":
        sorted_objs = sorted(objects, key=lambda o: bbox_center(o)[1], reverse=True)
        if abs(bbox_center(sorted_objs[0])[1] - bbox_center(sorted_objs[1])[1]) < 120:
            return None
        return sorted_objs[0]

    if relation == "larger":
        sorted_objs = sorted(objects, key=bbox_area, reverse=True)
        if bbox_area(sorted_objs[0]) / max(bbox_area(sorted_objs[1]), 1) < 1.8:
            return None
        return sorted_objs[0]

    if relation == "smaller":
        sorted_objs = sorted(objects, key=bbox_area)
        if bbox_area(sorted_objs[1]) / max(bbox_area(sorted_objs[0]), 1) < 1.8:
            return None
        return sorted_objs[0]

    return None


def build_samples_for_frame(dataset, case_id, frame_idx, image_path, objects):
    samples = []

    if len(objects) == 0:
        return samples

    # 1) unique semantic: 每帧最多两个，避免单一帧过多样本
    for obj in objects[:2]:
        add_sample(
            samples, dataset, case_id, frame_idx, image_path, objects,
            instruction=f"Segment the {obj['category']}.",
            intended_obj=obj,
            gt_state="unique",
            instruction_type="semantic",
            clarification_response="",
        )

    # 2) ambiguous semantic
    if dataset == "Amos":
        lk = find_obj_by_category(objects, "left kidney")
        rk = find_obj_by_category(objects, "right kidney")
        if lk is not None and rk is not None:
            add_sample(
                samples, dataset, case_id, frame_idx, image_path, objects,
                instruction="Segment the kidney.",
                intended_obj=rk,
                gt_state="ambiguous",
                instruction_type="semantic",
                clarification_response="right kidney",
            )

    if dataset == "Montgomery-County-CXR-Set":
        ll = find_obj_by_category(objects, "left lung")
        rl = find_obj_by_category(objects, "right lung")
        if ll is not None and rl is not None:
            add_sample(
                samples, dataset, case_id, frame_idx, image_path, objects,
                instruction="Segment the lung.",
                intended_obj=rl,
                gt_state="ambiguous",
                instruction_type="semantic",
                clarification_response="right lung",
            )

    if dataset == "ACDC":
        if len(objects) >= 2:
            intended = find_obj_by_category(objects, "myocardium") or objects[0]
            add_sample(
                samples, dataset, case_id, frame_idx, image_path, objects,
                instruction="Highlight the cardiac structure.",
                intended_obj=intended,
                gt_state="ambiguous",
                instruction_type="semantic",
                clarification_response=intended["category"],
            )

    if dataset == "CAMUS":
        lv = find_obj_by_category(objects, "Left Ventricle")
        la = find_obj_by_category(objects, "Left Atrium")
        if lv is not None and la is not None:
            add_sample(
                samples, dataset, case_id, frame_idx, image_path, objects,
                instruction="Show me the cardiac chamber.",
                intended_obj=la,
                gt_state="ambiguous",
                instruction_type="semantic",
                clarification_response="Left Atrium",
            )

    # 3) spatial: 每帧最多一条
    spatial_candidates = [
        ("upper", "Segment the upper object."),
        ("lower", "Segment the lower object."),
        ("larger", "Segment the object with the larger bounding box."),
        ("smaller", "Segment the object with the smaller bounding box."),
        ("left", "Segment the object on the left side of the image."),
        ("right", "Segment the object on the right side of the image."),
    ]

    for relation, instruction in spatial_candidates:
        obj = unique_spatial_object(objects, relation)
        if obj is not None:
            add_sample(
                samples, dataset, case_id, frame_idx, image_path, objects,
                instruction=instruction,
                intended_obj=obj,
                gt_state="unique",
                instruction_type="spatial",
                clarification_response="",
            )
            break

    return samples


def sample_frames(dataset, image_files, max_frames):
    random.seed(RANDOM_SEED)

    if len(image_files) <= max_frames:
        return image_files

    # 均匀采样 + 随机扰动，避免只抽到开头
    step = len(image_files) / max_frames
    selected = []
    for i in range(max_frames):
        start = int(i * step)
        end = int((i + 1) * step)
        if end <= start:
            end = min(start + 1, len(image_files))
        selected.append(random.choice(image_files[start:end]))

    return selected


def main():
    print("=" * 100)
    print("Build detector-generated target confirmation eval set V2")
    print("=" * 100)
    print("Output:", OUT_CSV)

    models = {}
    for dataset, weight in WEIGHTS.items():
        print(f"[Load] {dataset}: {weight}")
        models[dataset] = YOLO(weight)

    all_samples = []
    debug_rows = []

    for dataset, model in models.items():
        image_files = collect_image_files(dataset)
        max_frames = MAX_FRAMES_PER_DATASET[dataset]
        selected_files = sample_frames(dataset, image_files, max_frames)

        print(f"\n[Dataset] {dataset}: total_images={len(image_files)}, selected={len(selected_files)}")

        for idx, image_path in enumerate(selected_files, 1):
            try:
                case_id, frame_idx = parse_case_frame_from_image_path(dataset, image_path)
                objects = run_detector(model, image_path)

                frame_samples = build_samples_for_frame(
                    dataset=dataset,
                    case_id=case_id,
                    frame_idx=frame_idx,
                    image_path=image_path,
                    objects=objects,
                )

                all_samples.extend(frame_samples)

                debug_rows.append({
                    "dataset": dataset,
                    "case_id": case_id,
                    "frame_idx": frame_idx,
                    "image_path": str(image_path),
                    "num_detected_objects": len(objects),
                    "detected_objects": json.dumps(objects, ensure_ascii=False),
                    "num_generated_samples": len(frame_samples),
                    "error": "",
                })

            except Exception as e:
                debug_rows.append({
                    "dataset": dataset,
                    "case_id": "",
                    "frame_idx": "",
                    "image_path": str(image_path),
                    "num_detected_objects": 0,
                    "detected_objects": "[]",
                    "num_generated_samples": 0,
                    "error": repr(e),
                })

            if idx % 20 == 0:
                print(f"  [Done] {idx}/{len(selected_files)} | total_samples={len(all_samples)}")

    out_df = pd.DataFrame(all_samples)

    if len(out_df):
        out_df = out_df.drop_duplicates(
            subset=[
                "dataset", "case_id", "frame_idx",
                "instruction", "intended_target", "candidate_objects",
            ],
            keep="first",
        ).reset_index(drop=True)

        out_df["sample_id"] = [str(i) for i in range(len(out_df))]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    debug_df = pd.DataFrame(debug_rows)
    debug_df.to_csv(OUT_DEBUG, index=False)

    stats = []
    stats.append({"metric": "num_samples", "value": len(out_df)})

    if len(out_df):
        for k, v in out_df["gt_state"].value_counts().items():
            stats.append({"metric": f"gt_state::{k}", "value": int(v)})
        for k, v in out_df["instruction_type"].value_counts().items():
            stats.append({"metric": f"instruction_type::{k}", "value": int(v)})
        for k, v in out_df["dataset"].value_counts().items():
            stats.append({"metric": f"dataset::{k}", "value": int(v)})

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT_STATS, index=False)

    print("\n" + "=" * 100)
    print("[DONE]")
    print("Eval CSV:", OUT_CSV, out_df.shape)
    print("Debug CSV:", OUT_DEBUG)
    print("Stats:", OUT_STATS)
    print()
    print(stats_df.to_string(index=False))

    print("\nPreview:")
    if len(out_df):
        cols = [
            "sample_id", "dataset", "case_id", "frame_idx",
            "instruction", "intended_target", "intended_target_id",
            "gt_state", "instruction_type", "candidate_objects"
        ]
        print(out_df.head(20)[cols].to_string(index=False))
    else:
        print("No samples generated.")

    print("=" * 100)


if __name__ == "__main__":
    main()
