#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import json
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


DATA_ROOT = Path(f"{REPO_ROOT}/data")
FRAME_SOURCE_CSV = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_full.csv")

OUT_CSV = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_detector_generated.csv")
OUT_STATS = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_detector_generated_stats.csv")

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


def natural_key(p):
    nums = re.findall(r"\d+", Path(str(p)).name)
    return [int(x) for x in nums] if nums else [Path(str(p)).name]


def collect_files(folder, exts):
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files, key=natural_key)


def resolve_image_path(dataset, case_id, frame_idx):
    case_id = normalize_case_id(case_id)
    img_dir = DATA_ROOT / dataset / "img" / case_id

    imgs = collect_files(img_dir, ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"])

    if int(frame_idx) >= len(imgs):
        raise IndexError(f"frame_idx={frame_idx} exceeds image files in {img_dir}, n={len(imgs)}")

    return imgs[int(frame_idx)]


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

    # 每个 category 只保留最高置信度的一个 bbox，避免同类重复干扰 target confirmation
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


def has_category(objects, category):
    return find_obj_by_category(objects, category) is not None


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
        if abs(bbox_center(sorted_objs[0])[0] - bbox_center(sorted_objs[1])[0]) < 40:
            return None
        return sorted_objs[0]

    if relation == "right":
        sorted_objs = sorted(objects, key=lambda o: bbox_center(o)[0], reverse=True)
        if abs(bbox_center(sorted_objs[0])[0] - bbox_center(sorted_objs[1])[0]) < 40:
            return None
        return sorted_objs[0]

    if relation == "upper":
        sorted_objs = sorted(objects, key=lambda o: bbox_center(o)[1])
        if abs(bbox_center(sorted_objs[0])[1] - bbox_center(sorted_objs[1])[1]) < 40:
            return None
        return sorted_objs[0]

    if relation == "lower":
        sorted_objs = sorted(objects, key=lambda o: bbox_center(o)[1], reverse=True)
        if abs(bbox_center(sorted_objs[0])[1] - bbox_center(sorted_objs[1])[1]) < 40:
            return None
        return sorted_objs[0]

    if relation == "larger":
        sorted_objs = sorted(objects, key=bbox_area, reverse=True)
        if bbox_area(sorted_objs[0]) / max(bbox_area(sorted_objs[1]), 1) < 1.25:
            return None
        return sorted_objs[0]

    if relation == "smaller":
        sorted_objs = sorted(objects, key=bbox_area)
        if bbox_area(sorted_objs[1]) / max(bbox_area(sorted_objs[0]), 1) < 1.25:
            return None
        return sorted_objs[0]

    return None


def build_samples_for_frame(dataset, case_id, frame_idx, image_path, objects):
    samples = []

    if len(objects) == 0:
        return samples

    # 1) unique semantic instructions：每帧最多取前两个 category，避免样本过多
    for obj in objects[:2]:
        add_sample(
            samples, dataset, case_id, frame_idx, image_path, objects,
            instruction=f"Segment the {obj['category']}.",
            intended_obj=obj,
            gt_state="unique",
            instruction_type="semantic",
            clarification_response="",
        )

    # 2) ambiguous semantic instructions：根据检测到的候选自动构造
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
        # 多个心脏结构存在时，cardiac structure 是 ambiguous
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

    # 3) spatial instructions：每帧最多生成一条，基于 bbox 位置/面积
    spatial_candidates = [
        ("left", "Segment the object on the left."),
        ("right", "Segment the object on the right."),
        ("upper", "Segment the upper object."),
        ("lower", "Segment the lower object."),
        ("larger", "Segment the object with the larger bounding box."),
        ("smaller", "Segment the object with the smaller bounding box."),
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


def main():
    print("=" * 100)
    print("Build detector-generated target confirmation eval set")
    print("=" * 100)
    print("Frame source:", FRAME_SOURCE_CSV)
    print("Output:", OUT_CSV)

    source_df = pd.read_csv(FRAME_SOURCE_CSV)

    frame_df = source_df[["dataset", "case_id", "frame_idx"]].drop_duplicates().copy()
    frame_df["case_id"] = frame_df["case_id"].apply(normalize_case_id)
    frame_df["frame_idx"] = frame_df["frame_idx"].astype(int)

    print("Unique frames:", len(frame_df))

    models = {}
    for dataset, weight in WEIGHTS.items():
        print(f"[Load] {dataset}: {weight}")
        models[dataset] = YOLO(weight)

    all_samples = []
    detection_debug = []

    for idx, row in frame_df.iterrows():
        dataset = str(row["dataset"])
        case_id = normalize_case_id(row["case_id"])
        frame_idx = int(row["frame_idx"])

        if dataset not in models:
            print(f"[Skip] no model for dataset: {dataset}")
            continue

        try:
            image_path = resolve_image_path(dataset, case_id, frame_idx)
            objects = run_detector(models[dataset], image_path)

            frame_samples = build_samples_for_frame(
                dataset=dataset,
                case_id=case_id,
                frame_idx=frame_idx,
                image_path=image_path,
                objects=objects,
            )

            all_samples.extend(frame_samples)

            detection_debug.append({
                "dataset": dataset,
                "case_id": case_id,
                "frame_idx": frame_idx,
                "image_path": str(image_path),
                "num_detected_objects": len(objects),
                "detected_objects": json.dumps(objects, ensure_ascii=False),
                "num_generated_samples": len(frame_samples),
            })

        except Exception as e:
            detection_debug.append({
                "dataset": dataset,
                "case_id": case_id,
                "frame_idx": frame_idx,
                "image_path": "",
                "num_detected_objects": 0,
                "detected_objects": "[]",
                "num_generated_samples": 0,
                "error": repr(e),
            })

        if (len(detection_debug) % 20) == 0:
            print(f"[Done frames] {len(detection_debug)}/{len(frame_df)} | samples={len(all_samples)}")

    out_df = pd.DataFrame(all_samples)

    # 去重
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

    debug_df = pd.DataFrame(detection_debug)
    debug_path = OUT_CSV.with_name(OUT_CSV.stem + "_debug.csv")
    debug_df.to_csv(debug_path, index=False)

    stats = []
    stats.append({"metric": "num_frames", "value": len(frame_df)})
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

    print("=" * 100)
    print("[DONE]")
    print("Eval CSV:", OUT_CSV, out_df.shape)
    print("Debug CSV:", debug_path)
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
        print(out_df.head(10)[cols].to_string(index=False))
    else:
        print("No samples generated.")
    print("=" * 100)


if __name__ == "__main__":
    main()
