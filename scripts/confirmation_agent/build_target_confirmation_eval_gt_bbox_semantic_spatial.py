#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import math
import pandas as pd
from pathlib import Path


IN_VALID = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_gt_bbox_valid.csv")
OUT_FINAL = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_gt_bbox_semantic_spatial.csv")
OUT_STATS = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_gt_bbox_semantic_spatial_stats.csv")


def load_objects(s):
    if pd.isna(s):
        return []
    return json.loads(str(s))


def bbox_center(obj):
    x1, y1, x2, y2 = obj["bbox"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(obj):
    x1, y1, x2, y2 = obj["bbox"]
    return max(1, (x2 - x1 + 1) * (y2 - y1 + 1))


def unique_extreme(objects, key_fn, reverse=False, min_gap=30):
    """
    Return the object that is uniquely min/max according to key_fn.
    min_gap avoids generating spatial instructions when objects are too close.
    """
    if len(objects) < 2:
        return None

    sorted_objs = sorted(objects, key=key_fn, reverse=reverse)
    best = sorted_objs[0]
    second = sorted_objs[1]

    if abs(key_fn(best) - key_fn(second)) < min_gap:
        return None

    return best


def unique_size_extreme(objects, largest=True, ratio_thres=1.20):
    if len(objects) < 2:
        return None

    sorted_objs = sorted(objects, key=bbox_area, reverse=largest)
    best = sorted_objs[0]
    second = sorted_objs[1]

    a_best = bbox_area(best)
    a_second = bbox_area(second)

    if largest:
        if a_best / max(a_second, 1) < ratio_thres:
            return None
    else:
        if a_second / max(a_best, 1) < ratio_thres:
            return None

    return best


def make_spatial_instruction(row, objects):
    """
    Generate one spatial instruction per eligible row.
    Priority:
    1. left/right when distinguishable
    2. upper/lower when distinguishable
    3. larger/smaller when distinguishable
    """
    if len(objects) < 2:
        return None

    # x/y centers
    left_obj = unique_extreme(objects, key_fn=lambda o: bbox_center(o)[0], reverse=False, min_gap=40)
    right_obj = unique_extreme(objects, key_fn=lambda o: bbox_center(o)[0], reverse=True, min_gap=40)

    upper_obj = unique_extreme(objects, key_fn=lambda o: bbox_center(o)[1], reverse=False, min_gap=40)
    lower_obj = unique_extreme(objects, key_fn=lambda o: bbox_center(o)[1], reverse=True, min_gap=40)

    large_obj = unique_size_extreme(objects, largest=True, ratio_thres=1.25)
    small_obj = unique_size_extreme(objects, largest=False, ratio_thres=1.25)

    candidates = []

    if left_obj is not None:
        candidates.append(("Segment the object on the left.", left_obj))
    if right_obj is not None:
        candidates.append(("Segment the object on the right.", right_obj))
    if upper_obj is not None:
        candidates.append(("Segment the upper object.", upper_obj))
    if lower_obj is not None:
        candidates.append(("Segment the lower object.", lower_obj))
    if large_obj is not None:
        candidates.append(("Segment the larger object.", large_obj))
    if small_obj is not None:
        candidates.append(("Segment the smaller object.", small_obj))

    if not candidates:
        return None

    # 为了不要重复太多，固定取第一个可区分的空间关系
    instruction, target_obj = candidates[0]

    new_row = row.copy()
    new_row["instruction"] = instruction
    new_row["intended_target"] = target_obj["category"]
    new_row["intended_target_id"] = target_obj["id"]
    new_row["gt_state"] = "unique"
    new_row["clarification_response"] = ""
    new_row["instruction_type"] = "spatial"
    new_row["spatial_relation"] = instruction
    new_row["is_valid"] = True
    new_row["invalid_reason"] = ""

    return new_row


def main():
    df = pd.read_csv(IN_VALID)

    semantic_rows = []
    spatial_rows = []

    for _, row in df.iterrows():
        r = row.to_dict()
        r["instruction_type"] = "semantic"
        r["spatial_relation"] = ""
        semantic_rows.append(r)

        objects = load_objects(row["candidate_objects"])
        spatial = make_spatial_instruction(r, objects)
        if spatial is not None:
            spatial_rows.append(spatial)

    # 重新分配 sample_id
    final_rows = []
    for i, r in enumerate(semantic_rows + spatial_rows):
        r = dict(r)
        r["sample_id"] = str(i)
        final_rows.append(r)

    out_df = pd.DataFrame(final_rows)

    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_FINAL, index=False)

    stats = []
    stats.append({"metric": "num_total", "value": len(out_df)})
    stats.append({"metric": "num_semantic", "value": len(semantic_rows)})
    stats.append({"metric": "num_spatial", "value": len(spatial_rows)})

    for k, v in out_df["gt_state"].value_counts().items():
        stats.append({"metric": f"gt_state::{k}", "value": int(v)})

    for k, v in out_df["instruction_type"].value_counts().items():
        stats.append({"metric": f"instruction_type::{k}", "value": int(v)})

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT_STATS, index=False)

    print("=" * 100)
    print("[DONE] Built semantic + spatial GT-bbox target confirmation eval set")
    print("=" * 100)
    print("Input valid:", IN_VALID)
    print("Output final:", OUT_FINAL)
    print("Stats:", OUT_STATS)
    print("\nStats:")
    print(stats_df.to_string(index=False))

    print("\nPreview spatial:")
    spatial_df = out_df[out_df["instruction_type"].eq("spatial")]
    cols = [
        "sample_id", "dataset", "case_id", "frame_idx",
        "instruction", "candidate_objects", "intended_target",
        "intended_target_id", "gt_state", "instruction_type"
    ]
    if len(spatial_df):
        print(spatial_df.head(10)[cols].to_string(index=False))
    else:
        print("No spatial samples generated.")
    print("=" * 100)


if __name__ == "__main__":
    main()
