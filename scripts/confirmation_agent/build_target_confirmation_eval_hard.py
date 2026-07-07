#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a HARDER target-confirmation eval set, faithful to the paper's task.
Same task/schema/metrics as v5_balanced200 (4 types {semantic,spatial}x{unique,ambiguous},
balanced 50 each). Only instances are harder:
  - SPATIAL instructions are ORGAN-RELATIVE ("the organ on the left of the {ref}"),
    matching the paper example "the organ on the left of the liver" (unique if exactly
    one organ satisfies; ambiguous if >=2 -> needs clarification). Margin relaxed 120->40.
  - DENSE frames only (>=3 candidates).
Restricted to TEST cases (split_cases.txt [test]). Semantic types unchanged.
Out: data/text_prompt_eval/target_confirmation_eval_hard.csv (+ _stats.csv)
"""
import re, json, random

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
from collections import defaultdict
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

DATA_ROOT = Path(f"{REPO_ROOT}/data")
OUT_CSV = DATA_ROOT / "text_prompt_eval" / "target_confirmation_eval_hard.csv"
OUT_STATS = DATA_ROOT / "text_prompt_eval" / "target_confirmation_eval_hard_stats.csv"
WEIGHTS = {
    "Amos": f"{REPO_ROOT}/runs/yolo26_det/Amos_yolo26m_imgsz1024/weights/best.pt",
    "ACDC": f"{REPO_ROOT}/runs/yolo26_det/ACDC_yolo26m_imgsz1024/weights/best.pt",
    "CAMUS": f"{REPO_ROOT}/runs/yolo26_det/CAMUS_yolo26m_imgsz1024/weights/best.pt",
    "Montgomery-County-CXR-Set": f"{REPO_ROOT}/runs/yolo26_det/Montgomery-County-CXR-Set_yolo26m_imgsz1024/weights/best.pt",
    "PolypGen2021_MultiCenterData_v3": f"{REPO_ROOT}/runs/yolo26_det/PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024/weights/best.pt",
}
MAX_FRAMES = {"Amos": 450, "ACDC": 180, "CAMUS": 180, "Montgomery-County-CXR-Set": 180, "PolypGen2021_MultiCenterData_v3": 120}
FRAMES_PER_CASE = 10
CONF_THRES, IMGSZ, RANDOM_SEED, PER_BUCKET = 0.10, 1024, 2026, 50
SEP_MARGIN, DENSE_MIN, MAX_SPATIAL_PER_FRAME = 40, 3, 2

def normalize_text(x):
    x = "" if pd.isna(x) else str(x)
    return re.sub(r"\s+", " ", x.strip().lower().replace("_", " "))
def natural_key(p):
    nums = re.findall(r"\d+", Path(str(p)).name); return [int(x) for x in nums] if nums else [Path(str(p)).name]
def read_test_cases(dataset):
    sf = DATA_ROOT / "yolo_det" / dataset / "split_cases.txt"
    cases, cur = [], None
    for line in sf.read_text().splitlines():
        s = line.strip()
        if not s: continue
        if s.startswith("[") and s.endswith("]"): cur = s[1:-1].strip(); continue
        if cur == "test": cases.append(s)
    return cases
def collect_frames(dataset):
    frames = []
    for case in read_test_cases(dataset):
        cdir = DATA_ROOT / dataset / "img" / case
        if not cdir.is_dir(): continue
        fs = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]: fs.extend(cdir.glob(ext))
        fs = sorted(fs, key=natural_key)
        if not fs: continue
        if len(fs) <= FRAMES_PER_CASE: frames.extend(fs)
        else:
            step = len(fs) / FRAMES_PER_CASE
            frames.extend(fs[int(i * step)] for i in range(FRAMES_PER_CASE))
    return frames
def parse_case_frame(p):
    p = Path(p); nums = re.findall(r"\d+", p.stem); return str(p.parent.name), int(nums[-1]) if nums else 0
def bbox_center(o):
    x1, y1, x2, y2 = o["bbox"]; return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
def bbox_area(o):
    x1, y1, x2, y2 = o["bbox"]; return max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
def run_detector(model, image_path):
    res = model.predict(source=str(image_path), imgsz=IMGSZ, conf=CONF_THRES, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0: return []
    names = model.names
    xyxy = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy(); clss = res.boxes.cls.cpu().numpy().astype(int)
    best = {}
    for box, conf, cid in zip(xyxy, confs, clss):
        cat = str(names[int(cid)]); k = normalize_text(cat)
        if k not in best or conf > best[k]["confidence"]:
            best[k] = {"category": cat, "bbox": [int(round(float(v))) for v in box.tolist()], "confidence": float(conf)}
    return [{"id": f"o{i}", "category": d["category"], "bbox": d["bbox"], "confidence": round(d["confidence"], 4)}
            for i, d in enumerate(sorted(best.values(), key=lambda x: x["category"]), 1)]
def find_obj(objects, category):
    c = normalize_text(category)
    for o in objects:
        if normalize_text(o["category"]) == c: return o
    return None
def is_directional(cat):
    c = normalize_text(cat); return any(w in c for w in ["left", "right", "upper", "lower"])
def side_sorted(objects, ref, rel):
    # candidates strictly on that side of ref, sorted nearest-first by center-line distance.
    rx, ry = bbox_center(ref); out = []
    for o in objects:
        if o["id"] == ref["id"]: continue
        ox, oy = bbox_center(o)
        if rel == "left" and ox < rx: out.append((rx - ox, o))
        elif rel == "right" and ox > rx: out.append((ox - rx, o))
        elif rel == "above" and oy < ry: out.append((ry - oy, o))
        elif rel == "below" and oy > ry: out.append((oy - ry, o))
    out.sort(key=lambda t: t[0])
    return out
def mk(ds, c, fi, ip, objs, instr, intended, st, it, clar=""):
    if intended is None: return None
    return {"dataset": ds, "case_id": c, "frame_idx": fi, "image_path": str(ip), "instruction": instr,
            "intended_target": intended["category"], "intended_target_id": intended["id"], "gt_state": st,
            "candidate_objects": json.dumps(objs, ensure_ascii=False), "clarification_response": clar, "instruction_type": it}
def build_for_frame(ds, c, fi, ip, objs):
    S = []
    if len(objs) < 2: return S
    for o in objs[:2]:
        S.append(mk(ds, c, fi, ip, objs, f"Segment the {o['category']}.", o, "unique", "semantic"))
    if ds == "Amos":
        lk, rk = find_obj(objs, "left kidney"), find_obj(objs, "right kidney")
        if lk and rk: S.append(mk(ds, c, fi, ip, objs, "Segment the kidney.", rk, "ambiguous", "semantic", "right kidney"))
    if ds == "Montgomery-County-CXR-Set":
        ll, rl = find_obj(objs, "left lung"), find_obj(objs, "right lung")
        if ll and rl: S.append(mk(ds, c, fi, ip, objs, "Segment the lung.", rl, "ambiguous", "semantic", "right lung"))
    if ds == "CAMUS":
        lv, la = find_obj(objs, "Left Ventricle"), find_obj(objs, "Left Atrium")
        if lv and la: S.append(mk(ds, c, fi, ip, objs, "Show me the cardiac chamber.", la, "ambiguous", "semantic", "Left Atrium"))
    if ds == "ACDC" and len(objs) >= 2:
        t = find_obj(objs, "myocardium") or objs[0]
        S.append(mk(ds, c, fi, ip, objs, "Highlight the cardiac structure.", t, "ambiguous", "semantic", t["category"]))
    if len(objs) >= DENSE_MIN:
        rng = random.Random(hash((ds, c, fi)) & 0xffffffff)
        refs = [o for o in objs if not is_directional(o["category"])] or list(objs)
        rng.shuffle(refs); made = 0
        for ref in refs:
            if made >= MAX_SPATIAL_PER_FRAME: break
            rels = ["left", "right", "above", "below"]; rng.shuffle(rels)
            for rel in rels:
                side = side_sorted(objs, ref, rel)
                if not side: continue
                rc = ref["category"]
                instr_u = (f"Segment the organ immediately to the {rel} of the {rc}." if rel in ("left","right") else f"Segment the organ immediately {rel} the {rc}.")
                instr_a = (f"Segment the organ on the {rel} of the {rc}." if rel in ("left","right") else f"Segment the organ {rel} the {rc}.")
                nearest = side[0][1]
                clean = (len(side) == 1) or (side[1][0] - side[0][0] >= SEP_MARGIN)
                if clean and normalize_text(nearest["category"]) != normalize_text(rc):
                    S.append(mk(ds, c, fi, ip, objs, instr_u, nearest, "unique", "spatial")); made += 1
                if made < MAX_SPATIAL_PER_FRAME and len(side) >= 2:
                    t = rng.choice([o for _, o in side])
                    S.append(mk(ds, c, fi, ip, objs, instr_a, t, "ambiguous", "spatial", t["category"])); made += 1
                if made >= MAX_SPATIAL_PER_FRAME: break
    return [s for s in S if s is not None]
def main():
    rng = random.Random(RANDOM_SEED); alls = []
    for ds, w in WEIGHTS.items():
        if not Path(w).exists(): print(f"[skip] {ds}"); continue
        model = YOLO(w); frames = collect_frames(ds); rng.shuffle(frames); frames = frames[:MAX_FRAMES.get(ds, 120)]
        print(f"[{ds}] using {len(frames)} test frames")
        for f in frames:
            c, fi = parse_case_frame(f); alls.extend(build_for_frame(ds, c, fi, f, run_detector(model, f)))
    seen, uniq = set(), []
    for s in alls:
        k = (s["dataset"], s["case_id"], s["instruction"], s["intended_target_id"])
        if k in seen: continue
        seen.add(k); uniq.append(s)
    buckets = defaultdict(list)
    for s in uniq: buckets[(s["instruction_type"], s["gt_state"])].append(s)
    print("\n=== available per bucket ==="); [print(f"  {k}: {len(buckets[k])}") for k in sorted(buckets)]
    final = []
    for key in [("semantic", "unique"), ("semantic", "ambiguous"), ("spatial", "unique"), ("spatial", "ambiguous")]:
        pool = buckets.get(key, []); rng.shuffle(pool); take = pool[:PER_BUCKET]
        for i, s in enumerate(take): s["sample_id"] = f"{key[0]}_{key[1]}_{i:03d}"
        final.extend(take); print(f"  -> {key}: {len(take)}/{PER_BUCKET}")
    cols = ["dataset","case_id","frame_idx","image_path","instruction","intended_target","intended_target_id","gt_state","candidate_objects","clarification_response","instruction_type","sample_id"]
    pd.DataFrame(final)[cols].to_csv(OUT_CSV, index=False)
    pd.DataFrame(final).groupby(["instruction_type","gt_state"]).size().reset_index(name="n").to_csv(OUT_STATS, index=False)
    print(f"\nTotal: {len(final)} -> {OUT_CSV}")
if __name__ == "__main__":
    main()
