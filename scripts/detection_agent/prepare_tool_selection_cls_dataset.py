#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import re
import shutil
import json

DATA_ROOT = Path(f"{REPO_ROOT}/data")
OUT_ROOT = Path(f"{REPO_ROOT}/data/yolo_cls_tool_selection")

MAX_FRAMES_PER_CASE = 8
USE_SYMLINK = True

DATASET_TO_CLASS = {
    "Amos": "CT",
    "ACDC": "MRI",
    "CAMUS": "Ultrasound",
    "Montgomery-County-CXR-Set": "Xray",
    "PolypGen2021_MultiCenterData_v3": "Endoscopy",
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

def read_split_cases(split_file):
    split_cases = {"train": [], "val": [], "test": []}
    current = None

    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("[") and s.endswith("]"):
                current = s[1:-1].strip()
                continue
            if current in split_cases:
                split_cases[current].append(normalize_case_id(s))

    return split_cases

def collect_images(case_dir):
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        files.extend(case_dir.glob(ext))
    return sorted(files, key=natural_key)

def sample_frames(files, max_n):
    if len(files) <= max_n:
        return files
    if max_n <= 1:
        return [files[0]]

    idxs = []
    for i in range(max_n):
        idx = round(i * (len(files) - 1) / (max_n - 1))
        idxs.append(idx)

    idxs = sorted(set(idxs))
    return [files[i] for i in idxs]

def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if USE_SYMLINK:
        try:
            dst.symlink_to(src)
            return
        except Exception:
            pass

    shutil.copy2(src, dst)

def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    counts = {}

    for dataset_name, cls_name in DATASET_TO_CLASS.items():
        img_root = DATA_ROOT / dataset_name / "img"
        split_file = DATA_ROOT / "yolo_det" / dataset_name / "split_cases.txt"

        if not img_root.exists():
            raise FileNotFoundError(img_root)
        if not split_file.exists():
            raise FileNotFoundError(split_file)

        split_cases = read_split_cases(split_file)

        for split, cases in split_cases.items():
            counts.setdefault(split, {}).setdefault(cls_name, 0)

            out_cls_dir = OUT_ROOT / split / cls_name
            out_cls_dir.mkdir(parents=True, exist_ok=True)

            for case_id in cases:
                case_dir = img_root / str(case_id)
                if not case_dir.exists():
                    print(f"[WARN] missing case dir: {case_dir}")
                    continue

                imgs = collect_images(case_dir)
                sampled = sample_frames(imgs, MAX_FRAMES_PER_CASE)

                for src in sampled:
                    dst_name = f"{dataset_name}__case{case_id}__{src.stem}{src.suffix}"
                    dst = out_cls_dir / dst_name
                    link_or_copy(src.resolve(), dst)
                    counts[split][cls_name] += 1

    classes = list(DATASET_TO_CLASS.values())
    (OUT_ROOT / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    with open(OUT_ROOT / "counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("[DONE] YOLO classification dataset prepared")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"MAX_FRAMES_PER_CASE: {MAX_FRAMES_PER_CASE}")
    print("=" * 100)
    print(json.dumps(counts, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
