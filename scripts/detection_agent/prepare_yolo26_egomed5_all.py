#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import os
import shutil
import yaml
from collections import OrderedDict


SRC_ROOT = Path(f"{REPO_ROOT}/data/yolo_det")
OUT_ROOT = Path(f"{REPO_ROOT}/data/yolo_det/EgoMed5_All")

DATASETS = [
    "Amos",
    "CAMUS",
    "ACDC",
    "Montgomery-County-CXR-Set",
    "PolypGen2021_MultiCenterData_v3",
]

SPLITS = ["train", "val", "test"]

USE_SYMLINK = True
OVERWRITE = True


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_names(names):
    """
    YOLO names may be list or dict.
    Return old_id -> class_name.
    """
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError(f"Unsupported names format: {type(names)}")


def safe_link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if USE_SYMLINK:
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def find_split_dir(base, split, kind):
    """
    kind: images or labels

    Supports common YOLO layouts:
      images/train
      train/images
      images/val
      val/images
    """
    candidates = [
        base / kind / split,
        base / split / kind,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def collect_files(root, exts):
    files = []
    for ext in exts:
        files.extend(root.glob(f"*{ext}"))
    return sorted(files)


def image_to_label_path(image_path, image_dir, label_dir):
    rel = image_path.relative_to(image_dir)
    return label_dir / rel.with_suffix(".txt")


def main():
    if OUT_ROOT.exists() and OVERWRITE:
        shutil.rmtree(OUT_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build global class mapping by class name.
    # ------------------------------------------------------------------
    global_names = []
    per_dataset_old_to_new = {}

    print("=" * 100)
    print("Building global class mapping")
    print("=" * 100)

    for dataset in DATASETS:
        data_yaml = SRC_ROOT / dataset / "dataset.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Missing dataset.yaml: {data_yaml}")

        cfg = load_yaml(data_yaml)
        old_id_to_name = normalize_names(cfg["names"])

        old_to_new = {}
        for old_id, name in old_id_to_name.items():
            if name not in global_names:
                global_names.append(name)
            old_to_new[int(old_id)] = int(global_names.index(name))

        per_dataset_old_to_new[dataset] = old_to_new

        print(f"\n[{dataset}]")
        for old_id, name in old_id_to_name.items():
            print(f"  old {old_id:2d} -> new {old_to_new[old_id]:2d}: {name}")

    print("\nGlobal names:")
    for i, name in enumerate(global_names):
        print(f"  {i}: {name}")

    # ------------------------------------------------------------------
    # 2. Create merged image/label files with remapped labels.
    # ------------------------------------------------------------------
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    for split in SPLITS:
        out_img_dir = OUT_ROOT / "images" / split
        out_lab_dir = OUT_ROOT / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lab_dir.mkdir(parents=True, exist_ok=True)

        for dataset in DATASETS:
            dataset_root = SRC_ROOT / dataset

            image_dir = find_split_dir(dataset_root, split, "images")
            label_dir = find_split_dir(dataset_root, split, "labels")

            if image_dir is None or label_dir is None:
                print(f"[Warning] Skip missing split {dataset}/{split}: image_dir={image_dir}, label_dir={label_dir}")
                continue

            image_files = collect_files(image_dir, image_exts)
            old_to_new = per_dataset_old_to_new[dataset]

            print(f"\nMerging {dataset}/{split}: {len(image_files)} images")

            for image_path in image_files:
                label_path = image_to_label_path(image_path, image_dir, label_dir)
                if not label_path.exists():
                    print(f"[Warning] Missing label for image: {image_path}")
                    continue

                # Prefix dataset name to avoid filename collision.
                out_stem = f"{dataset}__{image_path.stem}"
                out_img = out_img_dir / f"{out_stem}{image_path.suffix.lower()}"
                out_lab = out_lab_dir / f"{out_stem}.txt"

                safe_link_or_copy(image_path.resolve(), out_img)

                remapped_lines = []
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue

                        parts = s.split()
                        old_cls = int(float(parts[0]))
                        if old_cls not in old_to_new:
                            raise ValueError(f"{dataset}: old class id {old_cls} not in mapping. File: {label_path}")

                        new_cls = old_to_new[old_cls]
                        parts[0] = str(new_cls)
                        remapped_lines.append(" ".join(parts))

                with open(out_lab, "w", encoding="utf-8") as f:
                    f.write("\n".join(remapped_lines) + ("\n" if remapped_lines else ""))

    # ------------------------------------------------------------------
    # 3. Write merged dataset.yaml.
    # ------------------------------------------------------------------
    out_yaml = OUT_ROOT / "dataset.yaml"
    yaml_dict = OrderedDict()
    yaml_dict["path"] = str(OUT_ROOT)
    yaml_dict["train"] = "images/train"
    yaml_dict["val"] = "images/val"
    yaml_dict["test"] = "images/test"
    yaml_dict["nc"] = len(global_names)
    yaml_dict["names"] = {i: name for i, name in enumerate(global_names)}

    # PyYAML OrderedDict dump is ugly; use normal dict preserving insertion order.
    yaml_dict = {
        "path": str(OUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(global_names),
        "names": {i: name for i, name in enumerate(global_names)},
    }

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False, allow_unicode=True)

    print("\n" + "=" * 100)
    print("Done.")
    print(f"Merged YOLO dataset: {OUT_ROOT}")
    print(f"dataset.yaml:        {out_yaml}")
    print("=" * 100)


if __name__ == "__main__":
    main()
