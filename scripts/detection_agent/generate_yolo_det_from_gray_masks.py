import random

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import shutil
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


DATA_ROOT = Path(f"{REPO_ROOT}/data")

DATASETS = [
    # "Amos",
    "CAMUS",
    # "ACDC",
    # "Montgomery-County-CXR-Set",
    # "PolypGen2021_MultiCenterData_v3",
]

IMG_DIR_NAME = "img"
LABEL_DIR_NAME = "label"

OUTPUT_ROOT = DATA_ROOT / "yolo_det"

SPLIT_RATIO = {
    "train": 0.5,
    "val": 0.2,
    "test": 0.3,
}

RANDOM_SEED = 42
FRAME_STRIDE = 10


def normalize_case_id(case_id):
    case_id = str(case_id).strip()
    return str(int(float(case_id))) if case_id.replace(".", "", 1).isdigit() else case_id


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


def build_dataset_gray_mapping(df):
    gray_mapping = {}

    for value in df["灰度标签像素含义"].dropna():
        one_mapping = parse_gray_label_mapping(value)

        for gray_value, cls_name in one_mapping.items():
            if gray_value in gray_mapping and gray_mapping[gray_value] != cls_name:
                print(
                    f"[Warning] gray value {gray_value} conflict: "
                    f"{gray_mapping[gray_value]} vs {cls_name}. "
                    f"Using {gray_mapping[gray_value]}."
                )
                continue

            gray_mapping[gray_value] = cls_name

    return dict(sorted(gray_mapping.items(), key=lambda x: x[0]))


def build_class_name_to_id(gray_mapping):
    class_name_to_id = {}

    for _, cls_name in gray_mapping.items():
        if cls_name not in class_name_to_id:
            class_name_to_id[cls_name] = len(class_name_to_id)

    return class_name_to_id


def get_bbox_from_class_mask(class_mask):
    ys, xs = class_mask.nonzero()

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def xyxy_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox

    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1

    x_center = x_min + box_w / 2.0
    y_center = y_min + box_h / 2.0

    return (
        x_center / img_w,
        y_center / img_h,
        box_w / img_w,
        box_h / img_h,
    )


def collect_image_files(case_img_dir):
    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
        image_files.extend(case_img_dir.glob(ext))

    return sorted(image_files)


def collect_label_files(case_label_dir):
    label_files = list(case_label_dir.glob("*.png"))

    def sort_key(path):
        stem = path.stem
        return int(stem) if stem.isdigit() else stem

    return sorted(label_files, key=sort_key)


def split_cases(pass_df):
    rows = list(pass_df.to_dict("records"))
    random.Random(RANDOM_SEED).shuffle(rows)

    n = len(rows)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])

    return {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }


def write_split_cases_file(output_dir, split_rows):
    split_file = output_dir / "split_cases.txt"

    with open(split_file, "w") as f:
        for split, rows in split_rows.items():
            f.write(f"[{split}]\n")
            for row in rows:
                case_id = normalize_case_id(row["序号"])
                f.write(f"{case_id}\n")
            f.write("\n")


def prepare_output_dirs(output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def process_one_case(
    dataset_name,
    row,
    split,
    img_root,
    label_root,
    output_img_dir,
    output_label_dir,
    gray_mapping,
    class_name_to_id,
):
    case_id = normalize_case_id(row["序号"])

    case_img_dir = img_root / case_id
    case_label_dir = label_root / case_id

    if not case_img_dir.exists():
        print(f"[Missing image dir] {case_img_dir}")
        return 0, 0

    if not case_label_dir.exists():
        print(f"[Missing label dir] {case_label_dir}")
        return 0, 0

    image_files = collect_image_files(case_img_dir)
    label_files = collect_label_files(case_label_dir)

    if len(image_files) != len(label_files):
        print(
            f"[Count mismatch] {dataset_name} case {case_id}: "
            f"images={len(image_files)}, labels={len(label_files)}"
        )

    paired_files = list(zip(image_files, label_files))
    paired_files = paired_files[::FRAME_STRIDE]

    sample_count = 0
    object_count = 0

    for img_file, label_file in paired_files:
        img = cv2.imread(str(img_file))
        label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[Read image failed] {img_file}")
            continue

        if label is None:
            print(f"[Read label failed] {label_file}")
            continue

        img_h, img_w = label.shape[:2]
        yolo_lines = []

        for gray_value, cls_name in gray_mapping.items():
            class_mask = (label == gray_value).astype("uint8")
            bbox = get_bbox_from_class_mask(class_mask)

            if bbox is None:
                continue

            cls_id = class_name_to_id[cls_name]

            x_center, y_center, box_w, box_h = xyxy_to_yolo(
                bbox, img_w, img_h
            )

            yolo_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            )

        if not yolo_lines:
            continue

        out_name = f"{dataset_name}_{case_id}_{label_file.stem}"

        out_img_path = output_img_dir / split / f"{out_name}{img_file.suffix.lower()}"
        out_label_path = output_label_dir / split / f"{out_name}.txt"

        cv2.imwrite(str(out_img_path), img)

        with open(out_label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        sample_count += 1
        object_count += len(yolo_lines)

    return sample_count, object_count


def write_dataset_files(output_dir, class_name_to_id):
    classes_path = output_dir / "classes.txt"
    yaml_path = output_dir / "dataset.yaml"

    with open(classes_path, "w") as f:
        for cls_name, cls_id in sorted(class_name_to_id.items(), key=lambda x: x[1]):
            f.write(f"{cls_id}: {cls_name}\n")

    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n")

        for cls_name, cls_id in sorted(class_name_to_id.items(), key=lambda x: x[1]):
            f.write(f"  {cls_id}: {cls_name}\n")


def process_one_dataset(dataset_name):
    dataset_dir = DATA_ROOT / dataset_name
    excel_path = dataset_dir / f"{dataset_name}.xlsx"

    img_root = dataset_dir / IMG_DIR_NAME
    label_root = dataset_dir / LABEL_DIR_NAME

    output_dir = OUTPUT_ROOT / dataset_name
    output_img_dir = output_dir / "images"
    output_label_dir = output_dir / "labels"

    prepare_output_dirs(output_dir)

    if not excel_path.exists():
        print(f"[Skip] Excel not found: {excel_path}")
        return

    df = pd.read_excel(excel_path)

    required_cols = ["序号", "检查1", "灰度标签像素含义"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{excel_path} missing column: {col}")

    gray_mapping = build_dataset_gray_mapping(df)
    class_name_to_id = build_class_name_to_id(gray_mapping)

    print(f"\n[{dataset_name}] Gray mapping: {gray_mapping}")
    print(f"[{dataset_name}] Classes: {class_name_to_id}")

    pass_df = df[df["检查1"].astype(str).str.lower().str.strip() == "pass"]
    split_rows = split_cases(pass_df)

    write_split_cases_file(output_dir, split_rows)

    total_images = {"train": 0, "val": 0, "test": 0}
    total_objects = {"train": 0, "val": 0, "test": 0}

    for split, rows in split_rows.items():
        for row in tqdm(rows, desc=f"{dataset_name}-{split}"):
            image_count, object_count = process_one_case(
                dataset_name=dataset_name,
                row=row,
                split=split,
                img_root=img_root,
                label_root=label_root,
                output_img_dir=output_img_dir,
                output_label_dir=output_label_dir,
                gray_mapping=gray_mapping,
                class_name_to_id=class_name_to_id,
            )

            total_images[split] += image_count
            total_objects[split] += object_count

    write_dataset_files(output_dir, class_name_to_id)

    print(f"\n[{dataset_name}] Done")
    print(f"Train images: {total_images['train']}, objects: {total_objects['train']}")
    print(f"Val images:   {total_images['val']}, objects: {total_objects['val']}")
    print(f"Test images:  {total_images['test']}, objects: {total_objects['test']}")
    print(f"Classes: {class_name_to_id}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS:
        process_one_dataset(dataset_name)


if __name__ == "__main__":
    main()