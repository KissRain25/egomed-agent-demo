import re

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import shutil
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from sam2.build_sam import build_sam2_video_predictor


# ============================================================
# Config
# ============================================================

DATA_ROOT = Path(f"{REPO_ROOT}/data")

DATASET_NAME = "Amos"

FOCUS_TARGETS = [
    "liver",
    "right kidney",
    "left kidney",
    "spleen",
    "stomach",
]

YOLO_RUN_ROOT = Path(f"{REPO_ROOT}/runs/yolo26_det")

YOLO_WEIGHT = YOLO_RUN_ROOT / "Amos_yolo26m_imgsz1024" / "weights" / "best.pt"

SAM2_CHECKPOINT = f"{REPO_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

RUN_ROOT = (
    Path(f"{REPO_ROOT}/runs/eval_tracking_amos_focus")
    / "egomed_ltc_yolo26m_sam2_base_plus"
)

DET_CONF_THRES = 0.3
TRACK_IOU_THRES = 0.3
YOLO_IMGSZ = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OVERWRITE_PRED_MASK = True


# ============================================================
# Utilities
# ============================================================

def natural_key(path_or_name):
    name = Path(path_or_name).name
    nums = re.findall(r"\d+", name)
    return [int(x) for x in nums] if nums else [name]


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


def read_split_cases(split_file, target_split="test"):
    cases = []
    current_split = None

    with open(split_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("[") and s.endswith("]"):
                current_split = s[1:-1]
                continue

            if current_split == target_split:
                cases.append(s)

    return cases


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


def build_dataset_gray_mapping(excel_file):
    df = pd.read_excel(excel_file)

    if "灰度标签像素含义" not in df.columns:
        raise ValueError(f"Missing column 灰度标签像素含义 in {excel_file}")

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


def load_yolo_class_names(data_yaml):
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):
        id_to_name = {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        id_to_name = {i: name for i, name in enumerate(names)}
    else:
        raise ValueError("Unsupported names format in dataset.yaml")

    return id_to_name


def build_class_id_to_gray(id_to_name, gray_to_name):
    name_to_gray = {}
    for gray_value, cls_name in gray_to_name.items():
        name_to_gray[cls_name] = gray_value

    class_id_to_gray = {}
    for class_id, cls_name in id_to_name.items():
        if cls_name not in name_to_gray:
            raise ValueError(
                f"Class name '{cls_name}' from dataset.yaml not found in Excel gray mapping."
            )
        class_id_to_gray[class_id] = name_to_gray[cls_name]

    return class_id_to_gray


def filter_focus_classes(id_to_name, class_id_to_gray):
    focus_set = set(FOCUS_TARGETS)

    focus_id_to_name = {
        class_id: class_name
        for class_id, class_name in id_to_name.items()
        if class_name in focus_set
    }

    missing = focus_set - set(focus_id_to_name.values())
    if missing:
        raise ValueError(f"Focus targets not found in dataset.yaml names: {missing}")

    focus_class_id_to_gray = {
        class_id: class_id_to_gray[class_id]
        for class_id in focus_id_to_name.keys()
    }

    return focus_id_to_name, focus_class_id_to_gray


# ============================================================
# Metric utilities
# ============================================================

def dice_score(pred_mask, gt_mask, eps=1e-6):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    inter = np.logical_and(pred_mask, gt_mask).sum()
    return float((2.0 * inter + eps) / (pred_sum + gt_sum + eps))


def mask_to_bbox(mask):
    if mask is None:
        return None

    mask = mask.astype(bool)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_iou(box1, box2, eps=1e-6):
    if box1 is None or box2 is None:
        return 0.0

    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1 + 1.0)
    inter_h = max(0.0, y2 - y1 + 1.0)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0] + 1.0)) * max(
        0.0, float(box1[3] - box1[1] + 1.0)
    )
    area2 = max(0.0, float(box2[2] - box2[0] + 1.0)) * max(
        0.0, float(box2[3] - box2[1] + 1.0)
    )

    return float(inter / (area1 + area2 - inter + eps))


def bbox_to_str(box):
    if box is None:
        return ""
    return ",".join([f"{float(x):.2f}" for x in box])


# ============================================================
# YOLO utilities
# ============================================================

def detect_one_frame(yolo_model, image_path, focus_class_ids):
    results = yolo_model.predict(
        source=str(image_path),
        conf=DET_CONF_THRES,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )

    detections_by_class = {}

    if not results:
        return detections_by_class

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return detections_by_class

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)

    for box, conf, class_id in zip(boxes, confs, clss):
        if class_id not in focus_class_ids:
            continue

        if class_id not in detections_by_class:
            detections_by_class[class_id] = {
                "box": box.astype(np.float32),
                "conf": float(conf),
            }
        elif float(conf) > detections_by_class[class_id]["conf"]:
            detections_by_class[class_id] = {
                "box": box.astype(np.float32),
                "conf": float(conf),
            }

    return detections_by_class


def precompute_yolo_detections(yolo_model, image_files, focus_class_ids):
    detections_by_frame = []
    for image_path in tqdm(image_files, desc="YOLO detect frames", leave=False):
        detections_by_frame.append(
            detect_one_frame(
                yolo_model=yolo_model,
                image_path=image_path,
                focus_class_ids=focus_class_ids,
            )
        )
    return detections_by_frame


# ============================================================
# SAM2 utilities
# ============================================================

def init_sam2_predictor():
    predictor = build_sam2_video_predictor(
        SAM2_MODEL_CFG,
        SAM2_CHECKPOINT,
        device=DEVICE,
    )
    return predictor


def sam2_add_box(predictor, inference_state, frame_idx, obj_id, box):
    box = np.asarray(box, dtype=np.float32)

    try:
        ret = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
            clear_old_points=True,
        )
    except TypeError:
        ret = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )

    return ret


def sam2_masks_from_output(out_obj_ids, out_mask_logits, obj_id_to_class_id):
    masks_by_class = {}

    for i, obj_id in enumerate(out_obj_ids):
        obj_id_int = int(obj_id)

        if obj_id_int not in obj_id_to_class_id:
            continue

        class_id = obj_id_to_class_id[obj_id_int]

        mask_logit = out_mask_logits[i]
        mask = (mask_logit > 0.0).detach().cpu().numpy()

        if mask.ndim == 3:
            mask = mask[0]

        masks_by_class[class_id] = mask.astype(bool)

    return masks_by_class


def propagate_video(predictor, inference_state, start_frame_idx):
    try:
        return predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
        )
    except TypeError:
        print(
            "[Warning] This SAM2 version does not support start_frame_idx. "
            "Falling back to default propagation."
        )
        return predictor.propagate_in_video(inference_state)


# ============================================================
# Prediction mask utilities
# ============================================================

def compose_pred_label(mask_by_class, class_id_to_gray, frame_shape):
    h, w = frame_shape
    pred_label = np.zeros((h, w), dtype=np.uint8)

    for class_id in sorted(mask_by_class.keys()):
        if class_id not in class_id_to_gray:
            continue

        mask = mask_by_class[class_id]

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        gray_value = class_id_to_gray[class_id]
        pred_label[mask] = gray_value

    return pred_label


def save_pred_label(pred_label, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), pred_label)


# ============================================================
# Case evaluation
# ============================================================

def evaluate_case_dice_from_saved_masks(
    dataset_name,
    case_id,
    image_files,
    gt_label_files,
    pred_dir,
    id_to_name,
    class_id_to_gray,
    tracking_stats,
):
    rows = []

    for class_id, class_name in id_to_name.items():
        gray_value = class_id_to_gray[class_id]

        frame_dices = []
        valid_gt_frames = 0

        for image_path, gt_path in zip(image_files, gt_label_files):
            pred_path = pred_dir / f"{image_path.stem}.png"

            gt_label = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            pred_label = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

            if gt_label is None:
                print(f"[Warning] Cannot read GT: {gt_path}")
                continue

            if pred_label is None:
                pred_label = np.zeros_like(gt_label, dtype=np.uint8)

            if pred_label.shape != gt_label.shape:
                pred_label = cv2.resize(
                    pred_label,
                    (gt_label.shape[1], gt_label.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            gt_mask = gt_label == gray_value
            pred_mask = pred_label == gray_value

            if gt_mask.sum() == 0:
                continue

            valid_gt_frames += 1
            frame_dices.append(dice_score(pred_mask, gt_mask))

        if valid_gt_frames == 0:
            mean_dice = np.nan
            std_dice = np.nan
        else:
            mean_dice = float(np.mean(frame_dices))
            std_dice = float(np.std(frame_dices))

        stats = tracking_stats.get(class_id, {})
        det_track_ious = stats.get("det_track_ious", [])

        rows.append({
            "dataset": dataset_name,
            "case_id": case_id,
            "target_class_id": class_id,
            "target_class": class_name,
            "label_gray_value": gray_value,
            "valid_gt_frames": valid_gt_frames,
            "mean_dice": mean_dice,
            "std_dice": std_dice,
            "registered": stats.get("registered", False),
            "registered_frame": stats.get("registered_frame", np.nan),
            "num_detection_frames": stats.get("num_detection_frames", 0),
            "num_registrations": stats.get("num_registrations", 0),
            "num_corrections": stats.get("num_corrections", 0),
            "mean_det_track_iou": float(np.mean(det_track_ious)) if det_track_ious else np.nan,
            "min_det_track_iou": float(np.min(det_track_ious)) if det_track_ious else np.nan,
        })

    return rows


def evaluate_case(
    dataset_name,
    case_id,
    img_root,
    gt_label_root,
    pred_mask_root,
    yolo_model,
    predictor,
    id_to_name,
    class_id_to_gray,
):
    case_img_dir = img_root / case_id
    case_gt_dir = gt_label_root / case_id
    case_pred_dir = pred_mask_root / case_id

    if OVERWRITE_PRED_MASK and case_pred_dir.exists():
        shutil.rmtree(case_pred_dir)
    case_pred_dir.mkdir(parents=True, exist_ok=True)

    if not case_img_dir.exists():
        print(f"[Skip] Missing image dir: {case_img_dir}")
        return [], []

    if not case_gt_dir.exists():
        print(f"[Skip] Missing GT label dir: {case_gt_dir}")
        return [], []

    image_files = collect_image_files(case_img_dir)
    gt_label_files = collect_label_files(case_gt_dir)

    if len(image_files) == 0:
        print(f"[Skip] No images in {case_img_dir}")
        return [], []

    if len(gt_label_files) == 0:
        print(f"[Skip] No GT labels in {case_gt_dir}")
        return [], []

    if len(image_files) != len(gt_label_files):
        print(
            f"[Warning] Count mismatch {dataset_name} case {case_id}: "
            f"images={len(image_files)}, labels={len(gt_label_files)}. "
            f"Will use min length."
        )

    num_frames = min(len(image_files), len(gt_label_files))
    image_files = image_files[:num_frames]
    gt_label_files = gt_label_files[:num_frames]

    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print(f"[Skip] Cannot read first image: {image_files[0]}")
        return [], []

    frame_h, frame_w = first_frame.shape[:2]

    focus_class_ids = set(id_to_name.keys())

    detections_by_frame = precompute_yolo_detections(
        yolo_model=yolo_model,
        image_files=image_files,
        focus_class_ids=focus_class_ids,
    )

    first_detect_frame_idx = None
    for idx, dets in enumerate(detections_by_frame):
        if len(dets) > 0:
            first_detect_frame_idx = idx
            break

    if first_detect_frame_idx is None:
        print(f"[Warning] No focus-target YOLO detections in {dataset_name} case {case_id}.")

        for image_path in image_files:
            empty_pred = np.zeros((frame_h, frame_w), dtype=np.uint8)
            save_pred_label(empty_pred, case_pred_dir / f"{image_path.stem}.png")

        dice_rows = evaluate_case_dice_from_saved_masks(
            dataset_name=dataset_name,
            case_id=case_id,
            image_files=image_files,
            gt_label_files=gt_label_files,
            pred_dir=case_pred_dir,
            id_to_name=id_to_name,
            class_id_to_gray=class_id_to_gray,
            tracking_stats={},
        )
        return dice_rows, []

    inference_state = predictor.init_state(video_path=str(case_img_dir))
    predictor.reset_state(inference_state)

    class_id_to_obj_id = {}
    obj_id_to_class_id = {}
    next_obj_id = 1

    tracking_stats = {
        class_id: {
            "registered": False,
            "registered_frame": None,
            "num_detection_frames": 0,
            "num_corrections": 0,
            "num_registrations": 0,
            "det_track_ious": [],
        }
        for class_id in id_to_name.keys()
    }

    correction_logs = []

    init_dets = detections_by_frame[first_detect_frame_idx]

    for class_id, det in init_dets.items():
        obj_id = next_obj_id
        next_obj_id += 1

        class_id_to_obj_id[class_id] = obj_id
        obj_id_to_class_id[obj_id] = class_id

        sam2_add_box(
            predictor=predictor,
            inference_state=inference_state,
            frame_idx=first_detect_frame_idx,
            obj_id=obj_id,
            box=det["box"],
        )

        tracking_stats[class_id]["registered"] = True
        tracking_stats[class_id]["registered_frame"] = first_detect_frame_idx
        tracking_stats[class_id]["num_registrations"] += 1

        correction_logs.append({
            "dataset": dataset_name,
            "case_id": case_id,
            "frame_idx": first_detect_frame_idx,
            "frame_name": image_files[first_detect_frame_idx].name,
            "class_id": class_id,
            "class_name": id_to_name[class_id],
            "event_type": "register",
            "det_conf": det["conf"],
            "det_track_iou": np.nan,
            "det_bbox": bbox_to_str(det["box"]),
            "track_bbox": "",
            "obj_id": obj_id,
        })

    pred_labels_by_frame = {}

    for frame_idx in range(first_detect_frame_idx):
        pred_labels_by_frame[frame_idx] = np.zeros((frame_h, frame_w), dtype=np.uint8)
        save_pred_label(
            pred_labels_by_frame[frame_idx],
            case_pred_dir / f"{image_files[frame_idx].stem}.png",
        )

    for out_frame_idx, out_obj_ids, out_mask_logits in propagate_video(
        predictor=predictor,
        inference_state=inference_state,
        start_frame_idx=first_detect_frame_idx,
    ):
        out_frame_idx = int(out_frame_idx)

        if out_frame_idx >= num_frames:
            break

        current_masks_by_class = sam2_masks_from_output(
            out_obj_ids=out_obj_ids,
            out_mask_logits=out_mask_logits,
            obj_id_to_class_id=obj_id_to_class_id,
        )

        current_dets = detections_by_frame[out_frame_idx]

        for class_id in current_dets.keys():
            if class_id in tracking_stats:
                tracking_stats[class_id]["num_detection_frames"] += 1

        for class_id, det in current_dets.items():
            det_box = det["box"]
            det_conf = det["conf"]

            if class_id not in class_id_to_obj_id:
                obj_id = next_obj_id
                next_obj_id += 1

                class_id_to_obj_id[class_id] = obj_id
                obj_id_to_class_id[obj_id] = class_id

                ret = sam2_add_box(
                    predictor=predictor,
                    inference_state=inference_state,
                    frame_idx=out_frame_idx,
                    obj_id=obj_id,
                    box=det_box,
                )

                tracking_stats[class_id]["registered"] = True
                tracking_stats[class_id]["registered_frame"] = out_frame_idx
                tracking_stats[class_id]["num_registrations"] += 1

                correction_logs.append({
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "frame_idx": out_frame_idx,
                    "frame_name": image_files[out_frame_idx].name,
                    "class_id": class_id,
                    "class_name": id_to_name[class_id],
                    "event_type": "register_new_target",
                    "det_conf": det_conf,
                    "det_track_iou": np.nan,
                    "det_bbox": bbox_to_str(det_box),
                    "track_bbox": "",
                    "obj_id": obj_id,
                })

                if ret is not None and len(ret) == 3:
                    _, ret_obj_ids, ret_mask_logits = ret
                    current_masks_by_class = sam2_masks_from_output(
                        out_obj_ids=ret_obj_ids,
                        out_mask_logits=ret_mask_logits,
                        obj_id_to_class_id=obj_id_to_class_id,
                    )

                continue

            track_mask = current_masks_by_class.get(class_id, None)
            track_box = mask_to_bbox(track_mask)

            det_track_iou = bbox_iou(det_box, track_box)
            tracking_stats[class_id]["det_track_ious"].append(det_track_iou)

            if det_track_iou < TRACK_IOU_THRES:
                obj_id = class_id_to_obj_id[class_id]

                ret = sam2_add_box(
                    predictor=predictor,
                    inference_state=inference_state,
                    frame_idx=out_frame_idx,
                    obj_id=obj_id,
                    box=det_box,
                )

                tracking_stats[class_id]["num_corrections"] += 1

                correction_logs.append({
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "frame_idx": out_frame_idx,
                    "frame_name": image_files[out_frame_idx].name,
                    "class_id": class_id,
                    "class_name": id_to_name[class_id],
                    "event_type": "correct_tracking",
                    "det_conf": det_conf,
                    "det_track_iou": det_track_iou,
                    "det_bbox": bbox_to_str(det_box),
                    "track_bbox": bbox_to_str(track_box),
                    "obj_id": class_id_to_obj_id[class_id],
                })

                if ret is not None and len(ret) == 3:
                    _, ret_obj_ids, ret_mask_logits = ret
                    current_masks_by_class = sam2_masks_from_output(
                        out_obj_ids=ret_obj_ids,
                        out_mask_logits=ret_mask_logits,
                        obj_id_to_class_id=obj_id_to_class_id,
                    )

        pred_label = compose_pred_label(
            mask_by_class=current_masks_by_class,
            class_id_to_gray=class_id_to_gray,
            frame_shape=(frame_h, frame_w),
        )

        pred_labels_by_frame[out_frame_idx] = pred_label
        save_pred_label(pred_label, case_pred_dir / f"{image_files[out_frame_idx].stem}.png")

    for frame_idx in range(num_frames):
        if frame_idx not in pred_labels_by_frame:
            empty_pred = np.zeros((frame_h, frame_w), dtype=np.uint8)
            pred_labels_by_frame[frame_idx] = empty_pred
            save_pred_label(empty_pred, case_pred_dir / f"{image_files[frame_idx].stem}.png")

    dice_rows = evaluate_case_dice_from_saved_masks(
        dataset_name=dataset_name,
        case_id=case_id,
        image_files=image_files,
        gt_label_files=gt_label_files,
        pred_dir=case_pred_dir,
        id_to_name=id_to_name,
        class_id_to_gray=class_id_to_gray,
        tracking_stats=tracking_stats,
    )

    del inference_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dice_rows, correction_logs


# ============================================================
# Dataset evaluation
# ============================================================

def evaluate_amos():
    dataset_name = DATASET_NAME

    dataset_root = DATA_ROOT / dataset_name
    img_root = dataset_root / "img"
    gt_label_root = dataset_root / "label"

    yolo_det_root = DATA_ROOT / "yolo_det" / dataset_name
    split_file = yolo_det_root / "split_cases.txt"
    data_yaml = yolo_det_root / "dataset.yaml"
    excel_file = dataset_root / f"{dataset_name}.xlsx"

    dataset_run_root = RUN_ROOT / dataset_name
    pred_mask_root = dataset_run_root / "pred_mask"
    excel_output = dataset_run_root / f"{dataset_name}_focus5_egomed_ltc_yolo26m_sam2_base_plus.xlsx"

    dataset_run_root.mkdir(parents=True, exist_ok=True)
    pred_mask_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print("Evaluating Amos focus targets")
    print("=" * 120)
    print(f"Focus targets: {FOCUS_TARGETS}")
    print(f"Image root: {img_root}")
    print(f"GT label root: {gt_label_root}")
    print(f"YOLO weight: {YOLO_WEIGHT}")
    print(f"Split file: {split_file}")
    print(f"Data yaml: {data_yaml}")
    print(f"Excel file: {excel_file}")
    print(f"Pred mask root: {pred_mask_root}")
    print(f"Excel output: {excel_output}")
    print("=" * 120)

    if not YOLO_WEIGHT.exists():
        raise FileNotFoundError(f"YOLO weight not found: {YOLO_WEIGHT}")
    if not split_file.exists():
        raise FileNotFoundError(f"split_cases.txt not found: {split_file}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")

    test_cases = read_split_cases(split_file, target_split="test")
    print(f"Number of test cases: {len(test_cases)}")
    print(f"First 10 test cases: {test_cases[:10]}")

    all_id_to_name = load_yolo_class_names(data_yaml)
    gray_to_name = build_dataset_gray_mapping(excel_file)
    all_class_id_to_gray = build_class_id_to_gray(all_id_to_name, gray_to_name)

    id_to_name, class_id_to_gray = filter_focus_classes(
        id_to_name=all_id_to_name,
        class_id_to_gray=all_class_id_to_gray,
    )

    print("\nFocus class mapping:")
    for class_id, class_name in id_to_name.items():
        print(
            f"  class_id={class_id}, "
            f"class_name={class_name}, "
            f"gray_value={class_id_to_gray[class_id]}"
        )

    print("\nLoading YOLO detector...")
    yolo_model = YOLO(str(YOLO_WEIGHT))

    print("Loading SAM2 video predictor...")
    predictor = init_sam2_predictor()

    all_dice_rows = []
    all_correction_logs = []

    for idx, case_id in enumerate(test_cases):
        print("\n" + "-" * 100)
        print(f"[Amos] [{idx + 1}/{len(test_cases)}] Evaluating case {case_id}")
        print("-" * 100)

        with torch.inference_mode():
            if DEVICE == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dice_rows, correction_logs = evaluate_case(
                        dataset_name=dataset_name,
                        case_id=case_id,
                        img_root=img_root,
                        gt_label_root=gt_label_root,
                        pred_mask_root=pred_mask_root,
                        yolo_model=yolo_model,
                        predictor=predictor,
                        id_to_name=id_to_name,
                        class_id_to_gray=class_id_to_gray,
                    )
            else:
                dice_rows, correction_logs = evaluate_case(
                    dataset_name=dataset_name,
                    case_id=case_id,
                    img_root=img_root,
                    gt_label_root=gt_label_root,
                    pred_mask_root=pred_mask_root,
                    yolo_model=yolo_model,
                    predictor=predictor,
                    id_to_name=id_to_name,
                    class_id_to_gray=class_id_to_gray,
                )

        all_dice_rows.extend(dice_rows)
        all_correction_logs.extend(correction_logs)

    dice_df = pd.DataFrame(all_dice_rows)
    correction_df = pd.DataFrame(all_correction_logs)

    if len(dice_df) > 0:
        valid_df = dice_df[dice_df["valid_gt_frames"] > 0].copy()

        class_summary = (
            valid_df.groupby(
                ["dataset", "target_class_id", "target_class", "label_gray_value"],
                dropna=False,
            )
            .agg(
                mean_dice=("mean_dice", "mean"),
                std_dice=("mean_dice", "std"),
                min_dice=("mean_dice", "min"),
                max_dice=("mean_dice", "max"),
                num_cases=("case_id", "count"),
                mean_valid_gt_frames=("valid_gt_frames", "mean"),
                total_valid_gt_frames=("valid_gt_frames", "sum"),
                registered_rate=("registered", "mean"),
                registered_cases=("registered", "sum"),
                mean_num_detection_frames=("num_detection_frames", "mean"),
                mean_num_corrections=("num_corrections", "mean"),
                total_num_corrections=("num_corrections", "sum"),
                mean_det_track_iou=("mean_det_track_iou", "mean"),
                min_det_track_iou=("min_det_track_iou", "min"),
            )
            .reset_index()
            .sort_values("target_class_id")
        )
    else:
        valid_df = pd.DataFrame()
        class_summary = pd.DataFrame()

    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        dice_df.to_excel(writer, sheet_name="case_class_dice", index=False)
        valid_df.to_excel(writer, sheet_name="valid_case_class_dice", index=False)
        class_summary.to_excel(writer, sheet_name="class_summary", index=False)
        correction_df.to_excel(writer, sheet_name="correction_log", index=False)

    print("\n[Amos focus targets] Done.")
    print(f"Saved Excel: {excel_output}")
    print(f"Saved pred masks: {pred_mask_root}")

    print("\nDataset-level mean Dice:")
    if len(class_summary) > 0:
        print(
            class_summary[
                [
                    "target_class",
                    "label_gray_value",
                    "mean_dice",
                    "std_dice",
                    "num_cases",
                    "registered_rate",
                    "mean_num_detection_frames",
                    "mean_num_corrections",
                ]
            ].to_string(index=False)
        )

    return dice_df, class_summary, correction_df


# ============================================================
# Main
# ============================================================

def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("EgoMed-Agent LTC Evaluation: Amos focus targets")
    print("=" * 120)
    print(f"Device: {DEVICE}")
    print(f"SAM2 checkpoint: {SAM2_CHECKPOINT}")
    print(f"SAM2 config: {SAM2_MODEL_CFG}")
    print(f"DET_CONF_THRES: {DET_CONF_THRES}")
    print(f"TRACK_IOU_THRES: {TRACK_IOU_THRES}")
    print(f"YOLO_IMGSZ: {YOLO_IMGSZ}")
    print(f"Run root: {RUN_ROOT}")
    print("=" * 120)

    evaluate_amos()


if __name__ == "__main__":
    main()