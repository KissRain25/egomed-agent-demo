#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EgoMed-Agent --- Localization-Guided Propagation evaluation (paper Sec. III-C).

This runs the assembled EgoMed-Agent for IEMIS. After Target Confirmation (Sec. III-B)
resolves the user-intended target, Localization-Guided Propagation produces the mask
sequence: the Detection Agent (YOLO26) localizes the confirmed target in each frame, the
Propagation Agent (SAM2, original weights) propagates the mask over time, and Consistency
Evaluation re-initializes the Propagation Agent when the IoU between the Detection Agent
box and the propagated mask falls below the consistency threshold tau2 (TRACK_IOU_THRES).

Fair-comparison schedule (matches the SAM3 / LISA text-prompt schedule):
  A target is not activated before its prompt_activate_frame; earlier detections are ignored.

Consistency Evaluation (re-initialization) rule:
  When a newly activated target appears, or the IoU between the Detection Agent box and the
  propagated mask box falls below tau2, the current propagation segment is stopped, the old
  SAM2 inference state is discarded, and a fresh state is initialized at that frame for all
  currently active targets that have detections.

Outputs:
  - per-frame predicted masks + overlay visualizations
  - per-(case, target) Dice CSV / Excel
  - re-initialization log
  - run.log / error.log / config.json

Run:
  conda activate sam2 && python <this script>   # tau2 is set by TRACK_IOU_THRES at the top
"""

import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import re
import sys
import time
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime

import cv2
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

PROMPT_SCHEDULE_CSV = Path(
    f"{REPO_ROOT}/data/text_prompt_eval/egomed5_test_prompt_schedule.csv"
)

DATASETS = [
    "Amos",
    "CAMUS",
    "ACDC",
    "Montgomery-County-CXR-Set",
    "PolypGen2021_MultiCenterData_v3",
]

YOLO_RUN_ROOT = Path(f"{REPO_ROOT}/runs/yolo26_det")

DATASET_CONFIGS = {
    "Amos": {
        "yolo_weight": YOLO_RUN_ROOT / "Amos_yolo26m_imgsz1024" / "weights" / "best.pt",
    },
    "CAMUS": {
        "yolo_weight": YOLO_RUN_ROOT / "CAMUS_yolo26m_imgsz1024" / "weights" / "best.pt",
    },
    "ACDC": {
        "yolo_weight": YOLO_RUN_ROOT / "ACDC_yolo26m_imgsz1024" / "weights" / "best.pt",
    },
    "Montgomery-County-CXR-Set": {
        "yolo_weight": YOLO_RUN_ROOT / "Montgomery-County-CXR-Set_yolo26m_imgsz1024" / "weights" / "best.pt",
    },
    "PolypGen2021_MultiCenterData_v3": {
        "yolo_weight": YOLO_RUN_ROOT / "PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024" / "weights" / "best.pt",
    },
}

# Same restriction as your previous YOLO+SAM2 eval.
FOCUS_TARGETS_BY_DATASET = {
    "Amos": ["liver", "right kidney", "left kidney", "spleen", "stomach"],
}

SAM2_CHECKPOINT = f"{REPO_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

RUN_ROOT = (
    Path(f"{REPO_ROOT}/runs/eval_yolo26_original_sam2_video_schedule_reset_retrack_correction")
    / "egomed5_yolo26m_original_sam2_online_schedule_reset_retrack_iou03"
)

# Segment cache for true re-tracking after correction.
# On each correction, we create a local frame folder starting at the correction frame:
#   000000.jpg -> original correction frame
#   000001.jpg -> next original frame
# This keeps the semantics of a fresh SAM2 state while avoiding full-video reload from frame 0.
SEGMENT_FRAME_CACHE_ROOT = RUN_ROOT / "segment_frame_cache"
CLEAR_SEGMENT_FRAME_CACHE = True
USE_SYMLINK_FOR_SEGMENT_FRAMES = True

# Set >0 if you want to prevent immediate repeated re-tracking.
# Default 0 preserves immediate correction -> re-track semantics.
REINIT_COOLDOWN_FRAMES = 0

DET_CONF_THRES = 0.3
TRACK_IOU_THRES = 0.3
YOLO_IMGSZ = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OVERWRITE_PRED_MASK = True
SAVE_OVERLAY = True

# Overlay colors:
#   red semi-transparent = predicted mask
#   green contour = GT
#   blue box = YOLO detection
#   yellow box = SAM2 tracked bbox
DRAW_YOLO_BOXES_ON_OVERLAY = True
DRAW_TRACK_BOXES_ON_OVERLAY = True
DRAW_GT_CONTOUR_ON_OVERLAY = True

LOG_ROOT = RUN_ROOT / "logs"
LOG_FILE = LOG_ROOT / "run.log"
ERROR_LOG_FILE = LOG_ROOT / "error.log"
CONFIG_JSON = LOG_ROOT / "config.json"


# ============================================================
# Logging
# ============================================================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_run_logging():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    log_f = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
    err_f = open(ERROR_LOG_FILE, "a", buffering=1, encoding="utf-8")

    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f, err_f)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        print("\n" + "=" * 120, file=sys.stderr)
        print("UNCAUGHT EXCEPTION", file=sys.stderr)
        print("=" * 120, file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

    sys.excepthook = handle_exception

    print("\n" + "=" * 120)
    print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file:       {LOG_FILE}")
    print(f"Error log file: {ERROR_LOG_FILE}")
    print("=" * 120)


def save_config_json():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    cfg = {
        "DATA_ROOT": str(DATA_ROOT),
        "PROMPT_SCHEDULE_CSV": str(PROMPT_SCHEDULE_CSV),
        "DATASETS": DATASETS,
        "YOLO_RUN_ROOT": str(YOLO_RUN_ROOT),
        "DATASET_CONFIGS": {
            k: {kk: str(vv) for kk, vv in v.items()}
            for k, v in DATASET_CONFIGS.items()
        },
        "FOCUS_TARGETS_BY_DATASET": FOCUS_TARGETS_BY_DATASET,
        "SAM2_CHECKPOINT": SAM2_CHECKPOINT,
        "SAM2_MODEL_CFG": SAM2_MODEL_CFG,
        "RUN_ROOT": str(RUN_ROOT),
        "SEGMENT_FRAME_CACHE_ROOT": str(SEGMENT_FRAME_CACHE_ROOT),
        "CLEAR_SEGMENT_FRAME_CACHE": CLEAR_SEGMENT_FRAME_CACHE,
        "USE_SYMLINK_FOR_SEGMENT_FRAMES": USE_SYMLINK_FOR_SEGMENT_FRAMES,
        "REINIT_COOLDOWN_FRAMES": REINIT_COOLDOWN_FRAMES,
        "DET_CONF_THRES": DET_CONF_THRES,
        "TRACK_IOU_THRES": TRACK_IOU_THRES,
        "YOLO_IMGSZ": YOLO_IMGSZ,
        "DEVICE": DEVICE,
        "OVERWRITE_PRED_MASK": OVERWRITE_PRED_MASK,
        "SAVE_OVERLAY": SAVE_OVERLAY,
        "DRAW_YOLO_BOXES_ON_OVERLAY": DRAW_YOLO_BOXES_ON_OVERLAY,
        "DRAW_TRACK_BOXES_ON_OVERLAY": DRAW_TRACK_BOXES_ON_OVERLAY,
        "DRAW_GT_CONTOUR_ON_OVERLAY": DRAW_GT_CONTOUR_ON_OVERLAY,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ============================================================
# General utilities
# ============================================================

def natural_key(path_or_name):
    name = Path(str(path_or_name)).name
    nums = re.findall(r"\d+", name)
    return [int(x) for x in nums] if nums else [name]


def normalize_case_id(case_id):
    case_id = str(case_id).strip()
    return str(int(float(case_id))) if case_id.replace(".", "", 1).isdigit() else case_id


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
                cases.append(normalize_case_id(s))

    return cases


def read_rgb_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_pred_label(pred_label, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), pred_label)


def prepare_segment_frame_dir(image_files, start_idx, dataset_name, case_id, segment_id):
    """
    Build a local segment folder for fresh SAM2 re-tracking.

    SAM2 video predictor indexes frames by local contiguous indices in video_path.
    For a correction at original/global frame g, this creates:
      000000.jpg -> original frame g
      000001.jpg -> original frame g + 1
      ...

    Returns:
      segment_dir, local_to_global

    This prevents each correction from reloading the full original case from frame 0.
    """
    segment_dir = (
        SEGMENT_FRAME_CACHE_ROOT
        / str(dataset_name)
        / str(case_id)
        / f"segment_{int(segment_id):04d}_from_{int(start_idx):06d}"
    )

    if segment_dir.exists():
        shutil.rmtree(segment_dir)
    segment_dir.mkdir(parents=True, exist_ok=True)

    local_to_global = {}

    for local_idx, global_idx in enumerate(range(int(start_idx), len(image_files))):
        src = Path(image_files[global_idx])
        dst = segment_dir / f"{local_idx:06d}.jpg"

        if USE_SYMLINK_FOR_SEGMENT_FRAMES and src.suffix.lower() in [".jpg", ".jpeg"]:
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
            except OSError:
                img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"Cannot read image: {src}")
                cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {src}")
            cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        local_to_global[local_idx] = global_idx

    return segment_dir, local_to_global


def bbox_to_str(box):
    if box is None:
        return ""
    return ",".join([f"{float(x):.2f}" for x in box])


# ============================================================
# Mapping and schedule
# ============================================================

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
                    f"[Warning] gray value {gray_value} conflict in {excel_file.name}: "
                    f"{gray_mapping[gray_value]} vs {cls_name}. Using {gray_mapping[gray_value]}."
                )
                continue
            gray_mapping[gray_value] = cls_name

    return dict(sorted(gray_mapping.items(), key=lambda x: x[0]))


def filter_gray_mapping(dataset_name, gray_mapping):
    focus_targets = FOCUS_TARGETS_BY_DATASET.get(dataset_name, None)
    if focus_targets is None:
        return gray_mapping

    focus_set = set(focus_targets)
    filtered = {
        gray_value: cls_name
        for gray_value, cls_name in gray_mapping.items()
        if cls_name in focus_set
    }

    missing = focus_set - set(filtered.values())
    if missing:
        raise ValueError(f"{dataset_name} focus targets not found in Excel gray mapping: {missing}")

    return filtered


def load_yolo_class_names_from_model(yolo_model):
    names = yolo_model.names
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: v for i, v in enumerate(names)}
    raise ValueError(f"Unsupported yolo_model.names format: {type(names)}")


def build_eval_class_mapping_from_yolo_model(dataset_name, yolo_model, excel_file):
    all_id_to_name = load_yolo_class_names_from_model(yolo_model)

    full_gray_mapping = build_dataset_gray_mapping(excel_file)
    gray_mapping = filter_gray_mapping(dataset_name, full_gray_mapping)

    allowed_class_names = set(gray_mapping.values())

    id_to_name = {
        class_id: class_name
        for class_id, class_name in all_id_to_name.items()
        if class_name in allowed_class_names
    }

    missing = allowed_class_names - set(id_to_name.values())
    if missing:
        raise ValueError(
            f"{dataset_name}: targets not found in YOLO checkpoint names: {missing}. "
            f"YOLO names are: {all_id_to_name}"
        )

    name_to_gray = {
        class_name: gray_value
        for gray_value, class_name in gray_mapping.items()
    }

    class_id_to_gray = {
        class_id: name_to_gray[class_name]
        for class_id, class_name in id_to_name.items()
    }

    return id_to_name, class_id_to_gray


def load_prompt_schedule_for_dataset(dataset_name, id_to_name):
    """
    Returns:
      schedule_by_case[case_id][class_id] = min prompt_activate_frame

    Only classes in id_to_name are retained, so YOLO/SAM2 class mapping stays consistent.
    """
    if not PROMPT_SCHEDULE_CSV.exists():
        raise FileNotFoundError(f"Prompt schedule CSV not found: {PROMPT_SCHEDULE_CSV}")

    df = pd.read_csv(PROMPT_SCHEDULE_CSV)
    required = ["dataset", "case_id", "target_class", "prompt_activate_frame"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in schedule CSV: {col}")

    df["dataset"] = df["dataset"].astype(str)
    df["case_id"] = df["case_id"].astype(str)

    df = df[df["dataset"] == str(dataset_name)].copy()

    name_to_class_id = {v: k for k, v in id_to_name.items()}
    schedule_by_case = {}

    for _, row in df.iterrows():
        case_id = normalize_case_id(row["case_id"])
        target_class = str(row["target_class"]).strip()
        if target_class not in name_to_class_id:
            continue

        class_id = int(name_to_class_id[target_class])
        act = int(row["prompt_activate_frame"])

        if case_id not in schedule_by_case:
            schedule_by_case[case_id] = {}

        if class_id not in schedule_by_case[case_id]:
            schedule_by_case[case_id][class_id] = act
        else:
            schedule_by_case[case_id][class_id] = min(schedule_by_case[case_id][class_id], act)

    return schedule_by_case


def active_class_ids_at_frame(case_schedule, frame_idx):
    return {
        int(class_id)
        for class_id, activate_frame in case_schedule.items()
        if int(activate_frame) <= int(frame_idx)
    }


def filter_detections_by_active(detections, active_class_ids):
    return {
        class_id: det
        for class_id, det in detections.items()
        if class_id in active_class_ids
    }


def find_next_active_detection_frame(detections_by_frame, case_schedule, start_idx):
    for idx in range(start_idx, len(detections_by_frame)):
        active = active_class_ids_at_frame(case_schedule, idx)
        if not active:
            continue
        dets = filter_detections_by_active(detections_by_frame[idx], active)
        if len(dets) > 0:
            return idx
    return None


# ============================================================
# Metrics
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

        if class_id not in detections_by_class or float(conf) > detections_by_class[class_id]["conf"]:
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
        return predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            box=box,
            clear_old_points=True,
        )
    except TypeError:
        return predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            box=box,
        )


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
            start_frame_idx=int(start_frame_idx),
        )
    except TypeError:
        print(
            "[Warning] This SAM2 version does not support start_frame_idx. "
            "Falling back to default propagation."
        )
        return predictor.propagate_in_video(inference_state)


# ============================================================
# Mask and visualization utilities
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

        gray_value = int(class_id_to_gray[class_id])
        pred_label[mask] = np.uint8(gray_value)

    return pred_label


def overlay_prediction(
    image_rgb,
    pred_label,
    gt_label,
    class_id_to_gray,
    id_to_name,
    active_dets=None,
    track_boxes=None,
    output_path=None,
):
    if not SAVE_OVERLAY:
        return

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    vis = image_bgr.copy()

    # Red prediction overlay.
    pred_mask = pred_label > 0
    if pred_mask.any():
        red = np.zeros_like(vis)
        red[:, :, 2] = 255
        vis[pred_mask] = cv2.addWeighted(vis, 0.45, red, 0.55, 0)[pred_mask]

    # Green GT contours.
    if DRAW_GT_CONTOUR_ON_OVERLAY and gt_label is not None:
        for gray_value in sorted(set(class_id_to_gray.values())):
            gt_mask = (gt_label == int(gray_value)).astype(np.uint8)
            if gt_mask.any():
                contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    # Blue YOLO boxes.
    if DRAW_YOLO_BOXES_ON_OVERLAY and active_dets:
        for class_id, det in active_dets.items():
            x1, y1, x2, y2 = det["box"].astype(int).tolist()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                vis,
                f"YOLO {id_to_name.get(class_id, class_id)} {det['conf']:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    # Yellow tracked boxes.
    if DRAW_TRACK_BOXES_ON_OVERLAY and track_boxes:
        for class_id, box in track_boxes.items():
            if box is None:
                continue
            x1, y1, x2, y2 = np.asarray(box).astype(int).tolist()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                vis,
                f"SAM2 {id_to_name.get(class_id, class_id)}",
                (x1, min(vis.shape[0] - 2, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# ============================================================
# Dice evaluation from saved masks
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
        gray_value = int(class_id_to_gray[class_id])

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

            # Same convention as your previous tracking eval:
            # Only frames where this target exists in GT count.
            if gt_mask.sum() == 0:
                continue

            valid_gt_frames += 1
            frame_dices.append(dice_score(pred_mask, gt_mask))

        stats = tracking_stats.get(class_id, {})
        det_track_ious = stats.get("det_track_ious", [])

        if valid_gt_frames == 0:
            mean_dice = np.nan
            std_dice = np.nan
        else:
            mean_dice = float(np.mean(frame_dices))
            std_dice = float(np.std(frame_dices))

        rows.append({
            "dataset": dataset_name,
            "case_id": case_id,
            "target_class_id": int(class_id),
            "target_class": class_name,
            "label_gray_value": gray_value,
            "prompt_activate_frame": stats.get("prompt_activate_frame", np.nan),
            "valid_gt_frames": int(valid_gt_frames),
            "mean_dice": mean_dice,
            "std_dice": std_dice,
            "registered": bool(stats.get("registered", False)),
            "registered_frame": stats.get("registered_frame", np.nan),
            "num_detection_frames_before_activation_ignored": int(stats.get("num_detection_frames_before_activation_ignored", 0)),
            "num_detection_frames_after_activation": int(stats.get("num_detection_frames_after_activation", 0)),
            "num_registrations": int(stats.get("num_registrations", 0)),
            "num_reinitializations": int(stats.get("num_reinitializations", 0)),
            "mean_det_track_iou": float(np.mean(det_track_ious)) if det_track_ious else np.nan,
            "min_det_track_iou": float(np.min(det_track_ious)) if det_track_ious else np.nan,
        })

    return rows


# ============================================================
# Case evaluation
# ============================================================

def evaluate_case(
    dataset_name,
    case_id,
    img_root,
    gt_label_root,
    pred_mask_root,
    overlay_root,
    yolo_model,
    predictor,
    id_to_name,
    class_id_to_gray,
    case_schedule,
):
    case_img_dir = img_root / case_id
    case_gt_dir = gt_label_root / case_id
    case_pred_dir = pred_mask_root / case_id
    case_overlay_dir = overlay_root / case_id

    if OVERWRITE_PRED_MASK and case_pred_dir.exists():
        shutil.rmtree(case_pred_dir)
    case_pred_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_OVERLAY:
        if OVERWRITE_PRED_MASK and case_overlay_dir.exists():
            shutil.rmtree(case_overlay_dir)
        case_overlay_dir.mkdir(parents=True, exist_ok=True)

    if not case_img_dir.exists():
        print(f"[Skip] Missing image dir: {case_img_dir}")
        return [], []

    if not case_gt_dir.exists():
        print(f"[Skip] Missing GT label dir: {case_gt_dir}")
        return [], []

    image_files = collect_image_files(case_img_dir)
    gt_label_files = collect_label_files(case_gt_dir)

    if len(image_files) == 0 or len(gt_label_files) == 0:
        print(f"[Skip] Empty images or labels in {dataset_name} case {case_id}")
        return [], []

    if len(image_files) != len(gt_label_files):
        print(
            f"[Warning] Count mismatch {dataset_name} case {case_id}: "
            f"images={len(image_files)}, labels={len(gt_label_files)}. Using min length."
        )

    num_frames = min(len(image_files), len(gt_label_files))
    image_files = image_files[:num_frames]
    gt_label_files = gt_label_files[:num_frames]

    first_frame = cv2.imread(str(image_files[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        print(f"[Skip] Cannot read first image: {image_files[0]}")
        return [], []

    frame_h, frame_w = first_frame.shape[:2]
    empty_label = np.zeros((frame_h, frame_w), dtype=np.uint8)

    focus_class_ids = set(id_to_name.keys())

    # If no schedule is found for this case, no class is activated. This is intentional:
    # the method must follow the same online schedule as SAM3/LISA text-prompt tests.
    if len(case_schedule) == 0:
        print(f"[Warning] No prompt schedule for {dataset_name} case {case_id}. Saving all-empty masks.")

    tracking_stats = {
        class_id: {
            "prompt_activate_frame": case_schedule.get(class_id, np.nan),
            "registered": False,
            "registered_frame": np.nan,
            "num_detection_frames_before_activation_ignored": 0,
            "num_detection_frames_after_activation": 0,
            "num_registrations": 0,
            "num_reinitializations": 0,
            "det_track_ious": [],
        }
        for class_id in id_to_name.keys()
    }

    # 1. Precompute YOLO detections for the whole case.
    detections_by_frame = precompute_yolo_detections(
        yolo_model=yolo_model,
        image_files=image_files,
        focus_class_ids=focus_class_ids,
    )

    # Count ignored/active detection frames.
    for frame_idx, dets in enumerate(detections_by_frame):
        active = active_class_ids_at_frame(case_schedule, frame_idx)
        for class_id in dets.keys():
            if class_id not in tracking_stats:
                continue
            if class_id in active:
                tracking_stats[class_id]["num_detection_frames_after_activation"] += 1
            else:
                tracking_stats[class_id]["num_detection_frames_before_activation_ignored"] += 1

    first_start_idx = find_next_active_detection_frame(detections_by_frame, case_schedule, 0)
    correction_logs = []

    if first_start_idx is None:
        print(f"[Warning] No active YOLO detections in {dataset_name} case {case_id}. Saving empty masks.")

        for frame_idx, image_path in enumerate(image_files):
            save_pred_label(empty_label, case_pred_dir / f"{image_path.stem}.png")
            if SAVE_OVERLAY:
                image_rgb = read_rgb_image(image_path)
                gt_label = cv2.imread(str(gt_label_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
                overlay_prediction(
                    image_rgb=image_rgb,
                    pred_label=empty_label,
                    gt_label=gt_label,
                    class_id_to_gray=class_id_to_gray,
                    id_to_name=id_to_name,
                    active_dets={},
                    track_boxes={},
                    output_path=case_overlay_dir / f"{image_path.stem}.jpg",
                )

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
        return dice_rows, correction_logs

    # Save empty predictions before the first active detection frame.
    for frame_idx in range(first_start_idx):
        save_pred_label(empty_label, case_pred_dir / f"{image_files[frame_idx].stem}.png")
        if SAVE_OVERLAY:
            image_rgb = read_rgb_image(image_files[frame_idx])
            gt_label = cv2.imread(str(gt_label_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
            active = active_class_ids_at_frame(case_schedule, frame_idx)
            active_dets = filter_detections_by_active(detections_by_frame[frame_idx], active)
            overlay_prediction(
                image_rgb=image_rgb,
                pred_label=empty_label,
                gt_label=gt_label,
                class_id_to_gray=class_id_to_gray,
                id_to_name=id_to_name,
                active_dets=active_dets,
                track_boxes={},
                output_path=case_overlay_dir / f"{image_files[frame_idx].stem}.jpg",
            )

    # Load all video frames only once for this case.
    # Later corrections/retracking use reset_state(inference_state), not init_state(),
    # so JPEG frames are not reloaded after every correction.
    print(f"[{dataset_name} case {case_id}] Loading SAM2 frames once for this case...")
    inference_state = predictor.init_state(video_path=str(case_img_dir))
    predictor.reset_state(inference_state)

    current_start_idx = first_start_idx
    segment_id = 0

    while current_start_idx is not None and current_start_idx < num_frames:
        segment_id += 1

        active_at_start = active_class_ids_at_frame(case_schedule, current_start_idx)
        start_dets = filter_detections_by_active(detections_by_frame[current_start_idx], active_at_start)

        if len(start_dets) == 0:
            next_idx = find_next_active_detection_frame(detections_by_frame, case_schedule, current_start_idx + 1)
            end_idx = next_idx if next_idx is not None else num_frames
            for frame_idx in range(current_start_idx, end_idx):
                pred_path = case_pred_dir / f"{image_files[frame_idx].stem}.png"
                if not pred_path.exists():
                    save_pred_label(empty_label, pred_path)
                    if SAVE_OVERLAY:
                        image_rgb = read_rgb_image(image_files[frame_idx])
                        gt_label = cv2.imread(str(gt_label_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
                        active = active_class_ids_at_frame(case_schedule, frame_idx)
                        active_dets = filter_detections_by_active(detections_by_frame[frame_idx], active)
                        overlay_prediction(
                            image_rgb=image_rgb,
                            pred_label=empty_label,
                            gt_label=gt_label,
                            class_id_to_gray=class_id_to_gray,
                            id_to_name=id_to_name,
                            active_dets=active_dets,
                            track_boxes={},
                            output_path=case_overlay_dir / f"{image_files[frame_idx].stem}.jpg",
                        )
            current_start_idx = next_idx
            continue

        print(
            f"[{dataset_name} case {case_id}] "
            f"Start segment {segment_id} at frame {current_start_idx}. "
            f"Active classes: {[id_to_name[c] for c in sorted(active_at_start) if c in id_to_name]}. "
            f"Registered detections: {[id_to_name[c] for c in sorted(start_dets) if c in id_to_name]}."
        )

        # Reset tracking state inside the same already-loaded inference_state.
        # This is the real correction-after-retracking behavior without reloading frames.
        predictor.reset_state(inference_state)

        class_id_to_obj_id = {}
        obj_id_to_class_id = {}
        next_obj_id = 1

        # Register only active classes with YOLO detections.
        for class_id, det in start_dets.items():
            obj_id = next_obj_id
            next_obj_id += 1

            class_id_to_obj_id[class_id] = obj_id
            obj_id_to_class_id[obj_id] = class_id

            sam2_add_box(
                predictor=predictor,
                inference_state=inference_state,
                frame_idx=current_start_idx,
                obj_id=obj_id,
                box=det["box"],
            )

            if not tracking_stats[class_id]["registered"]:
                tracking_stats[class_id]["registered"] = True
                tracking_stats[class_id]["registered_frame"] = current_start_idx

            tracking_stats[class_id]["num_registrations"] += 1
            if segment_id > 1:
                tracking_stats[class_id]["num_reinitializations"] += 1

            correction_logs.append({
                "dataset": dataset_name,
                "case_id": case_id,
                "segment_id": segment_id,
                "frame_idx": current_start_idx,
                "frame_name": image_files[current_start_idx].name,
                "class_id": int(class_id),
                "class_name": id_to_name[class_id],
                "prompt_activate_frame": case_schedule.get(class_id, np.nan),
                "event_type": "initialize" if segment_id == 1 else "reset_retrack",
                "det_conf": det["conf"],
                "det_track_iou": np.nan,
                "det_bbox": bbox_to_str(det["box"]),
                "track_bbox": "",
                "obj_id": int(obj_id),
                "note": "reset_state on existing inference_state; frames are not reloaded",
            })

        next_reinit_idx = None

        for out_frame_idx, out_obj_ids, out_mask_logits in propagate_video(
            predictor=predictor,
            inference_state=inference_state,
            start_frame_idx=current_start_idx,
        ):
            out_frame_idx = int(out_frame_idx)
            if out_frame_idx >= num_frames:
                break

            active_now = active_class_ids_at_frame(case_schedule, out_frame_idx)
            current_dets = filter_detections_by_active(detections_by_frame[out_frame_idx], active_now)

            current_masks_by_class = sam2_masks_from_output(
                out_obj_ids=out_obj_ids,
                out_mask_logits=out_mask_logits,
                obj_id_to_class_id=obj_id_to_class_id,
            )

            track_boxes = {
                class_id: mask_to_bbox(mask)
                for class_id, mask in current_masks_by_class.items()
            }

            trigger_reinit = False
            trigger_info = None

            if out_frame_idx > current_start_idx and (out_frame_idx - current_start_idx) >= REINIT_COOLDOWN_FRAMES:
                # A newly activated target or an active-but-unregistered detected target appears.
                for class_id, det in current_dets.items():
                    if class_id not in class_id_to_obj_id:
                        trigger_reinit = True
                        trigger_info = {
                            "class_id": class_id,
                            "class_name": id_to_name[class_id],
                            "event_type": "retrack_new_active_target",
                            "det_conf": det["conf"],
                            "det_bbox": det["box"],
                            "track_bbox": None,
                            "det_track_iou": np.nan,
                        }
                        break

                # Existing active target mismatch.
                if not trigger_reinit:
                    for class_id, det in current_dets.items():
                        track_box = track_boxes.get(class_id, None)
                        det_track_iou = bbox_iou(det["box"], track_box)

                        if class_id in tracking_stats:
                            tracking_stats[class_id]["det_track_ious"].append(det_track_iou)

                        if det_track_iou < TRACK_IOU_THRES:
                            trigger_reinit = True
                            trigger_info = {
                                "class_id": class_id,
                                "class_name": id_to_name[class_id],
                                "event_type": "retrack_low_iou",
                                "det_conf": det["conf"],
                                "det_bbox": det["box"],
                                "track_bbox": track_box,
                                "det_track_iou": det_track_iou,
                            }
                            break

            if trigger_reinit:
                next_reinit_idx = out_frame_idx

                correction_logs.append({
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "segment_id": segment_id,
                    "frame_idx": out_frame_idx,
                    "frame_name": image_files[out_frame_idx].name,
                    "class_id": int(trigger_info["class_id"]),
                    "class_name": trigger_info["class_name"],
                    "prompt_activate_frame": case_schedule.get(trigger_info["class_id"], np.nan),
                    "event_type": trigger_info["event_type"],
                    "det_conf": trigger_info["det_conf"],
                    "det_track_iou": trigger_info["det_track_iou"],
                    "det_bbox": bbox_to_str(trigger_info["det_bbox"]),
                    "track_bbox": bbox_to_str(trigger_info["track_bbox"]),
                    "obj_id": int(class_id_to_obj_id.get(trigger_info["class_id"], -1)),
                    "note": "trigger reset_state retracking; current frame will be predicted by next segment",
                })

                print(
                    f"[{dataset_name} case {case_id}] Retrack triggered at frame {out_frame_idx}: "
                    f"{trigger_info['event_type']} / {trigger_info['class_name']}."
                )
                break

            pred_label = compose_pred_label(
                mask_by_class=current_masks_by_class,
                class_id_to_gray=class_id_to_gray,
                frame_shape=(frame_h, frame_w),
            )

            save_pred_label(pred_label, case_pred_dir / f"{image_files[out_frame_idx].stem}.png")

            if SAVE_OVERLAY:
                image_rgb = read_rgb_image(image_files[out_frame_idx])
                gt_label = cv2.imread(str(gt_label_files[out_frame_idx]), cv2.IMREAD_GRAYSCALE)
                overlay_prediction(
                    image_rgb=image_rgb,
                    pred_label=pred_label,
                    gt_label=gt_label,
                    class_id_to_gray=class_id_to_gray,
                    id_to_name=id_to_name,
                    active_dets=current_dets,
                    track_boxes=track_boxes,
                    output_path=case_overlay_dir / f"{image_files[out_frame_idx].stem}.jpg",
                )

        current_start_idx = next_reinit_idx

    try:
        del inference_state
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Fill any unsaved frames.
    for frame_idx, image_path in enumerate(image_files):
        pred_path = case_pred_dir / f"{image_path.stem}.png"
        if not pred_path.exists():
            save_pred_label(empty_label, pred_path)
            if SAVE_OVERLAY:
                image_rgb = read_rgb_image(image_path)
                gt_label = cv2.imread(str(gt_label_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
                active = active_class_ids_at_frame(case_schedule, frame_idx)
                active_dets = filter_detections_by_active(detections_by_frame[frame_idx], active)
                overlay_prediction(
                    image_rgb=image_rgb,
                    pred_label=empty_label,
                    gt_label=gt_label,
                    class_id_to_gray=class_id_to_gray,
                    id_to_name=id_to_name,
                    active_dets=active_dets,
                    track_boxes={},
                    output_path=case_overlay_dir / f"{image_path.stem}.jpg",
                )

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

    return dice_rows, correction_logs


# ============================================================
# Dataset evaluation
# ============================================================

def evaluate_dataset(dataset_name, predictor):
    dataset_root = DATA_ROOT / dataset_name
    img_root = dataset_root / "img"
    gt_label_root = dataset_root / "label"

    yolo_det_root = DATA_ROOT / "yolo_det" / dataset_name
    split_file = yolo_det_root / "split_cases.txt"
    excel_file = dataset_root / f"{dataset_name}.xlsx"

    yolo_weight = DATASET_CONFIGS[dataset_name]["yolo_weight"]

    dataset_run_root = RUN_ROOT / dataset_name
    pred_mask_root = dataset_run_root / "pred_mask"
    overlay_root = dataset_run_root / "overlay"
    excel_output = dataset_run_root / f"{dataset_name}_yolo26m_original_sam2_online_schedule_reset_retrack_iou06.xlsx"

    dataset_run_root.mkdir(parents=True, exist_ok=True)
    pred_mask_root.mkdir(parents=True, exist_ok=True)
    if SAVE_OVERLAY:
        overlay_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print(f"Evaluating {dataset_name}: schedule-aware YOLO + original SAM2 reset-state retrack correction")
    print("=" * 120)
    print(f"Image root:       {img_root}")
    print(f"GT label root:    {gt_label_root}")
    print(f"YOLO weight:      {yolo_weight}")
    print(f"Split file:       {split_file}")
    print(f"Excel file:       {excel_file}")
    print(f"Pred mask root:   {pred_mask_root}")
    print(f"Overlay root:     {overlay_root}")
    print(f"Excel output:     {excel_output}")
    print("=" * 120)

    if not img_root.exists():
        raise FileNotFoundError(f"Image root not found: {img_root}")
    if not gt_label_root.exists():
        raise FileNotFoundError(f"GT label root not found: {gt_label_root}")
    if not yolo_weight.exists():
        raise FileNotFoundError(f"YOLO weight not found: {yolo_weight}")
    if not split_file.exists():
        raise FileNotFoundError(f"split_cases.txt not found: {split_file}")
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")

    test_cases = read_split_cases(split_file, target_split="test")
    print(f"Number of test cases: {len(test_cases)}")
    print(f"First 10 test cases: {test_cases[:10]}")

    print("\nLoading YOLO detector for this dataset...")
    yolo_model = YOLO(str(yolo_weight))

    id_to_name, class_id_to_gray = build_eval_class_mapping_from_yolo_model(
        dataset_name=dataset_name,
        yolo_model=yolo_model,
        excel_file=excel_file,
    )

    print("\nEvaluation class mapping:")
    for class_id, class_name in id_to_name.items():
        print(
            f"  class_id={class_id}, "
            f"class_name={class_name}, "
            f"gray_value={class_id_to_gray[class_id]}"
        )

    schedule_by_case = load_prompt_schedule_for_dataset(dataset_name, id_to_name)

    print("\nSchedule examples:")
    for k in list(schedule_by_case.keys())[:3]:
        print(f"  case {k}: {schedule_by_case[k]}")

    all_dice_rows = []
    all_correction_logs = []

    for idx, case_id in enumerate(test_cases):
        print("\n" + "-" * 100)
        print(f"[{dataset_name}] [{idx + 1}/{len(test_cases)}] Evaluating case {case_id}")
        print("-" * 100)

        case_schedule = schedule_by_case.get(str(case_id), {})
        if len(case_schedule) == 0:
            print(f"[Warning] No schedule found for {dataset_name} case {case_id}.")

        with torch.inference_mode():
            if DEVICE == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dice_rows, correction_logs = evaluate_case(
                        dataset_name=dataset_name,
                        case_id=case_id,
                        img_root=img_root,
                        gt_label_root=gt_label_root,
                        pred_mask_root=pred_mask_root,
                        overlay_root=overlay_root,
                        yolo_model=yolo_model,
                        predictor=predictor,
                        id_to_name=id_to_name,
                        class_id_to_gray=class_id_to_gray,
                        case_schedule=case_schedule,
                    )
            else:
                dice_rows, correction_logs = evaluate_case(
                    dataset_name=dataset_name,
                    case_id=case_id,
                    img_root=img_root,
                    gt_label_root=gt_label_root,
                    pred_mask_root=pred_mask_root,
                    overlay_root=overlay_root,
                    yolo_model=yolo_model,
                    predictor=predictor,
                    id_to_name=id_to_name,
                    class_id_to_gray=class_id_to_gray,
                    case_schedule=case_schedule,
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
                num_cases=("case_id", "count"),
                total_valid_gt_frames=("valid_gt_frames", "sum"),
                mean_num_detection_frames_after_activation=("num_detection_frames_after_activation", "mean"),
                total_num_detection_frames_after_activation=("num_detection_frames_after_activation", "sum"),
                mean_num_detection_frames_before_activation_ignored=("num_detection_frames_before_activation_ignored", "mean"),
                total_num_detection_frames_before_activation_ignored=("num_detection_frames_before_activation_ignored", "sum"),
                mean_num_registrations=("num_registrations", "mean"),
                total_num_registrations=("num_registrations", "sum"),
                mean_num_reinitializations=("num_reinitializations", "mean"),
                total_num_reinitializations=("num_reinitializations", "sum"),
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
        correction_df.to_excel(writer, sheet_name="reinit_correction_log", index=False)

    dice_df.to_csv(dataset_run_root / "case_class_dice.csv", index=False)
    valid_df.to_csv(dataset_run_root / "valid_case_class_dice.csv", index=False)
    class_summary.to_csv(dataset_run_root / "class_summary.csv", index=False)
    correction_df.to_csv(dataset_run_root / "reinit_correction_log.csv", index=False)

    print(f"\n[{dataset_name}] Done.")
    print(f"Saved Excel: {excel_output}")
    print(f"Saved pred masks: {pred_mask_root}")
    print(f"Saved overlays: {overlay_root}")

    if len(class_summary) > 0:
        print("\nDataset-level mean Dice:")
        display_cols = [
            "target_class",
            "label_gray_value",
            "mean_dice",
            "std_dice",
            "num_cases",
            "mean_num_detection_frames_after_activation",
            "mean_num_registrations",
            "mean_num_reinitializations",
            "mean_det_track_iou",
        ]
        print(class_summary[display_cols].to_string(index=False))

    return dice_df, class_summary, correction_df


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if CLEAR_SEGMENT_FRAME_CACHE and SEGMENT_FRAME_CACHE_ROOT.exists():
        shutil.rmtree(SEGMENT_FRAME_CACHE_ROOT)
    SEGMENT_FRAME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    save_config_json()

    print("=" * 120)
    print("YOLO26-det + original SAM2 Video Predictor Evaluation")
    print("Mode: schedule-aware online prompt + correction-triggered fresh re-tracking")
    print("=" * 120)
    print(f"Datasets:             {DATASETS}")
    print(f"Prompt schedule CSV:  {PROMPT_SCHEDULE_CSV}")
    print(f"Device:               {DEVICE}")
    print(f"YOLO run root:        {YOLO_RUN_ROOT}")
    print(f"SAM2 checkpoint:      {SAM2_CHECKPOINT}")
    print(f"SAM2 config:          {SAM2_MODEL_CFG}")
    print(f"DET_CONF_THRES:       {DET_CONF_THRES}")
    print(f"TRACK_IOU_THRES:      {TRACK_IOU_THRES}")
    print(f"YOLO_IMGSZ:           {YOLO_IMGSZ}")
    print(f"Run root:             {RUN_ROOT}")
    print(f"Segment cache root:   {SEGMENT_FRAME_CACHE_ROOT}")
    print(f"Reinit cooldown:      {REINIT_COOLDOWN_FRAMES}")
    print(f"Save overlay:         {SAVE_OVERLAY}")
    print(f"Run log:              {LOG_FILE}")
    print(f"Error log:            {ERROR_LOG_FILE}")
    print("=" * 120)

    if not PROMPT_SCHEDULE_CSV.exists():
        raise FileNotFoundError(f"Prompt schedule CSV not found: {PROMPT_SCHEDULE_CSV}")
    if not Path(SAM2_CHECKPOINT).exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}")

    print("\nLoading SAM2 (original weights) for the Propagation Agent...")
    predictor = init_sam2_predictor()
    print("SAM2 video predictor loaded.")

    all_dataset_class_summaries = []
    all_dataset_correction_logs = []

    for dataset_name in DATASETS:
        dice_df, class_summary, correction_df = evaluate_dataset(
            dataset_name=dataset_name,
            predictor=predictor,
        )

        if class_summary is not None and len(class_summary) > 0:
            all_dataset_class_summaries.append(class_summary)

        if correction_df is not None and len(correction_df) > 0:
            all_dataset_correction_logs.append(correction_df)

    all_summary = pd.concat(all_dataset_class_summaries, ignore_index=True) if all_dataset_class_summaries else pd.DataFrame()
    all_corrections = pd.concat(all_dataset_correction_logs, ignore_index=True) if all_dataset_correction_logs else pd.DataFrame()

    all_summary_path = RUN_ROOT / "egomed5_all_datasets_class_summary.xlsx"
    all_summary_csv = RUN_ROOT / "egomed5_all_datasets_class_summary.csv"
    all_correction_csv = RUN_ROOT / "egomed5_all_datasets_reinit_correction_log.csv"

    all_summary.to_excel(all_summary_path, index=False)
    all_summary.to_csv(all_summary_csv, index=False)
    all_corrections.to_csv(all_correction_csv, index=False)

    elapsed = time.time() - start_time

    print("\n" + "=" * 120)
    print("All datasets done.")
    print(f"Elapsed seconds:          {elapsed:.2f}")
    print(f"All summary Excel:        {all_summary_path}")
    print(f"All summary CSV:          {all_summary_csv}")
    print(f"All correction log CSV:   {all_correction_csv}")
    print(f"Run log:                  {LOG_FILE}")
    print(f"Error log:                {ERROR_LOG_FILE}")
    print("=" * 120)

    if len(all_summary) > 0:
        print("\nAll-dataset class summary:")
        display_cols = [
            "dataset",
            "target_class",
            "label_gray_value",
            "mean_dice",
            "std_dice",
            "num_cases",
            "mean_num_detection_frames_after_activation",
            "mean_num_registrations",
            "mean_num_reinitializations",
            "mean_det_track_iou",
        ]
        existing = [c for c in display_cols if c in all_summary.columns]
        print(all_summary[existing].to_string(index=False))


if __name__ == "__main__":
    setup_run_logging()
    main()
