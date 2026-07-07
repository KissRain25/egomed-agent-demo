#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MedSAM3 image-level text-prompt evaluation on EgoMed5 test prompt CSV.

This script uses MedSAM3's official infer_sam.py wrapper:
  SAM3 base checkpoint + MedSAM3 LoRA weights

Each prompt-frame row is evaluated independently:
  image_path + row["prompt"] -> MedSAM3 -> selected pred mask
  compare with label_path + target_gray_value

Important:
  For fair comparison with LISA/SAM3, this script uses the exact text prompt
  from the CSV by default:
      MEDSAM3_TEXT_SOURCE = "prompt"

Run:
  conda activate sam2
  cd $EGOMED_ROOT
  python eval_medsam3_image_text_prompt_egomed5_save_outputs.py
"""

import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
EXT_ROOT = _Path(_os.environ.get("EGOMED_EXT_ROOT", REPO_ROOT.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Config
# =============================================================================

MEDSAM3_REPO_ROOT = Path(f"{EXT_ROOT}/MedSAM3")
CONFIG_PATH = MEDSAM3_REPO_ROOT / "configs/full_lora_config.yaml"
LORA_WEIGHTS = MEDSAM3_REPO_ROOT / "checkpoints/best_lora_weights.pt"

# You already patched $EGOMED_EXT_ROOT/MedSAM3/infer_sam.py to use:
#   checkpoint_path=f"{EXT_ROOT}/sam3/checkpoints/sam3.pt"
# instead of load_from_HF=True.
SAM3_BASE_CHECKPOINT = Path(f"{EXT_ROOT}/sam3/checkpoints/sam3.pt")

PROMPT_CSV = Path(
    f"{REPO_ROOT}/data/text_prompt_eval/"
    "egomed5_test_text_prompts_online_schedule.csv"
)

RUN_ROOT = (
    Path(f"{REPO_ROOT}/runs/eval_medsam3_text_prompt")
    / "egomed5_medsam3_image_text_prompt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IMPORTANT FOR FAIR COMPARISON:
#   "prompt": use the exact prompt from CSV, e.g. "segment left ventricle"
#   "target_class": use only target_class, e.g. "left ventricle" -- for ablation only
MEDSAM3_TEXT_SOURCE = "prompt"

# These match your successful demo command.
DETECTION_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
RESOLUTION = 1008

# Select one prediction from MedSAM3 output without using GT.
SELECT_MASK_MODE = "top_score"

# Threshold masks. If MedSAM3 mask values look like logits, code uses > 0.
# If values look like probabilities in [0, 1], code uses > MASK_PROB_THRESHOLD.
MASK_PROB_THRESHOLD = 0.5

# Cache duplicate image+prompt predictions within the same run.
CACHE_PREDICTIONS = False

SAVE_PRED_MASK = True
SAVE_PRED_OVERLAY = False
SAVE_EMPTY_MASKS = False

PRED_MASK_ROOT = RUN_ROOT / "pred_mask"
PRED_OVERLAY_ROOT = RUN_ROOT / "pred_overlay"

SUMMARY_ONLY_GT_EXISTS = True

LOG_ROOT = RUN_ROOT / "logs"
LOG_FILE = LOG_ROOT / "run.log"
ERROR_LOG_FILE = LOG_ROOT / "error.log"
CONFIG_JSON = LOG_ROOT / "config.json"


# =============================================================================
# Logging
# =============================================================================

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
        "MEDSAM3_REPO_ROOT": str(MEDSAM3_REPO_ROOT),
        "CONFIG_PATH": str(CONFIG_PATH),
        "LORA_WEIGHTS": str(LORA_WEIGHTS),
        "SAM3_BASE_CHECKPOINT": str(SAM3_BASE_CHECKPOINT),
        "PROMPT_CSV": str(PROMPT_CSV),
        "RUN_ROOT": str(RUN_ROOT),
        "DEVICE": DEVICE,
        "MEDSAM3_TEXT_SOURCE": MEDSAM3_TEXT_SOURCE,
        "DETECTION_THRESHOLD": DETECTION_THRESHOLD,
        "NMS_IOU_THRESHOLD": NMS_IOU_THRESHOLD,
        "RESOLUTION": RESOLUTION,
        "SELECT_MASK_MODE": SELECT_MASK_MODE,
        "MASK_PROB_THRESHOLD": MASK_PROB_THRESHOLD,
        "CACHE_PREDICTIONS": CACHE_PREDICTIONS,
        "SAVE_PRED_MASK": SAVE_PRED_MASK,
        "SAVE_PRED_OVERLAY": SAVE_PRED_OVERLAY,
        "SAVE_EMPTY_MASKS": SAVE_EMPTY_MASKS,
        "PRED_MASK_ROOT": str(PRED_MASK_ROOT),
        "PRED_OVERLAY_ROOT": str(PRED_OVERLAY_ROOT),
        "SUMMARY_ONLY_GT_EXISTS": SUMMARY_ONLY_GT_EXISTS,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# =============================================================================
# Import MedSAM3 official wrapper
# =============================================================================

if str(MEDSAM3_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDSAM3_REPO_ROOT))

from infer_sam import SAM3LoRAInference


# =============================================================================
# Metrics and utilities
# =============================================================================

def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def normalize_medsam3_text(row: pd.Series) -> str:
    if MEDSAM3_TEXT_SOURCE == "target_class":
        text = str(row["target_class"]).strip()
    else:
        text = str(row["prompt"]).strip()

    if len(text) == 0:
        text = str(row.get("target_class", "object")).strip()

    return text


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps=1e-6) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    pred_sum = int(pred_mask.sum())
    gt_sum = int(gt_mask.sum())

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    return float((2.0 * inter + eps) / (pred_sum + gt_sum + eps))


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps=1e-6) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    union = int(np.logical_or(pred_mask, gt_mask).sum())
    if union == 0:
        return 1.0

    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    return float((inter + eps) / (union + eps))


def mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_to_str(box: Optional[np.ndarray]) -> str:
    if box is None:
        return ""
    box = np.asarray(box).reshape(-1)
    if len(box) != 4 or np.any(np.isnan(box)):
        return ""
    return ",".join([f"{float(x):.2f}" for x in box])


def bbox_area(box: Optional[np.ndarray]) -> float:
    if box is None:
        return 0.0
    box = np.asarray(box).reshape(-1)
    if len(box) != 4 or np.any(np.isnan(box)):
        return 0.0
    return max(0.0, float(box[2] - box[0] + 1.0)) * max(0.0, float(box[3] - box[1] + 1.0))


def bbox_iou(box1: Optional[np.ndarray], box2: Optional[np.ndarray], eps=1e-6) -> float:
    if box1 is None and box2 is None:
        return 1.0
    if box1 is None or box2 is None:
        return 0.0

    box1 = np.asarray(box1).reshape(-1)
    box2 = np.asarray(box2).reshape(-1)
    if len(box1) != 4 or len(box2) != 4 or np.any(np.isnan(box1)) or np.any(np.isnan(box2)):
        return 0.0

    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1 + 1.0)
    inter_h = max(0.0, y2 - y1 + 1.0)
    inter = inter_w * inter_h

    area1 = bbox_area(box1)
    area2 = bbox_area(box2)
    return float(inter / (area1 + area2 - inter + eps))


def read_gt_mask(label_path: str, gray_value: int) -> np.ndarray:
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"Cannot read label: {label_path}")
    return label == int(gray_value)


def ensure_mask_shape(mask: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask is None:
        return np.zeros((h, w), dtype=bool)

    mask = np.asarray(mask).astype(bool)
    if mask.shape != (h, w):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def safe_name(x) -> str:
    x = str(x)
    for ch in ["/", "\\", ":", " ", "\t", "\n", "|", ",", ";", "\"", "'"]:
        x = x.replace(ch, "_")
    return x


def make_pred_paths(dataset, case_id, frame_idx, frame_name, prompt_id, target_class):
    stem = Path(str(frame_name)).stem
    cls = safe_name(target_class)
    rel = Path(str(dataset)) / str(case_id)

    filename = f"{int(frame_idx):06d}_{stem}_prompt{int(prompt_id):03d}_{cls}.png"
    overlay_filename = f"{int(frame_idx):06d}_{stem}_prompt{int(prompt_id):03d}_{cls}.jpg"

    return (
        PRED_MASK_ROOT / rel / filename,
        PRED_OVERLAY_ROOT / rel / overlay_filename,
    )


def save_mask(mask_save_path: Path, pred_mask: np.ndarray):
    mask_save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_save_path), pred_mask.astype(np.uint8) * 255)


def save_overlay(overlay_path: Path, image_rgb: np.ndarray, pred_mask: np.ndarray, gt_mask=None):
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    pred_mask = ensure_mask_shape(pred_mask, image_rgb.shape[:2])
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    vis = img_bgr.copy()

    if pred_mask.any():
        red = np.zeros_like(vis)
        red[:, :, 2] = 255
        vis[pred_mask] = cv2.addWeighted(vis, 0.45, red, 0.55, 0)[pred_mask]

    if gt_mask is not None and np.asarray(gt_mask).any():
        gt_mask = ensure_mask_shape(gt_mask, image_rgb.shape[:2]).astype(np.uint8)
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    cv2.imwrite(str(overlay_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def load_image_rgb(image_path: str) -> np.ndarray:
    img = Image.open(str(image_path)).convert("RGB")
    return np.asarray(img)


def boxes_cxcywh_to_xyxy(boxes: np.ndarray, image_size_wh: Tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)

    boxes = boxes.reshape(-1, 4)
    w, h = image_size_wh

    if np.nanmax(np.abs(boxes)) <= 1.5:
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    return boxes.astype(np.float32)


def normalize_masks(masks) -> np.ndarray:
    masks = to_numpy(masks)
    if masks is None:
        return np.zeros((0, 1, 1), dtype=np.float32)

    masks = np.asarray(masks)

    if masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim == 3:
        pass
    elif masks.ndim == 4:
        if masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        elif masks.shape[0] == 1:
            masks = masks[0]
        else:
            masks = np.squeeze(masks)
            if masks.ndim == 2:
                masks = masks[None, :, :]
            elif masks.ndim != 3:
                raise ValueError(f"Unsupported mask shape after squeeze: {masks.shape}")
    else:
        raise ValueError(f"Unsupported mask shape: {masks.shape}")

    return masks.astype(np.float32)


def mask_to_bool(mask_values: np.ndarray) -> np.ndarray:
    mask_values = np.asarray(mask_values)

    if mask_values.size == 0:
        return mask_values.astype(bool)

    mn = float(np.nanmin(mask_values))
    mx = float(np.nanmax(mask_values))

    if mn >= 0.0 and mx <= 1.0:
        return mask_values > MASK_PROB_THRESHOLD

    return mask_values > 0.0


def normalize_scores(scores, n: int) -> np.ndarray:
    scores = to_numpy(scores)
    if scores is None:
        return np.full((n,), np.nan, dtype=np.float32)

    scores = np.asarray(scores, dtype=np.float32)

    if scores.ndim == 2:
        scores = scores.max(axis=1)
    else:
        scores = scores.reshape(-1)

    if len(scores) < n:
        scores = np.pad(scores, (0, n - len(scores)), constant_values=np.nan)

    return scores[:n].astype(np.float32)


def get_prompt_prediction(raw_result: Any, prompt_index: int = 0) -> Dict[str, Any]:
    if isinstance(raw_result, dict):
        if prompt_index in raw_result and isinstance(raw_result[prompt_index], dict):
            return raw_result[prompt_index]
        if str(prompt_index) in raw_result and isinstance(raw_result[str(prompt_index)], dict):
            return raw_result[str(prompt_index)]

        if any(k in raw_result for k in ["boxes", "scores", "masks"]):
            return raw_result

        for v in raw_result.values():
            if isinstance(v, dict) and any(k in v for k in ["boxes", "scores", "masks"]):
                return v

    raise ValueError(f"Unsupported MedSAM3 predict result format: {type(raw_result)}")


# =============================================================================
# MedSAM3 predictor wrapper
# =============================================================================

class MedSAM3ImageTextPredictor:
    def __init__(self):
        self.predictor = None
        self.cache: Dict[tuple, Dict[str, Any]] = {}

    def load(self):
        print("\nLoading MedSAM3 model...")
        print(f"  repo root:      {MEDSAM3_REPO_ROOT}")
        print(f"  config:         {CONFIG_PATH}")
        print(f"  LoRA weights:   {LORA_WEIGHTS}")
        print(f"  base SAM3 ckpt: {SAM3_BASE_CHECKPOINT}")
        print(f"  device:         {DEVICE}")
        print(f"  threshold:      {DETECTION_THRESHOLD}")
        print(f"  nms_iou:        {NMS_IOU_THRESHOLD}")
        print(f"  resolution:     {RESOLUTION}")

        if not MEDSAM3_REPO_ROOT.exists():
            raise FileNotFoundError(f"MedSAM3 repo not found: {MEDSAM3_REPO_ROOT}")
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"MedSAM3 config not found: {CONFIG_PATH}")
        if not LORA_WEIGHTS.exists():
            raise FileNotFoundError(f"MedSAM3 LoRA weights not found: {LORA_WEIGHTS}")
        if not SAM3_BASE_CHECKPOINT.exists():
            raise FileNotFoundError(f"Base SAM3 checkpoint not found: {SAM3_BASE_CHECKPOINT}")

        # MedSAM3 infer_sam.py uses some repo-relative asset paths.
        # Switch cwd to the MedSAM3 repo before constructing the wrapper.
        os.chdir(MEDSAM3_REPO_ROOT)

        self.predictor = SAM3LoRAInference(
            config_path=str(CONFIG_PATH),
            weights_path=str(LORA_WEIGHTS),
            resolution=RESOLUTION,
            detection_threshold=DETECTION_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
            device=DEVICE,
        )

        print("MedSAM3 loaded.")

    @torch.no_grad()
    def predict(self, image_path: str, text_prompt: str):
        key = (str(image_path), str(text_prompt))
        if CACHE_PREDICTIONS and key in self.cache:
            return dict(self.cache[key])

        image_rgb = load_image_rgb(image_path)
        h, w = image_rgb.shape[:2]

        raw_result = self.predictor.predict(
            image_path=str(image_path),
            text_prompts=[str(text_prompt)],
        )
        pred = get_prompt_prediction(raw_result, prompt_index=0)

        masks = normalize_masks(pred.get("masks", None))
        n_masks = int(masks.shape[0])

        scores = normalize_scores(pred.get("scores", None), n_masks)
        boxes_raw = to_numpy(pred.get("boxes", None))
        if boxes_raw is None:
            boxes_xyxy = np.full((n_masks, 4), np.nan, dtype=np.float32)
        else:
            boxes_raw = np.asarray(boxes_raw, dtype=np.float32).reshape(-1, 4)
            boxes_xyxy = boxes_cxcywh_to_xyxy(boxes_raw, image_size_wh=(w, h))
            if len(boxes_xyxy) < n_masks:
                pad = np.full((n_masks - len(boxes_xyxy), 4), np.nan, dtype=np.float32)
                boxes_xyxy = np.concatenate([boxes_xyxy, pad], axis=0)
            boxes_xyxy = boxes_xyxy[:n_masks]

        if n_masks == 0:
            selected_idx = -1
            selected_score = np.nan
            selected_box = np.full((4,), np.nan, dtype=np.float32)
            pred_mask = None
        else:
            if SELECT_MASK_MODE == "top_score" and not np.all(np.isnan(scores)):
                selected_idx = int(np.nanargmax(scores))
            else:
                selected_idx = 0

            selected_score = float(scores[selected_idx]) if selected_idx < len(scores) else np.nan
            selected_box = boxes_xyxy[selected_idx] if selected_idx < len(boxes_xyxy) else np.full((4,), np.nan, dtype=np.float32)
            pred_mask = mask_to_bool(masks[selected_idx])

        result = {
            "pred_mask": pred_mask,
            "image_rgb": image_rgb,
            "num_masks": n_masks,
            "selected_idx": selected_idx,
            "selected_score": selected_score,
            "selected_box": selected_box,
            "all_scores": scores,
            "all_boxes": boxes_xyxy,
            "raw_result_type": str(type(raw_result)),
            "pred_keys": list(pred.keys()),
        }

        if CACHE_PREDICTIONS:
            self.cache[key] = dict(result)

        return result


# =============================================================================
# Summary
# =============================================================================

def summarize(frame_df: pd.DataFrame):
    if "error" not in frame_df.columns:
        frame_df["error"] = ""

    metric_df = (
        frame_df[frame_df["gt_exists"] == True].copy()
        if SUMMARY_ONLY_GT_EXISTS
        else frame_df.copy()
    )
    metric_df = metric_df[metric_df["error"].fillna("") == ""]

    if len(metric_df) > 0:
        case_df = (
            metric_df.groupby(
                [
                    "dataset",
                    "case_id",
                    "prompt_type",
                    "prompt",
                    "target_class",
                    "target_gray_value",
                ],
                dropna=False,
            )
            .agg(
                mean_dice=("dice", "mean"),
                std_dice=("dice", "std"),
                mean_mask_iou=("mask_iou", "mean"),
                std_mask_iou=("mask_iou", "std"),
                mean_bbox_iou=("bbox_iou", "mean"),
                mean_medsam3_score=("medsam3_score", "mean"),
                mean_pred_area=("pred_area", "mean"),
                mean_gt_area=("gt_area", "mean"),
                mean_num_masks=("num_masks", "mean"),
                num_gt_exists_frames=("gt_exists", "count"),
            )
            .reset_index()
        )

        class_summary = (
            case_df.groupby(
                ["dataset", "prompt_type", "prompt", "target_class", "target_gray_value"],
                dropna=False,
            )
            .agg(
                mean_dice=("mean_dice", "mean"),
                std_dice=("mean_dice", "std"),
                mean_mask_iou=("mean_mask_iou", "mean"),
                std_mask_iou=("mean_mask_iou", "std"),
                mean_bbox_iou=("mean_bbox_iou", "mean"),
                mean_medsam3_score=("mean_medsam3_score", "mean"),
                mean_pred_area=("mean_pred_area", "mean"),
                mean_gt_area=("mean_gt_area", "mean"),
                mean_num_masks=("mean_num_masks", "mean"),
                num_cases=("case_id", "count"),
                total_gt_exists_frames=("num_gt_exists_frames", "sum"),
            )
            .reset_index()
            .sort_values(["dataset", "prompt_type", "target_gray_value", "prompt"])
        )
    else:
        case_df = pd.DataFrame()
        class_summary = pd.DataFrame()

    absent_df = frame_df[
        (frame_df["gt_exists"] == False)
        & (frame_df["error"].fillna("") == "")
    ].copy()

    if len(absent_df) > 0:
        absent_summary = (
            absent_df.groupby(
                ["dataset", "prompt_type", "prompt", "target_class", "target_gray_value"],
                dropna=False,
            )
            .agg(
                num_absent_frames=("gt_exists", "count"),
                num_absent_false_positive=("absent_false_positive", "sum"),
                num_absent_empty_correct=("absent_empty_correct", "sum"),
                mean_pred_area_absent=("pred_area", "mean"),
            )
            .reset_index()
        )
        absent_summary["absent_false_positive_rate"] = (
            absent_summary["num_absent_false_positive"]
            / absent_summary["num_absent_frames"].clip(lower=1)
        )
        absent_summary["absent_empty_correct_rate"] = (
            absent_summary["num_absent_empty_correct"]
            / absent_summary["num_absent_frames"].clip(lower=1)
        )
    else:
        absent_summary = pd.DataFrame()

    return case_df, class_summary, absent_summary


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = time.time()

    if not PROMPT_CSV.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {PROMPT_CSV}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if SAVE_PRED_MASK:
        PRED_MASK_ROOT.mkdir(parents=True, exist_ok=True)
    if SAVE_PRED_OVERLAY:
        PRED_OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)

    save_config_json()

    print("=" * 120)
    print("MedSAM3 Image Text-Prompt Evaluation on EgoMed5 Test Online Schedule")
    print("=" * 120)
    print(f"Prompt CSV:          {PROMPT_CSV}")
    print(f"MedSAM3 repo root:   {MEDSAM3_REPO_ROOT}")
    print(f"Config:              {CONFIG_PATH}")
    print(f"LoRA weights:        {LORA_WEIGHTS}")
    print(f"Base SAM3 ckpt:      {SAM3_BASE_CHECKPOINT}")
    print(f"Device:              {DEVICE}")
    print(f"Run root:            {RUN_ROOT}")
    print(f"Text source:         {MEDSAM3_TEXT_SOURCE}")
    print(f"Threshold:           {DETECTION_THRESHOLD}")
    print(f"NMS IoU:             {NMS_IOU_THRESHOLD}")
    print(f"Resolution:          {RESOLUTION}")
    print(f"Select mask mode:    {SELECT_MASK_MODE}")
    print(f"Save pred masks:     {SAVE_PRED_MASK}")
    print(f"Pred mask root:      {PRED_MASK_ROOT}")
    print(f"Save overlay:        {SAVE_PRED_OVERLAY}")
    print(f"Overlay root:        {PRED_OVERLAY_ROOT}")
    print(f"Save empty masks:    {SAVE_EMPTY_MASKS}")
    print("=" * 120)

    df = pd.read_csv(PROMPT_CSV)

    required_cols = [
        "dataset",
        "case_id",
        "frame_idx",
        "frame_name",
        "image_path",
        "label_path",
        "prompt",
        "prompt_type",
        "target_class",
        "target_gray_value",
        "case_prompt_id",
        "prompt_activate_frame",
        "gt_exists",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in prompt CSV: {col}")

    df["dataset"] = df["dataset"].astype(str)
    df["case_id"] = df["case_id"].astype(str)

    df = df.sort_values(["dataset", "case_id", "frame_idx", "case_prompt_id"]).reset_index(drop=True)

    predictor = MedSAM3ImageTextPredictor()
    predictor.load()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MedSAM3 image eval"):
        dataset = str(row["dataset"])
        case_id = str(row["case_id"])
        frame_idx = int(row["frame_idx"])
        frame_name = str(row["frame_name"])
        image_path = str(row["image_path"])
        label_path = str(row["label_path"])
        prompt = str(row["prompt"])
        medsam3_text_prompt = normalize_medsam3_text(row)
        prompt_type = str(row["prompt_type"])
        target_class = str(row["target_class"])
        target_gray_value = int(row["target_gray_value"])
        case_prompt_id = int(row["case_prompt_id"])
        prompt_activate_frame = int(row["prompt_activate_frame"])
        gt_exists = bool(row["gt_exists"])

        pred_mask_path = ""
        pred_overlay_path = ""

        try:
            gt_mask = read_gt_mask(label_path, target_gray_value)
            gt_h, gt_w = gt_mask.shape[:2]

            out = predictor.predict(image_path=image_path, text_prompt=medsam3_text_prompt)
            pred_mask = ensure_mask_shape(out["pred_mask"], (gt_h, gt_w))

            d = dice_score(pred_mask, gt_mask)
            miou = mask_iou(pred_mask, gt_mask)

            pred_area = int(pred_mask.sum())
            gt_area = int(gt_mask.sum())

            pred_bbox = mask_to_bbox(pred_mask)
            gt_bbox = mask_to_bbox(gt_mask)
            biou = bbox_iou(pred_bbox, gt_bbox)

            absent_false_positive = (not gt_exists) and pred_area > 0
            absent_empty_correct = (not gt_exists) and pred_area == 0

            if SAVE_PRED_MASK or SAVE_PRED_OVERLAY:
                mask_path, overlay_path = make_pred_paths(
                    dataset=dataset,
                    case_id=case_id,
                    frame_idx=frame_idx,
                    frame_name=frame_name,
                    prompt_id=case_prompt_id,
                    target_class=target_class,
                )

                should_save = SAVE_EMPTY_MASKS or pred_area > 0

                if should_save and SAVE_PRED_MASK:
                    save_mask(mask_path, pred_mask)
                    pred_mask_path = str(mask_path)

                if should_save and SAVE_PRED_OVERLAY:
                    save_overlay(
                        overlay_path=overlay_path,
                        image_rgb=out["image_rgb"],
                        pred_mask=pred_mask,
                        gt_mask=gt_mask,
                    )
                    pred_overlay_path = str(overlay_path)

            selected_box = out["selected_box"]
            selected_box_str = bbox_to_str(selected_box)

            rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "image_path": image_path,
                    "label_path": label_path,
                    "prompt": prompt,
                    "medsam3_text_prompt": medsam3_text_prompt,
                    "prompt_type": prompt_type,
                    "target_class": target_class,
                    "target_gray_value": target_gray_value,
                    "case_prompt_id": case_prompt_id,
                    "prompt_activate_frame": prompt_activate_frame,
                    "gt_exists": gt_exists,
                    "gt_area": gt_area,
                    "pred_area": pred_area,
                    "gt_bbox": bbox_to_str(gt_bbox),
                    "pred_bbox": bbox_to_str(pred_bbox),
                    "dice": float(d),
                    "mask_iou": float(miou),
                    "bbox_iou": float(biou),
                    "num_masks": int(out["num_masks"]),
                    "selected_mask_idx": int(out["selected_idx"]),
                    "medsam3_score": float(out["selected_score"]) if not np.isnan(out["selected_score"]) else np.nan,
                    "medsam3_box": selected_box_str,
                    "pred_keys": "|".join([str(x) for x in out["pred_keys"]]),
                    "absent_false_positive": bool(absent_false_positive),
                    "absent_empty_correct": bool(absent_empty_correct),
                    "pred_mask_path": pred_mask_path,
                    "pred_overlay_path": pred_overlay_path,
                    "error": "",
                }
            )

        except Exception as e:
            print("\n" + "=" * 120, file=sys.stderr)
            print(
                f"[ERROR] Failed row: dataset={dataset}, case_id={case_id}, "
                f"frame_idx={frame_idx}, prompt_id={case_prompt_id}, target={target_class}",
                file=sys.stderr,
            )
            print("=" * 120, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

            rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "image_path": image_path,
                    "label_path": label_path,
                    "prompt": prompt,
                    "medsam3_text_prompt": medsam3_text_prompt,
                    "prompt_type": prompt_type,
                    "target_class": target_class,
                    "target_gray_value": target_gray_value,
                    "case_prompt_id": case_prompt_id,
                    "prompt_activate_frame": prompt_activate_frame,
                    "gt_exists": gt_exists,
                    "gt_area": np.nan,
                    "pred_area": np.nan,
                    "gt_bbox": "",
                    "pred_bbox": "",
                    "dice": np.nan,
                    "mask_iou": np.nan,
                    "bbox_iou": np.nan,
                    "num_masks": np.nan,
                    "selected_mask_idx": np.nan,
                    "medsam3_score": np.nan,
                    "medsam3_box": "",
                    "pred_keys": "",
                    "absent_false_positive": np.nan,
                    "absent_empty_correct": np.nan,
                    "pred_mask_path": pred_mask_path,
                    "pred_overlay_path": pred_overlay_path,
                    "error": repr(e),
                }
            )

    frame_df = pd.DataFrame(rows)
    case_df, class_summary, absent_summary = summarize(frame_df)

    out_xlsx = RUN_ROOT / "egomed5_medsam3_image_text_prompt_eval.xlsx"
    frame_csv = RUN_ROOT / "frame_prompt_metrics.csv"
    case_csv = RUN_ROOT / "case_prompt_metrics_gt_exists.csv"
    summary_csv = RUN_ROOT / "class_prompt_summary_gt_exists.csv"
    absent_csv = RUN_ROOT / "absent_frame_summary.csv"

    frame_df.to_csv(frame_csv, index=False)
    case_df.to_csv(case_csv, index=False)
    class_summary.to_csv(summary_csv, index=False)
    absent_summary.to_csv(absent_csv, index=False)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        frame_df.to_excel(writer, sheet_name="frame_prompt_metrics", index=False)
        case_df.to_excel(writer, sheet_name="case_prompt_metrics", index=False)
        class_summary.to_excel(writer, sheet_name="class_prompt_summary", index=False)
        absent_summary.to_excel(writer, sheet_name="absent_frame_summary", index=False)

    elapsed = time.time() - start_time

    print("\n" + "=" * 120)
    print("MedSAM3 image evaluation done.")
    print(f"Elapsed seconds:      {elapsed:.2f}")
    print(f"Excel:                {out_xlsx}")
    print(f"Frame CSV:            {frame_csv}")
    print(f"Case CSV:             {case_csv}")
    print(f"Class summary CSV:    {summary_csv}")
    print(f"Absent summary CSV:   {absent_csv}")
    print(f"Pred mask root:       {PRED_MASK_ROOT if SAVE_PRED_MASK else 'disabled'}")
    print(f"Pred overlay root:    {PRED_OVERLAY_ROOT if SAVE_PRED_OVERLAY else 'disabled'}")
    print("=" * 120)

    if len(class_summary) > 0:
        display_cols = [
            "dataset",
            "prompt_type",
            "prompt",
            "target_class",
            "mean_dice",
            "std_dice",
            "mean_mask_iou",
            "mean_bbox_iou",
            "mean_medsam3_score",
            "mean_num_masks",
            "num_cases",
            "total_gt_exists_frames",
        ]
        existing = [c for c in display_cols if c in class_summary.columns]
        print("\nClass prompt summary:")
        print(class_summary[existing].to_string(index=False))

    if len(absent_summary) > 0:
        display_cols = [
            "dataset",
            "prompt_type",
            "prompt",
            "target_class",
            "num_absent_frames",
            "absent_false_positive_rate",
            "mean_pred_area_absent",
        ]
        existing = [c for c in display_cols if c in absent_summary.columns]
        print("\nAbsent-frame summary:")
        print(absent_summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
