#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grounded-SAM2 image-level text-prompt evaluation on EgoMed5 test prompt CSV.

Pipeline per row:
  image_path + row["prompt"]
  -> GroundingDINO text-conditioned boxes
  -> choose one box without GT
  -> SAM2 image predictor with box prompt
  -> selected mask
  -> compare with label_path + target_gray_value

Fair comparison:
  TEXT_SOURCE = "prompt" uses the exact CSV prompt, same as LISA/SAM3/MedSAM3.

Run:
  conda activate grounded_sam2
  cd $EGOMED_ROOT
  python eval_grounded_sam2_image_text_prompt_egomed5_save_outputs.py
"""

import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
EXT_ROOT = _Path(_os.environ.get("EGOMED_EXT_ROOT", REPO_ROOT.parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Keep the torch C++ libs visible for GroundingDINO custom ops.
_CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")
if _CONDA_PREFIX:
    torch_lib = f"{_CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib"
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{_CONDA_PREFIX}/lib:{_CONDA_PREFIX}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

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

GROUNDED_SAM2_ROOT = Path(f"{EXT_ROOT}/Grounded-SAM-2")
GROUNDING_DINO_ROOT = GROUNDED_SAM2_ROOT / "grounding_dino"

# Change these two if your GroundingDINO files are named differently.
GROUNDING_DINO_CONFIG = (
    GROUNDED_SAM2_ROOT
    / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT = (
    GROUNDED_SAM2_ROOT
    / "gdino_checkpoints/groundingdino_swint_ogc.pth"
)

# Original SAM2.1 Hiera-B+ baseline.
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT = Path(f"{REPO_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt")

PROMPT_CSV = Path(
    f"{REPO_ROOT}/data/text_prompt_eval/"
    "egomed5_test_text_prompts_online_schedule.csv"
)

RUN_ROOT = (
    Path(f"{REPO_ROOT}/runs/eval_grounded_sam2_text_prompt")
    / "egomed5_grounded_sam2_image_text_prompt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fair comparison: use exact CSV prompt.
TEXT_SOURCE = "prompt"

# GroundingDINO thresholds.
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# Selection without GT.
SELECT_BOX_MODE = "top_score"       # top_score | largest_box
SELECT_SAM2_MASK_MODE = "top_score" # top_score | largest_mask
MULTIMASK_OUTPUT = True

# Resource-safe outputs.
SAVE_PRED_MASK = True
SAVE_PRED_OVERLAY = False
SAVE_EMPTY_MASKS = False

PRED_MASK_ROOT = RUN_ROOT / "pred_mask"
PRED_OVERLAY_ROOT = RUN_ROOT / "pred_overlay"
SUMMARY_ONLY_GT_EXISTS = True
CONFIG_JSON = RUN_ROOT / "config.json"


# =============================================================================
# Imports from Grounded-SAM2
# =============================================================================

if str(GROUNDED_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDED_SAM2_ROOT))
if str(GROUNDING_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDING_DINO_ROOT))

from groundingdino.util.inference import load_model as load_grounding_dino_model
from groundingdino.util.inference import predict as grounding_dino_predict
from groundingdino.util.inference import load_image as grounding_dino_load_image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =============================================================================
# Utility functions
# =============================================================================

def save_config_json():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "GROUNDED_SAM2_ROOT": str(GROUNDED_SAM2_ROOT),
        "GROUNDING_DINO_CONFIG": str(GROUNDING_DINO_CONFIG),
        "GROUNDING_DINO_CHECKPOINT": str(GROUNDING_DINO_CHECKPOINT),
        "SAM2_MODEL_CFG": SAM2_MODEL_CFG,
        "SAM2_CHECKPOINT": str(SAM2_CHECKPOINT),
        "PROMPT_CSV": str(PROMPT_CSV),
        "RUN_ROOT": str(RUN_ROOT),
        "DEVICE": DEVICE,
        "TEXT_SOURCE": TEXT_SOURCE,
        "BOX_THRESHOLD": BOX_THRESHOLD,
        "TEXT_THRESHOLD": TEXT_THRESHOLD,
        "SELECT_BOX_MODE": SELECT_BOX_MODE,
        "SELECT_SAM2_MASK_MODE": SELECT_SAM2_MASK_MODE,
        "MULTIMASK_OUTPUT": MULTIMASK_OUTPUT,
        "SAVE_PRED_MASK": SAVE_PRED_MASK,
        "SAVE_PRED_OVERLAY": SAVE_PRED_OVERLAY,
        "SAVE_EMPTY_MASKS": SAVE_EMPTY_MASKS,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def normalize_text(row: pd.Series) -> str:
    if TEXT_SOURCE == "target_class":
        text = str(row["target_class"]).strip()
    else:
        text = str(row["prompt"]).strip()
    return text if text else str(row.get("target_class", "object")).strip()


def load_image_rgb(image_path: str) -> np.ndarray:
    return np.asarray(Image.open(str(image_path)).convert("RGB"))


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
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps=1e-6) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    ps = int(pred_mask.sum())
    gs = int(gt_mask.sum())
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    return float((2.0 * inter + eps) / (ps + gs + eps))


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
    inter = max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)
    return float(inter / (bbox_area(box1) + bbox_area(box2) - inter + eps))


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
    return PRED_MASK_ROOT / rel / filename, PRED_OVERLAY_ROOT / rel / overlay_filename


def save_mask(mask_save_path: Path, pred_mask: np.ndarray):
    mask_save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_save_path), pred_mask.astype(np.uint8) * 255)


def save_overlay(overlay_path: Path, image_rgb: np.ndarray, pred_mask: np.ndarray, gt_mask=None, grounding_box=None):
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    pred_mask = ensure_mask_shape(pred_mask, image_rgb.shape[:2])
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if pred_mask.any():
        red = np.zeros_like(vis)
        red[:, :, 2] = 255
        vis[pred_mask] = cv2.addWeighted(vis, 0.45, red, 0.55, 0)[pred_mask]
    if gt_mask is not None and np.asarray(gt_mask).any():
        gt_mask = ensure_mask_shape(gt_mask, image_rgb.shape[:2]).astype(np.uint8)
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    if grounding_box is not None and not np.any(np.isnan(grounding_box)):
        x1, y1, x2, y2 = np.asarray(grounding_box).astype(int).tolist()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(str(overlay_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def cxcywh_norm_to_xyxy_pixel(boxes: np.ndarray, image_size_wh: Tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    w, h = image_size_wh
    cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
    xyxy = np.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], axis=1)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)
    return xyxy.astype(np.float32)


def select_box(boxes_xyxy: np.ndarray, scores: np.ndarray) -> Tuple[int, np.ndarray, float]:
    if len(boxes_xyxy) == 0:
        return -1, np.full((4,), np.nan, dtype=np.float32), np.nan
    if SELECT_BOX_MODE == "largest_box":
        areas = np.maximum(0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0] + 1) * np.maximum(0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1] + 1)
        idx = int(np.argmax(areas))
    else:
        idx = int(np.nanargmax(scores))
    return idx, boxes_xyxy[idx].astype(np.float32), float(scores[idx])


def select_sam2_mask(masks: np.ndarray, scores: np.ndarray) -> Tuple[int, Optional[np.ndarray], float]:
    if masks is None or len(masks) == 0:
        return -1, None, np.nan
    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1) if scores is not None else np.full((len(masks),), np.nan)
    if SELECT_SAM2_MASK_MODE == "largest_mask":
        idx = int(np.argmax(masks.reshape(len(masks), -1).sum(axis=1)))
    elif len(scores) == len(masks) and not np.all(np.isnan(scores)):
        idx = int(np.nanargmax(scores))
    else:
        idx = 0
    return idx, masks[idx].astype(bool), float(scores[idx]) if idx < len(scores) else np.nan


# =============================================================================
# Predictor
# =============================================================================

class GroundedSAM2ImagePredictor:
    def __init__(self):
        self.grounding_model = None
        self.sam2_predictor = None

    def load(self):
        print("\nLoading Grounded-SAM2...")
        print(f"  Grounded-SAM2 root: {GROUNDED_SAM2_ROOT}")
        print(f"  GroundingDINO cfg:  {GROUNDING_DINO_CONFIG}")
        print(f"  GroundingDINO ckpt: {GROUNDING_DINO_CHECKPOINT}")
        print(f"  SAM2 cfg:           {SAM2_MODEL_CFG}")
        print(f"  SAM2 ckpt:          {SAM2_CHECKPOINT}")
        print(f"  device:             {DEVICE}")

        for p, msg in [
            (GROUNDED_SAM2_ROOT, "Grounded-SAM2 root"),
            (GROUNDING_DINO_CONFIG, "GroundingDINO config"),
            (GROUNDING_DINO_CHECKPOINT, "GroundingDINO checkpoint"),
            (SAM2_CHECKPOINT, "SAM2 checkpoint"),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"{msg} not found: {p}")

        self.grounding_model = load_grounding_dino_model(
            model_config_path=str(GROUNDING_DINO_CONFIG),
            model_checkpoint_path=str(GROUNDING_DINO_CHECKPOINT),
            device=DEVICE,
        )

        sam2_model = build_sam2(
            config_file=SAM2_MODEL_CFG,
            ckpt_path=str(SAM2_CHECKPOINT),
            device=DEVICE,
        )
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        print("Grounded-SAM2 loaded.")

    @torch.no_grad()
    def predict(self, image_path: str, text_prompt: str) -> Dict[str, Any]:
        image_rgb = load_image_rgb(image_path)
        h, w = image_rgb.shape[:2]

        _, image_tensor = grounding_dino_load_image(str(image_path))

        boxes, logits, phrases = grounding_dino_predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=str(text_prompt),
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )

        boxes_np = to_numpy(boxes)
        logits_np = to_numpy(logits)

        if boxes_np is None or len(boxes_np) == 0:
            return {
                "pred_mask": None,
                "image_rgb": image_rgb,
                "num_boxes": 0,
                "selected_box_idx": -1,
                "grounding_score": np.nan,
                "grounding_box": np.full((4,), np.nan, dtype=np.float32),
                "grounding_phrases": [],
                "sam2_mask_idx": -1,
                "sam2_score": np.nan,
            }

        boxes_xyxy = cxcywh_norm_to_xyxy_pixel(boxes_np, image_size_wh=(w, h))
        logits_np = np.asarray(logits_np, dtype=np.float32).reshape(-1)
        box_idx, selected_box, grounding_score = select_box(boxes_xyxy, logits_np)

        self.sam2_predictor.set_image(image_rgb)
        masks, sam2_scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=selected_box[None, :],
            multimask_output=MULTIMASK_OUTPUT,
        )
        sam2_mask_idx, pred_mask, sam2_score = select_sam2_mask(masks, sam2_scores)

        return {
            "pred_mask": pred_mask,
            "image_rgb": image_rgb,
            "num_boxes": int(len(boxes_xyxy)),
            "selected_box_idx": int(box_idx),
            "grounding_score": float(grounding_score),
            "grounding_box": selected_box.astype(np.float32),
            "grounding_phrases": [str(x) for x in phrases],
            "sam2_mask_idx": int(sam2_mask_idx),
            "sam2_score": float(sam2_score) if not np.isnan(sam2_score) else np.nan,
        }


# =============================================================================
# Summary
# =============================================================================

def summarize(frame_df: pd.DataFrame):
    if "error" not in frame_df.columns:
        frame_df["error"] = ""
    metric_df = frame_df[frame_df["gt_exists"] == True].copy() if SUMMARY_ONLY_GT_EXISTS else frame_df.copy()
    metric_df = metric_df[metric_df["error"].fillna("") == ""]

    if len(metric_df) > 0:
        case_df = (
            metric_df.groupby(["dataset", "case_id", "prompt_type", "prompt", "target_class", "target_gray_value"], dropna=False)
            .agg(
                mean_dice=("dice", "mean"),
                std_dice=("dice", "std"),
                mean_mask_iou=("mask_iou", "mean"),
                std_mask_iou=("mask_iou", "std"),
                mean_bbox_iou=("bbox_iou", "mean"),
                mean_grounding_score=("grounding_score", "mean"),
                mean_sam2_score=("sam2_score", "mean"),
                mean_pred_area=("pred_area", "mean"),
                mean_gt_area=("gt_area", "mean"),
                mean_num_boxes=("num_boxes", "mean"),
                num_gt_exists_frames=("gt_exists", "count"),
            )
            .reset_index()
        )

        class_summary = (
            case_df.groupby(["dataset", "prompt_type", "prompt", "target_class", "target_gray_value"], dropna=False)
            .agg(
                mean_dice=("mean_dice", "mean"),
                std_dice=("mean_dice", "std"),
                mean_mask_iou=("mean_mask_iou", "mean"),
                std_mask_iou=("mean_mask_iou", "std"),
                mean_bbox_iou=("mean_bbox_iou", "mean"),
                mean_grounding_score=("mean_grounding_score", "mean"),
                mean_sam2_score=("mean_sam2_score", "mean"),
                mean_pred_area=("mean_pred_area", "mean"),
                mean_gt_area=("mean_gt_area", "mean"),
                mean_num_boxes=("mean_num_boxes", "mean"),
                num_cases=("case_id", "count"),
                total_gt_exists_frames=("num_gt_exists_frames", "sum"),
            )
            .reset_index()
            .sort_values(["dataset", "prompt_type", "target_gray_value", "prompt"])
        )
    else:
        case_df = pd.DataFrame()
        class_summary = pd.DataFrame()

    absent_df = frame_df[(frame_df["gt_exists"] == False) & (frame_df["error"].fillna("") == "")].copy()
    if len(absent_df) > 0:
        absent_summary = (
            absent_df.groupby(["dataset", "prompt_type", "prompt", "target_class", "target_gray_value"], dropna=False)
            .agg(
                num_absent_frames=("gt_exists", "count"),
                num_absent_false_positive=("absent_false_positive", "sum"),
                num_absent_empty_correct=("absent_empty_correct", "sum"),
                mean_pred_area_absent=("pred_area", "mean"),
            )
            .reset_index()
        )
        absent_summary["absent_false_positive_rate"] = absent_summary["num_absent_false_positive"] / absent_summary["num_absent_frames"].clip(lower=1)
        absent_summary["absent_empty_correct_rate"] = absent_summary["num_absent_empty_correct"] / absent_summary["num_absent_frames"].clip(lower=1)
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
    print("Grounded-SAM2 Image Text-Prompt Evaluation on EgoMed5 Test Online Schedule")
    print("=" * 120)
    print(f"Prompt CSV:          {PROMPT_CSV}")
    print(f"Grounded-SAM2 root:  {GROUNDED_SAM2_ROOT}")
    print(f"GroundingDINO cfg:   {GROUNDING_DINO_CONFIG}")
    print(f"GroundingDINO ckpt:  {GROUNDING_DINO_CHECKPOINT}")
    print(f"SAM2 cfg:            {SAM2_MODEL_CFG}")
    print(f"SAM2 ckpt:           {SAM2_CHECKPOINT}")
    print(f"Device:              {DEVICE}")
    print(f"Run root:            {RUN_ROOT}")
    print(f"Text source:         {TEXT_SOURCE}")
    print(f"Box threshold:       {BOX_THRESHOLD}")
    print(f"Text threshold:      {TEXT_THRESHOLD}")
    print(f"Select box mode:     {SELECT_BOX_MODE}")
    print(f"Select SAM2 mask:    {SELECT_SAM2_MASK_MODE}")
    print(f"Save pred masks:     {SAVE_PRED_MASK}")
    print(f"Save overlay:        {SAVE_PRED_OVERLAY}")
    print(f"Save empty masks:    {SAVE_EMPTY_MASKS}")
    print("=" * 120)

    df = pd.read_csv(PROMPT_CSV)
    required_cols = [
        "dataset", "case_id", "frame_idx", "frame_name", "image_path", "label_path", "prompt",
        "prompt_type", "target_class", "target_gray_value", "case_prompt_id", "prompt_activate_frame", "gt_exists",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in prompt CSV: {col}")

    df["dataset"] = df["dataset"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df = df.sort_values(["dataset", "case_id", "frame_idx", "case_prompt_id"]).reset_index(drop=True)

    predictor = GroundedSAM2ImagePredictor()
    predictor.load()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grounded-SAM2 image eval"):
        dataset = str(row["dataset"])
        case_id = str(row["case_id"])
        frame_idx = int(row["frame_idx"])
        frame_name = str(row["frame_name"])
        image_path = str(row["image_path"])
        label_path = str(row["label_path"])
        prompt = str(row["prompt"])
        text_prompt = normalize_text(row)
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
            out = predictor.predict(image_path=image_path, text_prompt=text_prompt)
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
                mask_path, overlay_path = make_pred_paths(dataset, case_id, frame_idx, frame_name, case_prompt_id, target_class)
                should_save = SAVE_EMPTY_MASKS or pred_area > 0
                if should_save and SAVE_PRED_MASK:
                    save_mask(mask_path, pred_mask)
                    pred_mask_path = str(mask_path)
                if should_save and SAVE_PRED_OVERLAY:
                    save_overlay(overlay_path, out["image_rgb"], pred_mask, gt_mask=gt_mask, grounding_box=out["grounding_box"])
                    pred_overlay_path = str(overlay_path)

            rows.append({
                "dataset": dataset,
                "case_id": case_id,
                "frame_idx": frame_idx,
                "frame_name": frame_name,
                "image_path": image_path,
                "label_path": label_path,
                "prompt": prompt,
                "grounded_sam2_text_prompt": text_prompt,
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
                "num_boxes": int(out["num_boxes"]),
                "selected_box_idx": int(out["selected_box_idx"]),
                "grounding_score": float(out["grounding_score"]) if not np.isnan(out["grounding_score"]) else np.nan,
                "grounding_box": bbox_to_str(out["grounding_box"]),
                "grounding_phrases": "|".join(out["grounding_phrases"]),
                "sam2_mask_idx": int(out["sam2_mask_idx"]),
                "sam2_score": float(out["sam2_score"]) if not np.isnan(out["sam2_score"]) else np.nan,
                "absent_false_positive": bool(absent_false_positive),
                "absent_empty_correct": bool(absent_empty_correct),
                "pred_mask_path": pred_mask_path,
                "pred_overlay_path": pred_overlay_path,
                "error": "",
            })

        except Exception as e:
            print("\n" + "=" * 120, file=sys.stderr)
            print(f"[ERROR] Failed row: dataset={dataset}, case_id={case_id}, frame_idx={frame_idx}, prompt_id={case_prompt_id}, target={target_class}", file=sys.stderr)
            print("=" * 120, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            rows.append({
                "dataset": dataset,
                "case_id": case_id,
                "frame_idx": frame_idx,
                "frame_name": frame_name,
                "image_path": image_path,
                "label_path": label_path,
                "prompt": prompt,
                "grounded_sam2_text_prompt": text_prompt,
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
                "num_boxes": np.nan,
                "selected_box_idx": np.nan,
                "grounding_score": np.nan,
                "grounding_box": "",
                "grounding_phrases": "",
                "sam2_mask_idx": np.nan,
                "sam2_score": np.nan,
                "absent_false_positive": np.nan,
                "absent_empty_correct": np.nan,
                "pred_mask_path": pred_mask_path,
                "pred_overlay_path": pred_overlay_path,
                "error": repr(e),
            })

    frame_df = pd.DataFrame(rows)
    case_df, class_summary, absent_summary = summarize(frame_df)

    out_xlsx = RUN_ROOT / "egomed5_grounded_sam2_image_text_prompt_eval.xlsx"
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
    print("Grounded-SAM2 image evaluation done.")
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
            "dataset", "prompt_type", "prompt", "target_class", "mean_dice", "std_dice",
            "mean_mask_iou", "mean_bbox_iou", "mean_grounding_score", "mean_sam2_score",
            "mean_num_boxes", "num_cases", "total_gt_exists_frames",
        ]
        existing = [c for c in display_cols if c in class_summary.columns]
        print("\nClass prompt summary:")
        print(class_summary[existing].to_string(index=False))

    if len(absent_summary) > 0:
        display_cols = ["dataset", "prompt_type", "prompt", "target_class", "num_absent_frames", "absent_false_positive_rate", "mean_pred_area_absent"]
        existing = [c for c in display_cols if c in absent_summary.columns]
        print("\nAbsent-frame summary:")
        print(absent_summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
