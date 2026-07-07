#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM3 image-level text-prompt evaluation on EgoMed5 test prompt CSV.
使用 Sam3Processor 进行图像级文本提示分割评估。

Run:
  conda activate sam2
  cd $EGOMED_ROOT
  python eval_sam3_image_text_prompt_egomed5.py
"""

import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
EXT_ROOT = _Path(_os.environ.get("EGOMED_EXT_ROOT", REPO_ROOT.parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm




# =============================================================================
# SAM3 dtype safety patch
# =============================================================================

def patch_linear_dtype_for_sam3(max_print: int = 20):
    """
    SAM3 当前环境里会出现：
      input=torch.bfloat16, weight=torch.float32
    或相反的 Linear dtype mismatch。

    这里做一个兜底：
    每次 Linear forward 前，把 input 转成该 Linear weight 的 dtype。
    这样不会因为局部 dtype 不一致直接崩溃。

    max_print 控制最多打印多少次 mismatch，避免 137046 张评估时日志爆炸。
    """
    import torch.nn as nn
    import torch.nn.functional as F

    if getattr(nn.Linear, "_sam3_dtype_patch_enabled", False):
        return

    old_forward = nn.Linear.forward
    counter = {"n": 0}

    def new_forward(self, input):
        if input.dtype != self.weight.dtype:
            if counter["n"] < max_print:
                print(
                    f"[DTYPE PATCH] Linear dtype mismatch fixed: "
                    f"input={input.dtype}, weight={self.weight.dtype}, "
                    f"in_features={self.in_features}, out_features={self.out_features}"
                )
            counter["n"] += 1
            input = input.to(self.weight.dtype)

        bias = self.bias
        if bias is not None and bias.dtype != self.weight.dtype:
            bias = bias.to(self.weight.dtype)

        return F.linear(input, self.weight, bias)

    nn.Linear.forward = new_forward
    nn.Linear._sam3_dtype_patch_enabled = True
    nn.Linear._sam3_old_forward = old_forward
    print("[INFO] Enabled SAM3 Linear dtype safety patch.")


# =============================================================================
# Config
# =============================================================================

SAM3_REPO_ROOT   = Path(f"{EXT_ROOT}/sam3")
SAM3_CHECKPOINT  = Path(f"{EXT_ROOT}/sam3/checkpoints/sam3.pt")

PROMPT_CSV = Path(
    f"{REPO_ROOT}/data/text_prompt_eval/"
    "egomed5_test_text_prompts_online_schedule.csv"
)

RUN_ROOT = (
    Path(f"{REPO_ROOT}/runs/eval_sam3_text_prompt")
    / "egomed5_sam3_image_text_prompt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE_THRESHOLD = 0.5
RESOLUTION           = 1008
SELECT_MASK_MODE     = "top_score"

SAVE_PRED_MASK    = True
SAVE_PRED_OVERLAY = False
SAVE_EMPTY_MASKS  = False

PRED_MASK_ROOT    = RUN_ROOT / "pred_mask"
PRED_OVERLAY_ROOT = RUN_ROOT / "pred_overlay"

SUMMARY_ONLY_GT_EXISTS = True

LOG_ROOT       = RUN_ROOT / "logs"
LOG_FILE       = LOG_ROOT / "run.log"
ERROR_LOG_FILE = LOG_ROOT / "error.log"
CONFIG_JSON    = LOG_ROOT / "config.json"


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
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

    sys.excepthook = handle_exception

    print("\n" + "=" * 120)
    print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file:       {LOG_FILE}")
    print("=" * 120)


def save_config_json():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "SAM3_REPO_ROOT":        str(SAM3_REPO_ROOT),
        "SAM3_CHECKPOINT":       str(SAM3_CHECKPOINT),
        "PROMPT_CSV":            str(PROMPT_CSV),
        "RUN_ROOT":              str(RUN_ROOT),
        "DEVICE":                DEVICE,
        "CONFIDENCE_THRESHOLD":  CONFIDENCE_THRESHOLD,
        "RESOLUTION":            RESOLUTION,
        "SELECT_MASK_MODE":      SELECT_MASK_MODE,
        "SAVE_PRED_MASK":        SAVE_PRED_MASK,
        "SAVE_PRED_OVERLAY":     SAVE_PRED_OVERLAY,
        "SAVE_EMPTY_MASKS":      SAVE_EMPTY_MASKS,
        "CUDA_VISIBLE_DEVICES":  os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# =============================================================================
# Metrics and utilities
# =============================================================================

def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps=1e-6) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask   = gt_mask.astype(bool)
    pred_sum  = int(pred_mask.sum())
    gt_sum    = int(gt_mask.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    return float((2.0 * inter + eps) / (pred_sum + gt_sum + eps))


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps=1e-6) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask   = gt_mask.astype(bool)
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    return float((inter + eps) / (union + eps))


def mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_to_str(box: Optional[np.ndarray]) -> str:
    if box is None:
        return ""
    return ",".join([f"{float(x):.2f}" for x in np.asarray(box).reshape(-1)])


def bbox_iou(box1, box2, eps=1e-6) -> float:
    if box1 is None and box2 is None:
        return 1.0
    if box1 is None or box2 is None:
        return 0.0
    box1 = np.asarray(box1).reshape(-1)
    box2 = np.asarray(box2).reshape(-1)
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)
    area1 = (float(box1[2]) - float(box1[0]) + 1.0) * (float(box1[3]) - float(box1[1]) + 1.0)
    area2 = (float(box2[2]) - float(box2[0]) + 1.0) * (float(box2[3]) - float(box2[1]) + 1.0)
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
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return mask


def safe_name(x) -> str:
    x = str(x)
    for ch in ["/", "\\", ":", " ", "\t", "\n", "|", ",", ";", "\"", "'"]:
        x = x.replace(ch, "_")
    return x


def make_pred_paths(dataset, case_id, frame_idx, frame_name, prompt_id, target_class):
    stem     = Path(str(frame_name)).stem
    cls      = safe_name(target_class)
    rel      = Path(str(dataset)) / str(case_id)
    filename = f"{int(frame_idx):06d}_{stem}_prompt{int(prompt_id):03d}_{cls}.png"
    overlay_filename = f"{int(frame_idx):06d}_{stem}_prompt{int(prompt_id):03d}_{cls}.jpg"
    return (
        PRED_MASK_ROOT    / rel / filename,
        PRED_OVERLAY_ROOT / rel / overlay_filename,
    )


def save_mask(path: Path, pred_mask: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), pred_mask.astype(np.uint8) * 255)


def save_overlay(path: Path, image_rgb: np.ndarray, pred_mask: np.ndarray, gt_mask=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_mask = ensure_mask_shape(pred_mask, image_rgb.shape[:2])
    img_bgr   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    vis       = img_bgr.copy()
    if pred_mask.any():
        red = np.zeros_like(vis)
        red[:, :, 2] = 255
        vis[pred_mask] = cv2.addWeighted(vis, 0.45, red, 0.55, 0)[pred_mask]
    if gt_mask is not None and np.asarray(gt_mask).any():
        gt_mask   = ensure_mask_shape(gt_mask, image_rgb.shape[:2]).astype(np.uint8)
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(str(path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


# =============================================================================
# SAM3 predictor wrapper
# =============================================================================

class SAM3ImageTextPredictor:
    def __init__(self):
        self.processor = None

    def load(self):
        patch_linear_dtype_for_sam3(max_print=20)

        print("\nLoading SAM3 model...")
        print(f"  repo root:  {SAM3_REPO_ROOT}")
        print(f"  checkpoint: {SAM3_CHECKPOINT}")
        print(f"  device:     {DEVICE}")
        print(f"  resolution: {RESOLUTION}")
        print(f"  threshold:  {CONFIDENCE_THRESHOLD}")

        if not SAM3_REPO_ROOT.exists():
            raise FileNotFoundError(f"SAM3 repo not found: {SAM3_REPO_ROOT}")
        if not SAM3_CHECKPOINT.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {SAM3_CHECKPOINT}")

        if str(SAM3_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(SAM3_REPO_ROOT))

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            checkpoint_path=str(SAM3_CHECKPOINT),
            load_from_HF=False,
            device=DEVICE,
            eval_mode=True,
        )

        # 关键修复：
        # 不要手动转 bfloat16。先统一使用 float32，避免 Float / BFloat16 混用。
        model.eval()

        # 打印一个参数 dtype，方便确认模型当前 dtype。
        try:
            first_name, first_param = next(model.named_parameters())
            print(f"  model dtype check: {first_name} -> {first_param.dtype}, {first_param.device}")
        except StopIteration:
            print("  model dtype check: no parameters found")

        first_param = next(model.parameters())
        print(f"Model first parameter dtype: {first_param.dtype}, device: {first_param.device}")

        self.processor = Sam3Processor(
            model=model,
            resolution=RESOLUTION,
            device=DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        print("SAM3 loaded.")

    @torch.no_grad()
    def predict(self, image_path: str, text_prompt: str) -> dict:
        image = Image.open(str(image_path)).convert("RGB")
        w, h  = image.size

        # 关键修复：
        # 显式关闭 autocast，避免输入激活变成 BFloat16、模型权重仍是 Float32。
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(text_prompt, state)
        else:
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(text_prompt, state)

        masks  = state.get("masks", None)   # (N, 1, H, W) bool tensor
        scores = state.get("scores", None)  # (N,) tensor
        boxes  = state.get("boxes", None)   # (N, 4) tensor xyxy

        # Convert to numpy
        if masks is not None:
            masks_np  = masks.squeeze(1).cpu().numpy().astype(bool)  # (N, H, W)
        else:
            masks_np  = np.zeros((0, h, w), dtype=bool)

        if scores is not None:
            scores_np = scores.float().cpu().numpy().astype(np.float32)
        else:
            scores_np = np.full(len(masks_np), np.nan, dtype=np.float32)

        if boxes is not None:
            boxes_np  = boxes.float().cpu().numpy().astype(np.float32)
        else:
            boxes_np  = np.full((len(masks_np), 4), np.nan, dtype=np.float32)

        n_masks = len(masks_np)

        if n_masks == 0:
            return {
                "pred_mask":      None,
                "num_masks":      0,
                "selected_idx":   -1,
                "selected_score": np.nan,
                "selected_box":   np.full(4, np.nan, dtype=np.float32),
            }

        if SELECT_MASK_MODE == "top_score" and not np.all(np.isnan(scores_np)):
            selected_idx = int(np.nanargmax(scores_np))
        else:
            selected_idx = 0

        return {
            "pred_mask":      masks_np[selected_idx],
            "num_masks":      n_masks,
            "selected_idx":   selected_idx,
            "selected_score": float(scores_np[selected_idx]),
            "selected_box":   boxes_np[selected_idx],
        }


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
                ["dataset", "case_id", "prompt_type", "prompt", "target_class", "target_gray_value"],
                dropna=False,
            )
            .agg(
                mean_dice           = ("dice",       "mean"),
                std_dice            = ("dice",       "std"),
                mean_mask_iou       = ("mask_iou",   "mean"),
                mean_bbox_iou       = ("bbox_iou",   "mean"),
                mean_sam3_score     = ("sam3_score", "mean"),
                mean_pred_area      = ("pred_area",  "mean"),
                mean_gt_area        = ("gt_area",    "mean"),
                mean_num_masks      = ("num_masks",  "mean"),
                num_gt_exists_frames= ("gt_exists",  "count"),
            )
            .reset_index()
        )

        class_summary = (
            case_df.groupby(
                ["dataset", "prompt_type", "prompt", "target_class", "target_gray_value"],
                dropna=False,
            )
            .agg(
                mean_dice            = ("mean_dice",       "mean"),
                std_dice             = ("mean_dice",       "std"),
                mean_mask_iou        = ("mean_mask_iou",   "mean"),
                mean_bbox_iou        = ("mean_bbox_iou",   "mean"),
                mean_sam3_score      = ("mean_sam3_score", "mean"),
                mean_pred_area       = ("mean_pred_area",  "mean"),
                mean_gt_area         = ("mean_gt_area",    "mean"),
                mean_num_masks       = ("mean_num_masks",  "mean"),
                num_cases            = ("case_id",         "count"),
                total_gt_exists_frames=("num_gt_exists_frames", "sum"),
            )
            .reset_index()
            .sort_values(["dataset", "prompt_type", "target_gray_value", "prompt"])
        )
    else:
        case_df       = pd.DataFrame()
        class_summary = pd.DataFrame()

    absent_df = frame_df[
        (frame_df["gt_exists"] == False) &
        (frame_df["error"].fillna("") == "")
    ].copy()

    if len(absent_df) > 0:
        absent_summary = (
            absent_df.groupby(
                ["dataset", "prompt_type", "prompt", "target_class", "target_gray_value"],
                dropna=False,
            )
            .agg(
                num_absent_frames          = ("gt_exists",             "count"),
                num_absent_false_positive  = ("absent_false_positive", "sum"),
                num_absent_empty_correct   = ("absent_empty_correct",  "sum"),
                mean_pred_area_absent      = ("pred_area",             "mean"),
            )
            .reset_index()
        )
        absent_summary["absent_false_positive_rate"] = (
            absent_summary["num_absent_false_positive"]
            / absent_summary["num_absent_frames"].clip(lower=1)
        )
    else:
        absent_summary = pd.DataFrame()

    return case_df, class_summary, absent_summary


# =============================================================================
# Main
# =============================================================================

def main():
    setup_run_logging()
    save_config_json()
    start_time = time.time()

    if not PROMPT_CSV.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {PROMPT_CSV}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if SAVE_PRED_MASK:
        PRED_MASK_ROOT.mkdir(parents=True, exist_ok=True)
    if SAVE_PRED_OVERLAY:
        PRED_OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("SAM3 Image Text-Prompt Evaluation on EgoMed5")
    print("=" * 120)
    print(f"Prompt CSV:    {PROMPT_CSV}")
    print(f"Checkpoint:    {SAM3_CHECKPOINT}")
    print(f"Device:        {DEVICE}")
    print(f"Run root:      {RUN_ROOT}")
    print(f"Pred mask root:{PRED_MASK_ROOT}")
    print("=" * 120)

    df = pd.read_csv(PROMPT_CSV)
    df["dataset"] = df["dataset"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df = df.sort_values(["dataset", "case_id", "frame_idx", "case_prompt_id"]).reset_index(drop=True)

    predictor = SAM3ImageTextPredictor()
    predictor.load()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="SAM3 image eval"):
        dataset              = str(row["dataset"])
        case_id              = str(row["case_id"])
        frame_idx            = int(row["frame_idx"])
        frame_name           = str(row["frame_name"])
        image_path           = str(row["image_path"])
        label_path           = str(row["label_path"])
        prompt               = str(row["prompt"])
        prompt_type          = str(row["prompt_type"])
        target_class         = str(row["target_class"])
        target_gray_value    = int(row["target_gray_value"])
        case_prompt_id       = int(row["case_prompt_id"])
        prompt_activate_frame= int(row["prompt_activate_frame"])
        gt_exists            = bool(row["gt_exists"])

        pred_mask_path    = ""
        pred_overlay_path = ""

        try:
            gt_mask  = read_gt_mask(label_path, target_gray_value)
            gt_h, gt_w = gt_mask.shape[:2]

            out       = predictor.predict(image_path=image_path, text_prompt=prompt)
            pred_mask = ensure_mask_shape(out["pred_mask"], (gt_h, gt_w))

            d         = dice_score(pred_mask, gt_mask)
            miou      = mask_iou(pred_mask, gt_mask)
            pred_area = int(pred_mask.sum())
            gt_area   = int(gt_mask.sum())
            pred_bbox = mask_to_bbox(pred_mask)
            gt_bbox   = mask_to_bbox(gt_mask)
            biou      = bbox_iou(pred_bbox, gt_bbox)

            absent_false_positive = (not gt_exists) and pred_area > 0
            absent_empty_correct  = (not gt_exists) and pred_area == 0

            if SAVE_PRED_MASK or SAVE_PRED_OVERLAY:
                mask_path, overlay_path = make_pred_paths(
                    dataset, case_id, frame_idx, frame_name, case_prompt_id, target_class
                )
                should_save = SAVE_EMPTY_MASKS or pred_area > 0
                if should_save and SAVE_PRED_MASK:
                    save_mask(mask_path, pred_mask)
                    pred_mask_path = str(mask_path)
                if should_save and SAVE_PRED_OVERLAY:
                    image_rgb = np.array(Image.open(image_path).convert("RGB"))
                    save_overlay(overlay_path, image_rgb, pred_mask, gt_mask)
                    pred_overlay_path = str(overlay_path)

            rows.append({
                "dataset":               dataset,
                "case_id":               case_id,
                "frame_idx":             frame_idx,
                "frame_name":            frame_name,
                "image_path":            image_path,
                "label_path":            label_path,
                "prompt":                prompt,
                "prompt_type":           prompt_type,
                "target_class":          target_class,
                "target_gray_value":     target_gray_value,
                "case_prompt_id":        case_prompt_id,
                "prompt_activate_frame": prompt_activate_frame,
                "gt_exists":             gt_exists,
                "gt_area":               gt_area,
                "pred_area":             pred_area,
                "gt_bbox":               bbox_to_str(gt_bbox),
                "pred_bbox":             bbox_to_str(pred_bbox),
                "dice":                  float(d),
                "mask_iou":              float(miou),
                "bbox_iou":              float(biou),
                "num_masks":             int(out["num_masks"]),
                "selected_mask_idx":     int(out["selected_idx"]),
                "sam3_score":            float(out["selected_score"]) if not np.isnan(out["selected_score"]) else np.nan,
                "sam3_box":              bbox_to_str(out["selected_box"]),
                "absent_false_positive": bool(absent_false_positive),
                "absent_empty_correct":  bool(absent_empty_correct),
                "pred_mask_path":        pred_mask_path,
                "pred_overlay_path":     pred_overlay_path,
                "error":                 "",
            })

        except Exception as e:
            print(f"\n[ERROR] dataset={dataset}, case_id={case_id}, frame_idx={frame_idx}, "
                  f"prompt_id={case_prompt_id}, target={target_class}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            rows.append({
                "dataset": dataset, "case_id": case_id, "frame_idx": frame_idx,
                "frame_name": frame_name, "image_path": image_path, "label_path": label_path,
                "prompt": prompt, "prompt_type": prompt_type, "target_class": target_class,
                "target_gray_value": target_gray_value, "case_prompt_id": case_prompt_id,
                "prompt_activate_frame": prompt_activate_frame, "gt_exists": gt_exists,
                "gt_area": np.nan, "pred_area": np.nan, "gt_bbox": "", "pred_bbox": "",
                "dice": np.nan, "mask_iou": np.nan, "bbox_iou": np.nan,
                "num_masks": np.nan, "selected_mask_idx": np.nan, "sam3_score": np.nan,
                "sam3_box": "", "absent_false_positive": np.nan, "absent_empty_correct": np.nan,
                "pred_mask_path": pred_mask_path, "pred_overlay_path": pred_overlay_path,
                "error": repr(e),
            })

    frame_df = pd.DataFrame(rows)
    case_df, class_summary, absent_summary = summarize(frame_df)

    out_xlsx    = RUN_ROOT / "egomed5_sam3_image_text_prompt_eval.xlsx"
    frame_csv   = RUN_ROOT / "frame_prompt_metrics.csv"
    case_csv    = RUN_ROOT / "case_prompt_metrics_gt_exists.csv"
    summary_csv = RUN_ROOT / "class_prompt_summary_gt_exists.csv"
    absent_csv  = RUN_ROOT / "absent_frame_summary.csv"

    frame_df.to_csv(frame_csv, index=False)
    case_df.to_csv(case_csv, index=False)
    class_summary.to_csv(summary_csv, index=False)
    absent_summary.to_csv(absent_csv, index=False)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        frame_df.to_excel(writer,      sheet_name="frame_prompt_metrics", index=False)
        case_df.to_excel(writer,       sheet_name="case_prompt_metrics",  index=False)
        class_summary.to_excel(writer, sheet_name="class_prompt_summary", index=False)
        absent_summary.to_excel(writer,sheet_name="absent_frame_summary", index=False)

    elapsed = time.time() - start_time
    print("\n" + "=" * 120)
    print("SAM3 image evaluation done.")
    print(f"Elapsed:       {elapsed:.2f}s")
    print(f"Excel:         {out_xlsx}")
    print(f"Pred mask root:{PRED_MASK_ROOT if SAVE_PRED_MASK else 'disabled'}")
    print("=" * 120)

    if len(class_summary) > 0:
        display_cols = ["dataset", "prompt_type", "prompt", "target_class",
                        "mean_dice", "std_dice", "mean_mask_iou", "mean_bbox_iou",
                        "mean_sam3_score", "mean_num_masks", "num_cases", "total_gt_exists_frames"]
        existing = [c for c in display_cols if c in class_summary.columns]
        print("\nClass prompt summary:")
        print(class_summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
