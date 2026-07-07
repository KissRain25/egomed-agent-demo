#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangSAM image-level text-prompt evaluation on EgoMed5 test prompt CSV.

Each row:
  image_path + prompt -> LangSAM -> selected mask -> metrics

Run:
  conda activate langsam
  cd $EGOMED_ROOT
  python eval_langsam_image_text_prompt_egomed5_save_outputs.py
"""

import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
EXT_ROOT = _Path(_os.environ.get("EGOMED_EXT_ROOT", REPO_ROOT.parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


LANGSAM_REPO_ROOT = Path(f"{EXT_ROOT}/lang-segment-anything")

PROMPT_CSV = Path(
    f"{REPO_ROOT}/data/text_prompt_eval/"
    "egomed5_test_text_prompts_online_schedule.csv"
)

RUN_ROOT = (
    Path(f"{REPO_ROOT}/runs/eval_langsam_image_text_prompt")
    / "egomed5_langsam_image_text_prompt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_SOURCE = "prompt"
SELECT_MASK_MODE = "top_score"  # top_score | largest_mask | first
MASK_PROB_THRESHOLD = 0.5
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

SAVE_PRED_MASK = True
SAVE_PRED_OVERLAY = False
SAVE_EMPTY_MASKS = False

PRED_MASK_ROOT = RUN_ROOT / "pred_mask"
PRED_OVERLAY_ROOT = RUN_ROOT / "pred_overlay"
SUMMARY_ONLY_GT_EXISTS = True
CONFIG_JSON = RUN_ROOT / "config.json"

MAX_ROWS = None  # e.g. 100 for smoke test
CACHE_PREDICTIONS = False


if str(LANGSAM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(LANGSAM_REPO_ROOT))

from lang_sam import LangSAM


def save_config_json():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "LANGSAM_REPO_ROOT": str(LANGSAM_REPO_ROOT),
        "PROMPT_CSV": str(PROMPT_CSV),
        "RUN_ROOT": str(RUN_ROOT),
        "DEVICE": DEVICE,
        "TEXT_SOURCE": TEXT_SOURCE,
        "SELECT_MASK_MODE": SELECT_MASK_MODE,
        "MASK_PROB_THRESHOLD": MASK_PROB_THRESHOLD,
        "BOX_THRESHOLD": BOX_THRESHOLD,
        "TEXT_THRESHOLD": TEXT_THRESHOLD,
        "SAVE_PRED_MASK": SAVE_PRED_MASK,
        "SAVE_PRED_OVERLAY": SAVE_PRED_OVERLAY,
        "SAVE_EMPTY_MASKS": SAVE_EMPTY_MASKS,
        "PRED_MASK_ROOT": str(PRED_MASK_ROOT),
        "PRED_OVERLAY_ROOT": str(PRED_OVERLAY_ROOT),
        "SUMMARY_ONLY_GT_EXISTS": SUMMARY_ONLY_GT_EXISTS,
        "MAX_ROWS": MAX_ROWS,
        "CACHE_PREDICTIONS": CACHE_PREDICTIONS,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def normalize_text(row: pd.Series) -> str:
    if TEXT_SOURCE == "target_class":
        text = str(row["target_class"]).strip()
    else:
        text = str(row["prompt"]).strip()
    if not text or text.lower() == "nan":
        text = str(row.get("target_class", "object")).strip()
    return text


def load_image_pil(image_path: str) -> Image.Image:
    return Image.open(str(image_path)).convert("RGB")


def load_image_rgb(image_path: str) -> np.ndarray:
    return np.asarray(load_image_pil(image_path))


def read_gt_mask(label_path: str, gray_value: int) -> np.ndarray:
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"Cannot read label: {label_path}")
    return label == int(gray_value)


def ensure_mask_shape(mask: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask is None:
        return np.zeros((h, w), dtype=bool)

    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask[0]

    if mask.dtype == bool:
        out = mask
    else:
        try:
            out = mask > MASK_PROB_THRESHOLD if np.nanmax(mask) <= 1.0 else mask > 0
        except ValueError:
            out = np.zeros((h, w), dtype=bool)

    if out.shape != (h, w):
        out = cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    return out.astype(bool)


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
    mask = np.asarray(mask).astype(bool)
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


def save_overlay(overlay_path: Path, image_rgb: np.ndarray, pred_mask: np.ndarray, gt_mask=None, box=None, score=np.nan):
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    pred_mask = ensure_mask_shape(pred_mask, image_rgb.shape[:2])
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    if pred_mask.any():
        red = np.zeros_like(vis)
        red[:, :, 2] = 255
        vis[pred_mask] = cv2.addWeighted(vis, 0.45, red, 0.55, 0)[pred_mask]

    if gt_mask is not None and np.asarray(gt_mask).any():
        gt_mask = ensure_mask_shape(gt_mask, image_rgb.shape[:2]).astype(np.uint8)
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    if box is not None:
        box = np.asarray(box).reshape(-1)
        if len(box) == 4 and not np.any(np.isnan(box)):
            x1, y1, x2, y2 = box.astype(int).tolist()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"LangSAM {score:.2f}" if not np.isnan(score) else "LangSAM"
            cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(overlay_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def normalize_masks_array(masks):
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
                return np.zeros((0, 1, 1), dtype=np.float32)
    else:
        return np.zeros((0, 1, 1), dtype=np.float32)
    return masks.astype(np.float32)


def normalize_scores(scores, n):
    scores = to_numpy(scores)
    if scores is None:
        return np.ones((n,), dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) < n:
        scores = np.pad(scores, (0, n - len(scores)), constant_values=np.nan)
    return scores[:n]


def normalize_boxes(boxes, n):
    boxes = to_numpy(boxes)
    if boxes is None:
        return np.full((n, 4), np.nan, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.full((n, 4), np.nan, dtype=np.float32)
    boxes = boxes.reshape(-1, 4)
    if len(boxes) < n:
        pad = np.full((n - len(boxes), 4), np.nan, dtype=np.float32)
        boxes = np.concatenate([boxes, pad], axis=0)
    return boxes[:n]


def parse_langsam_result(raw_result: Any) -> Dict[str, Any]:
    if isinstance(raw_result, list):
        if len(raw_result) == 0:
            return {"masks": None, "boxes": None, "scores": None, "phrases": [], "raw_keys": []}
        raw_result = raw_result[0]

    if isinstance(raw_result, dict):
        masks = None
        for k in ["masks", "mask", "pred_masks"]:
            if k in raw_result:
                masks = raw_result[k]
                break

        boxes = None
        for k in ["boxes", "box", "bboxes", "pred_boxes"]:
            if k in raw_result:
                boxes = raw_result[k]
                break

        scores = None
        for k in ["scores", "score", "logits", "confidences", "confidence"]:
            if k in raw_result:
                scores = raw_result[k]
                break

        phrases = raw_result.get("phrases", raw_result.get("labels", raw_result.get("texts", [])))
        return {"masks": masks, "boxes": boxes, "scores": scores, "phrases": phrases, "raw_keys": list(raw_result.keys())}

    if isinstance(raw_result, tuple):
        if len(raw_result) >= 4:
            masks, boxes, phrases, logits = raw_result[:4]
            return {"masks": masks, "boxes": boxes, "scores": logits, "phrases": phrases, "raw_keys": ["tuple_masks", "tuple_boxes", "tuple_phrases", "tuple_logits"]}
        if len(raw_result) >= 2:
            masks, boxes = raw_result[:2]
            return {"masks": masks, "boxes": boxes, "scores": None, "phrases": [], "raw_keys": ["tuple_masks", "tuple_boxes"]}

    return {"masks": None, "boxes": None, "scores": None, "phrases": [], "raw_keys": [str(type(raw_result))]}


def select_candidate(parsed: Dict[str, Any]):
    masks = normalize_masks_array(parsed.get("masks", None))
    n = int(masks.shape[0])
    boxes = normalize_boxes(parsed.get("boxes", None), n)
    scores = normalize_scores(parsed.get("scores", None), n)

    if n == 0:
        return {
            "pred_mask": None,
            "selected_idx": -1,
            "selected_score": np.nan,
            "selected_box": np.full((4,), np.nan, dtype=np.float32),
            "num_masks": 0,
            "phrases": parsed.get("phrases", []),
            "raw_keys": parsed.get("raw_keys", []),
        }

    bool_masks = [ensure_mask_shape(masks[i], masks[i].shape[-2:]) for i in range(n)]

    if SELECT_MASK_MODE == "largest_mask":
        idx = int(np.argmax([int(m.sum()) for m in bool_masks]))
    elif SELECT_MASK_MODE == "top_score" and scores is not None and len(scores) >= n and not np.all(np.isnan(scores[:n])):
        idx = int(np.nanargmax(scores[:n]))
    else:
        idx = 0

    return {
        "pred_mask": bool_masks[idx],
        "selected_idx": idx,
        "selected_score": float(scores[idx]) if idx < len(scores) and not np.isnan(scores[idx]) else np.nan,
        "selected_box": boxes[idx] if idx < len(boxes) else np.full((4,), np.nan, dtype=np.float32),
        "num_masks": n,
        "phrases": parsed.get("phrases", []),
        "raw_keys": parsed.get("raw_keys", []),
    }


class LangSAMImageTextPredictor:
    def __init__(self):
        self.model = None
        self.cache = {}

    def load(self):
        print("\nLoading LangSAM...")
        print(f"  repo root: {LANGSAM_REPO_ROOT}")
        print(f"  device:    {DEVICE}")
        print(f"  HF mirror: {os.environ.get('HF_ENDPOINT', '')}")
        if not LANGSAM_REPO_ROOT.exists():
            raise FileNotFoundError(f"LangSAM repo not found: {LANGSAM_REPO_ROOT}")
        self.model = LangSAM()
        print("LangSAM loaded.")

    @torch.no_grad()
    def predict(self, image_path: str, text_prompt: str) -> Dict[str, Any]:
        key = (str(image_path), str(text_prompt))
        if CACHE_PREDICTIONS and key in self.cache:
            return dict(self.cache[key])

        image_pil = load_image_pil(image_path)
        raw = None
        last_errors = []

        # Current README API.
        try:
            raw = self.model.predict([image_pil], [str(text_prompt)])
        except Exception as e:
            last_errors.append(repr(e))

        # Some package versions use named args.
        if raw is None:
            for kwargs in [
                {"images_pil": [image_pil], "texts_prompt": [str(text_prompt)]},
                {"images": [image_pil], "texts": [str(text_prompt)]},
            ]:
                try:
                    raw = self.model.predict(**kwargs)
                    break
                except Exception as e:
                    last_errors.append(repr(e))

        # Older API.
        if raw is None:
            try:
                raw = self.model.predict(
                    image_pil,
                    str(text_prompt),
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                )
            except Exception as e:
                last_errors.append(repr(e))

        if raw is None:
            try:
                raw = self.model.predict(image_pil, str(text_prompt), BOX_THRESHOLD, TEXT_THRESHOLD)
            except Exception as e:
                last_errors.append(repr(e))
                raise RuntimeError(f"LangSAM predict failed. Errors: {' | '.join(last_errors)}")

        parsed = parse_langsam_result(raw)
        selected = select_candidate(parsed)

        result = {
            "pred_mask": selected["pred_mask"],
            "num_masks": int(selected["num_masks"]),
            "selected_idx": int(selected["selected_idx"]),
            "selected_score": selected["selected_score"],
            "selected_box": selected["selected_box"],
            "phrases": selected["phrases"],
            "raw_keys": selected["raw_keys"],
        }

        if CACHE_PREDICTIONS:
            self.cache[key] = dict(result)

        return result


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
                mean_langsam_score=("langsam_score", "mean"),
                mean_pred_area=("pred_area", "mean"),
                mean_gt_area=("gt_area", "mean"),
                mean_num_masks=("num_masks", "mean"),
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
                mean_langsam_score=("mean_langsam_score", "mean"),
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
    print("LangSAM Image-Level Text-Prompt Evaluation on EgoMed5 Test Online Schedule")
    print("=" * 120)
    print(f"Prompt CSV:          {PROMPT_CSV}")
    print(f"LangSAM repo root:   {LANGSAM_REPO_ROOT}")
    print(f"Device:              {DEVICE}")
    print(f"Run root:            {RUN_ROOT}")
    print(f"Text source:         {TEXT_SOURCE}")
    print(f"Select mask mode:    {SELECT_MASK_MODE}")
    print(f"Box threshold:       {BOX_THRESHOLD}")
    print(f"Text threshold:      {TEXT_THRESHOLD}")
    print(f"Save pred masks:     {SAVE_PRED_MASK}")
    print(f"Pred mask root:      {PRED_MASK_ROOT}")
    print(f"Save overlay:        {SAVE_PRED_OVERLAY}")
    print(f"Overlay root:        {PRED_OVERLAY_ROOT if SAVE_PRED_OVERLAY else 'disabled'}")
    print(f"Save empty masks:    {SAVE_EMPTY_MASKS}")
    print(f"Max rows:            {MAX_ROWS}")
    print("=" * 120)

    df = pd.read_csv(PROMPT_CSV)
    required_cols = [
        "dataset", "case_id", "frame_idx", "frame_name", "image_path", "label_path",
        "prompt", "prompt_type", "target_class", "target_gray_value",
        "case_prompt_id", "prompt_activate_frame", "gt_exists",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in prompt CSV: {col}")

    df["dataset"] = df["dataset"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df = df.sort_values(["dataset", "case_id", "frame_idx", "case_prompt_id"]).reset_index(drop=True)

    if MAX_ROWS is not None:
        df = df.head(int(MAX_ROWS)).copy()

    predictor = LangSAMImageTextPredictor()
    predictor.load()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="LangSAM image eval"):
        dataset = str(row["dataset"])
        case_id = str(row["case_id"])
        frame_idx = int(row["frame_idx"])
        frame_name = str(row["frame_name"])
        image_path = str(row["image_path"])
        label_path = str(row["label_path"])
        prompt = str(row["prompt"])
        langsam_text_prompt = normalize_text(row)
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

            out = predictor.predict(image_path=image_path, text_prompt=langsam_text_prompt)
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
                    image_rgb = load_image_rgb(image_path)
                    save_overlay(overlay_path, image_rgb, pred_mask, gt_mask=gt_mask, box=out["selected_box"], score=out["selected_score"])
                    pred_overlay_path = str(overlay_path)

            rows.append({
                "dataset": dataset,
                "case_id": case_id,
                "frame_idx": frame_idx,
                "frame_name": frame_name,
                "image_path": image_path,
                "label_path": label_path,
                "prompt": prompt,
                "langsam_text_prompt": langsam_text_prompt,
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
                "langsam_score": float(out["selected_score"]) if not np.isnan(out["selected_score"]) else np.nan,
                "langsam_box": bbox_to_str(out["selected_box"]),
                "langsam_phrases": "|".join([str(x) for x in out.get("phrases", [])]),
                "langsam_raw_keys": "|".join([str(x) for x in out.get("raw_keys", [])]),
                "absent_false_positive": bool(absent_false_positive),
                "absent_empty_correct": bool(absent_empty_correct),
                "pred_mask_path": pred_mask_path,
                "pred_overlay_path": pred_overlay_path,
                "error": "",
            })

            del gt_mask, pred_mask, out

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
                "langsam_text_prompt": langsam_text_prompt,
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
                "langsam_score": np.nan,
                "langsam_box": "",
                "langsam_phrases": "",
                "langsam_raw_keys": "",
                "absent_false_positive": np.nan,
                "absent_empty_correct": np.nan,
                "pred_mask_path": pred_mask_path,
                "pred_overlay_path": pred_overlay_path,
                "error": repr(e),
            })

    frame_df = pd.DataFrame(rows)
    case_df, class_summary, absent_summary = summarize(frame_df)

    out_xlsx = RUN_ROOT / "egomed5_langsam_image_text_prompt_eval.xlsx"
    frame_csv = RUN_ROOT / "frame_prompt_metrics.csv"
    case_csv = RUN_ROOT / "case_prompt_metrics_gt_exists.csv"
    summary_csv = RUN_ROOT / "class_prompt_summary_gt_exists.csv"
    absent_csv = RUN_ROOT / "absent_frame_summary.csv"

    frame_df.to_csv(frame_csv, index=False)
    case_df.to_csv(case_csv, index=False)
    class_summary.to_csv(summary_csv, index=False)
    absent_summary.to_csv(absent_csv, index=False)

    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            frame_df.to_excel(writer, sheet_name="frame_prompt_metrics", index=False)
            case_df.to_excel(writer, sheet_name="case_prompt_metrics", index=False)
            class_summary.to_excel(writer, sheet_name="class_prompt_summary", index=False)
            absent_summary.to_excel(writer, sheet_name="absent_frame_summary", index=False)
    except ModuleNotFoundError as e:
        print(f"[Warning] Excel skipped because dependency missing: {repr(e)}")
        print("CSV files were saved successfully.")

    elapsed = time.time() - start_time

    print("\n" + "=" * 120)
    print("LangSAM image evaluation done.")
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
        cols = [
            "dataset", "prompt_type", "prompt", "target_class",
            "mean_dice", "std_dice", "mean_mask_iou", "mean_bbox_iou",
            "mean_langsam_score", "mean_num_masks", "num_cases", "total_gt_exists_frames",
        ]
        existing = [c for c in cols if c in class_summary.columns]
        print("\nClass prompt summary:")
        print(class_summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
