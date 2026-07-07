import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
EXT_ROOT = _Path(_os.environ.get("EGOMED_EXT_ROOT", REPO_ROOT.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# SAM3 video builder may still need facebook/sam3/config.json.
# Do NOT force offline mode unless config.json is already cached locally.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
import time
import json
import inspect
import shutil
import traceback
import gc
from datetime import datetime
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

PROMPT_FRAME_CSV = Path(
    f"{REPO_ROOT}/data/text_prompt_eval/"
    "egomed5_test_text_prompts_online_schedule.csv"
)

PROMPT_SCHEDULE_CSV = Path(
    f"{REPO_ROOT}/data/text_prompt_eval/"
    "egomed5_test_prompt_schedule.csv"
)

SAM3_CHECKPOINT = Path(f"{EXT_ROOT}/sam3/checkpoints/sam3.pt")

RUN_ROOT = (
    Path(f"{REPO_ROOT}/runs/eval_sam3_text_prompt")
    / "egomed5_sam3_video_text_prompt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM3 video examples use a JPEG folder as resource_path. This script converts each case
# to a cached sequential JPEG folder so file names are safe for SAM3.
FRAME_CACHE_ROOT = RUN_ROOT / "_sam3_video_jpeg_cache"
REBUILD_FRAME_CACHE = True

# When a text prompt creates multiple object IDs, select one without using GT.
# "top_score": choose the object with highest SAM3 output probability at activation.
# "first": choose the first returned object.
SELECT_OBJECT_MODE = "top_score"

# Save predicted masks and optional visualization overlays.
# SAVE_EMPTY_MASKS=True saves every prompt-frame mask, including all-black masks.
SAVE_PRED_MASK = True
SAVE_PRED_OVERLAY = False
SAVE_EMPTY_MASKS = False

PRED_MASK_ROOT = RUN_ROOT / "pred_mask"
PRED_OVERLAY_ROOT = RUN_ROOT / "pred_overlay"

# Main Dice summary uses only frames where GT exists.
SUMMARY_ONLY_GT_EXISTS = True

# Logging outputs.
LOG_ROOT = RUN_ROOT / "logs"
LOG_FILE = LOG_ROOT / "run.log"
ERROR_LOG_FILE = LOG_ROOT / "error.log"
CONFIG_JSON = LOG_ROOT / "config.json"



# ============================================================
# Logging helpers
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
        "PROMPT_FRAME_CSV": str(PROMPT_FRAME_CSV),
        "PROMPT_SCHEDULE_CSV": str(PROMPT_SCHEDULE_CSV),
        "SAM3_CHECKPOINT": str(SAM3_CHECKPOINT),
        "RUN_ROOT": str(RUN_ROOT),
        "DEVICE": DEVICE,
        "FRAME_CACHE_ROOT": str(FRAME_CACHE_ROOT),
        "REBUILD_FRAME_CACHE": REBUILD_FRAME_CACHE,
        "SELECT_OBJECT_MODE": SELECT_OBJECT_MODE,
        "SAVE_PRED_MASK": SAVE_PRED_MASK,
        "SAVE_PRED_OVERLAY": SAVE_PRED_OVERLAY,
        "SAVE_EMPTY_MASKS": SAVE_EMPTY_MASKS,
        "PRED_MASK_ROOT": str(PRED_MASK_ROOT),
        "PRED_OVERLAY_ROOT": str(PRED_OVERLAY_ROOT),
        "SUMMARY_ONLY_GT_EXISTS": SUMMARY_ONLY_GT_EXISTS,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

# ============================================================
# Metrics and mask utilities
# ============================================================

def dice_score(pred_mask, gt_mask, eps=1e-6):
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


def mask_iou(pred_mask, gt_mask, eps=1e-6):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    union = int(np.logical_or(pred_mask, gt_mask).sum())
    if union == 0:
        return 1.0

    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    return float((inter + eps) / (union + eps))


def mask_to_bbox(mask):
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_to_str(box):
    if box is None:
        return ""
    return ",".join([f"{float(x):.2f}" for x in box])


def bbox_area(box):
    if box is None:
        return 0.0
    return max(0.0, float(box[2] - box[0] + 1.0)) * max(
        0.0, float(box[3] - box[1] + 1.0)
    )


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

    area1 = bbox_area(box1)
    area2 = bbox_area(box2)
    return float(inter / (area1 + area2 - inter + eps))


def read_gt_mask(label_path, gray_value):
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"Cannot read label: {label_path}")
    return label == int(gray_value)


def ensure_mask_shape(mask, shape_hw):
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


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        # bfloat16 cannot be converted directly to numpy.
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


def extract_video_outputs(outputs):
    """
    Convert SAM3 video output dict to:
      obj_ids: list[int]
      masks: list[np.ndarray bool H W]
      scores: list[float]

    Official visualization utilities indicate output keys:
      out_obj_ids
      out_binary_masks
      out_probs
    """
    if outputs is None:
        return [], [], []

    if not isinstance(outputs, dict):
        raise ValueError(f"Unsupported SAM3 video outputs type: {type(outputs)}")

    obj_ids = None
    for key in ["out_obj_ids", "obj_ids", "object_ids"]:
        if key in outputs:
            obj_ids = to_numpy(outputs[key])
            break

    masks = None
    for key in ["out_binary_masks", "masks", "pred_masks"]:
        if key in outputs:
            masks = to_numpy(outputs[key])
            break

    scores = None
    for key in ["out_probs", "scores", "pred_scores", "object_scores"]:
        if key in outputs:
            scores = to_numpy(outputs[key])
            break

    if obj_ids is None or masks is None:
        return [], [], []

    obj_ids = np.asarray(obj_ids).reshape(-1).astype(int)

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

    masks = masks.astype(bool)

    if scores is None:
        scores = np.ones((len(obj_ids),), dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores) < len(obj_ids):
            scores = np.pad(scores, (0, len(obj_ids) - len(scores)), constant_values=np.nan)

    out_masks = []
    out_scores = []
    for i in range(len(obj_ids)):
        out_masks.append(masks[i])
        out_scores.append(float(scores[i]) if i < len(scores) else np.nan)

    return obj_ids.tolist(), out_masks, out_scores


def outputs_to_obj_mask_score(outputs):
    obj_ids, masks, scores = extract_video_outputs(outputs)
    d = {}
    for obj_id, mask, score in zip(obj_ids, masks, scores):
        d[int(obj_id)] = {
            "mask": mask,
            "score": score,
        }
    return d


# ============================================================
# Save prediction helpers
# ============================================================

def safe_name(x):
    x = str(x)
    for ch in ["/", "\\", ":", " ", "\t", "\n", "|", ",", ";"]:
        x = x.replace(ch, "_")
    return x


def make_pred_paths(dataset, case_id, frame_idx, frame_name, prompt_id, target_class):
    stem = Path(str(frame_name)).stem
    cls = safe_name(target_class)
    rel = Path(str(dataset)) / str(case_id)
    filename = f"{int(frame_idx):06d}_{stem}_prompt{int(prompt_id):03d}_{cls}.png"
    overlay_filename = f"{int(frame_idx):06d}_{stem}_prompt{int(prompt_id):03d}_{cls}.jpg"
    return PRED_MASK_ROOT / rel / filename, PRED_OVERLAY_ROOT / rel / overlay_filename


def save_pred_mask(mask_path, pred_mask):
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), pred_mask.astype(np.uint8) * 255)


def save_pred_overlay(overlay_path, image_path, pred_mask, gt_mask=None):
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    pred_mask = ensure_mask_shape(pred_mask, img.shape[:2])
    vis = img.copy()

    # Red overlay for prediction.
    if pred_mask.any():
        red = np.zeros_like(vis)
        red[:, :, 2] = 255
        vis[pred_mask] = cv2.addWeighted(vis, 0.45, red, 0.55, 0)[pred_mask]

    # Green contour for GT, useful for quick visual checking.
    if gt_mask is not None and np.asarray(gt_mask).any():
        gt_mask = ensure_mask_shape(gt_mask, img.shape[:2]).astype(np.uint8)
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    cv2.imwrite(str(overlay_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return True


# ============================================================
# SAM3 video loading and API helpers
# ============================================================

def build_sam3_video_predictor_local():
    from sam3.model_builder import build_sam3_video_predictor

    print("\nLoading SAM3 video predictor...")
    print(f"  checkpoint: {SAM3_CHECKPOINT}")
    print(f"  device:     {DEVICE}")

    if not SAM3_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {SAM3_CHECKPOINT}\n"
            "Please modify SAM3_CHECKPOINT in this script."
        )

    # IMPORTANT:
    # Current SAM3 video predictor accepts checkpoint_path.
    # Do not call build_sam3_video_predictor() without checkpoint_path,
    # otherwise it will try to download facebook/sam3 from Hugging Face.
    kwargs = {
        "checkpoint_path": str(SAM3_CHECKPOINT),
    }

    if DEVICE == "cuda":
        kwargs["gpus_to_use"] = [torch.cuda.current_device()]
    else:
        kwargs["gpus_to_use"] = []

    print(f"  builder kwargs: {kwargs}")

    predictor = build_sam3_video_predictor(**kwargs)

    print("SAM3 video predictor loaded.")
    return predictor


def start_session(predictor, resource_path):
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(resource_path),
        )
    )
    return response["session_id"]


def close_session(predictor, session_id):
    for req_type in ["close_session", "end_session"]:
        try:
            predictor.handle_request(
                request=dict(
                    type=req_type,
                    session_id=session_id,
                )
            )
            return
        except Exception:
            pass


def add_text_prompt(predictor, session_id, frame_idx, prompt):
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=int(frame_idx),
            text=str(prompt),
        )
    )
    return response.get("outputs", None)


def propagate_in_video(predictor, session_id):
    """
    Stream SAM3 video outputs frame-by-frame.

    IMPORTANT:
    The previous version stored all frame outputs in a dict:
        outputs_per_frame[frame_index] = outputs

    That can consume huge CPU memory because each outputs contains masks.
    This generator yields one frame at a time so the caller can evaluate and
    immediately release it.
    """
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        yield int(response["frame_index"]), response["outputs"]


# ============================================================
# Frame cache
# ============================================================

def prepare_case_jpeg_cache(case_df, dataset, case_id):
    """
    SAM3 video API expects a JPEG frame folder with contiguous local indices:
      000000.jpg, 000001.jpg, ...

    The original frame_idx in the CSV may not be equal to SAM3's local frame index.
    This function returns frame_info with:
      frame_idx: original frame index from CSV
      local_frame_idx: contiguous SAM3 frame index
    """
    frame_info = (
        case_df[["frame_idx", "image_path", "label_path", "frame_name"]]
        .drop_duplicates("frame_idx")
        .sort_values("frame_idx")
        .reset_index(drop=True)
    )

    frame_info = frame_info.copy()
    frame_info["local_frame_idx"] = np.arange(len(frame_info), dtype=int)

    cache_dir = FRAME_CACHE_ROOT / str(dataset) / str(case_id)

    expected = len(frame_info)
    expected_files = {f"{i:06d}.jpg" for i in range(expected)}
    existing_files = {p.name for p in cache_dir.glob("*.jpg")} if cache_dir.exists() else set()

    cache_ok = cache_dir.exists() and existing_files == expected_files

    if REBUILD_FRAME_CACHE and cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_ok = False

    if not cache_ok:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(
            frame_info.iterrows(),
            total=len(frame_info),
            desc=f"cache jpg {dataset}/{case_id}",
            leave=False,
        ):
            local_frame_idx = int(row["local_frame_idx"])
            src = str(row["image_path"])
            dst = cache_dir / f"{local_frame_idx:06d}.jpg"

            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {src}")

            cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return cache_dir, frame_info


# ============================================================
# Prompt-object mapping
# ============================================================

def choose_object_for_prompt(outputs, already_assigned_obj_ids):
    """
    SAM3 text prompt may create multiple object IDs.
    Select one object without using GT.

    Prefer newly introduced object IDs. If none are new, fall back to all objects.
    """
    obj_ids, masks, scores = extract_video_outputs(outputs)

    if len(obj_ids) == 0:
        return None, np.nan, 0

    candidates = []
    for obj_id, score in zip(obj_ids, scores):
        is_new = int(obj_id) not in already_assigned_obj_ids
        candidates.append({
            "obj_id": int(obj_id),
            "score": float(score) if not np.isnan(score) else -1e9,
            "is_new": is_new,
        })

    new_candidates = [c for c in candidates if c["is_new"]]
    if len(new_candidates) > 0:
        candidates = new_candidates

    if SELECT_OBJECT_MODE == "top_score":
        chosen = sorted(candidates, key=lambda c: c["score"], reverse=True)[0]
    else:
        chosen = candidates[0]

    return int(chosen["obj_id"]), float(chosen["score"]), len(obj_ids)


# ============================================================
# Case evaluation
# ============================================================

def evaluate_case(predictor, case_df, schedule_df, dataset, case_id):
    cache_dir, frame_info = prepare_case_jpeg_cache(case_df, dataset, case_id)

    orig_to_local = {
        int(r["frame_idx"]): int(r["local_frame_idx"])
        for _, r in frame_info.iterrows()
    }

    local_to_orig = {
        int(r["local_frame_idx"]): int(r["frame_idx"])
        for _, r in frame_info.iterrows()
    }

    session_id = None
    rows = []
    activation_rows = []

    try:
        session_id = start_session(predictor, cache_dir)

        # Add prompts in activation order.
        prompt_to_obj = {}
        assigned_obj_ids = set()

        schedule_case = (
            schedule_df[
                (schedule_df["dataset"].astype(str) == str(dataset))
                & (schedule_df["case_id"].astype(str) == str(case_id))
            ]
            .sort_values(["prompt_activate_frame", "case_prompt_id"])
            .reset_index(drop=True)
        )

        for _, p in schedule_case.iterrows():
            prompt_id = int(p["case_prompt_id"])
            orig_activate_frame_idx = int(p["prompt_activate_frame"])
            prompt = str(p["prompt"])

            if orig_activate_frame_idx not in orig_to_local:
                raise ValueError(
                    f"prompt_activate_frame={orig_activate_frame_idx} not found in loaded frames "
                    f"for {dataset}/{case_id}. "
                    f"Available original frame range: "
                    f"{min(orig_to_local.keys())} ~ {max(orig_to_local.keys())}, "
                    f"num_loaded_frames={len(orig_to_local)}"
                )

            local_activate_frame_idx = orig_to_local[orig_activate_frame_idx]

            outputs = add_text_prompt(
                predictor=predictor,
                session_id=session_id,
                frame_idx=local_activate_frame_idx,
                prompt=prompt,
            )

            chosen_obj_id, activation_score, num_objects_returned = choose_object_for_prompt(
                outputs=outputs,
                already_assigned_obj_ids=assigned_obj_ids,
            )
            del outputs

            if chosen_obj_id is not None:
                assigned_obj_ids.add(chosen_obj_id)

            prompt_to_obj[prompt_id] = {
                "obj_id": chosen_obj_id,
                "activation_score": activation_score,
                "num_objects_returned": num_objects_returned,
            }

            activation_rows.append({
                "dataset": dataset,
                "case_id": case_id,
                "case_prompt_id": prompt_id,
                "prompt": prompt,
                "prompt_type": p["prompt_type"],
                "target_class": p["target_class"],
                "target_gray_value": int(p["target_gray_value"]),
                "prompt_activate_frame": orig_activate_frame_idx,
                "sam3_local_prompt_activate_frame": local_activate_frame_idx,
                "chosen_sam3_obj_id": chosen_obj_id if chosen_obj_id is not None else np.nan,
                "activation_score": activation_score,
                "num_objects_returned_at_activation": num_objects_returned,
            })

        # Propagate once after all scheduled prompts have been added.
        # Memory-safe version:
        #   process one streamed frame at a time
        #   do NOT keep outputs_per_frame / outputs_per_local_frame in memory.
        sorted_case_df = case_df.sort_values(["frame_idx", "case_prompt_id"]).copy()
        rows_by_frame = {
            int(frame_idx): frame_rows.copy()
            for frame_idx, frame_rows in sorted_case_df.groupby("frame_idx", sort=True)
        }
        processed_frame_indices = set()

        def evaluate_prompt_rows_for_frame(frame_idx, frame_outputs):
            frame_idx = int(frame_idx)
            frame_rows = rows_by_frame.get(frame_idx, None)
            if frame_rows is None or len(frame_rows) == 0:
                return

            processed_frame_indices.add(frame_idx)
            local_frame_idx = orig_to_local.get(frame_idx, np.nan)

            # Convert this frame output once, then immediately release after this frame.
            obj_mask_score = outputs_to_obj_mask_score(frame_outputs)

            for _, row in frame_rows.iterrows():
                prompt_id = int(row["case_prompt_id"])
                gray_value = int(row["target_gray_value"])
                gt_exists = bool(row["gt_exists"])

                gt_mask = read_gt_mask(str(row["label_path"]), gray_value)
                gt_h, gt_w = gt_mask.shape[:2]

                obj_info = prompt_to_obj.get(prompt_id, {})
                obj_id = obj_info.get("obj_id", None)
                activation_score = obj_info.get("activation_score", np.nan)

                if obj_id is None or obj_id not in obj_mask_score:
                    pred_mask = np.zeros_like(gt_mask, dtype=bool)
                    sam3_score = np.nan
                    object_found_in_frame = False
                else:
                    pred_mask = ensure_mask_shape(obj_mask_score[obj_id]["mask"], (gt_h, gt_w))
                    sam3_score = obj_mask_score[obj_id]["score"]
                    object_found_in_frame = True

                d = dice_score(pred_mask, gt_mask)
                miou = mask_iou(pred_mask, gt_mask)

                pred_area = int(pred_mask.sum())
                gt_area = int(gt_mask.sum())

                pred_bbox = mask_to_bbox(pred_mask)
                gt_bbox = mask_to_bbox(gt_mask)
                biou = bbox_iou(pred_bbox, gt_bbox) if gt_bbox is not None else np.nan

                absent_false_positive = (not gt_exists) and pred_area > 0
                absent_empty_correct = (not gt_exists) and pred_area == 0

                pred_mask_path = ""
                pred_overlay_path = ""

                if SAVE_PRED_MASK or SAVE_PRED_OVERLAY:
                    mask_path, overlay_path = make_pred_paths(
                        dataset=dataset,
                        case_id=case_id,
                        frame_idx=frame_idx,
                        frame_name=row["frame_name"],
                        prompt_id=prompt_id,
                        target_class=row["target_class"],
                    )

                    should_save = SAVE_EMPTY_MASKS or pred_area > 0
                    if should_save and SAVE_PRED_MASK:
                        save_pred_mask(mask_path, pred_mask)
                        pred_mask_path = str(mask_path)

                    if should_save and SAVE_PRED_OVERLAY:
                        ok = save_pred_overlay(
                            overlay_path=overlay_path,
                            image_path=row["image_path"],
                            pred_mask=pred_mask,
                            gt_mask=gt_mask,
                        )
                        if ok:
                            pred_overlay_path = str(overlay_path)

                rows.append({
                    "dataset": dataset,
                    "case_id": case_id,
                    "frame_idx": frame_idx,
                    "sam3_local_frame_idx": local_frame_idx,
                    "frame_name": row["frame_name"],
                    "image_path": row["image_path"],
                    "label_path": row["label_path"],

                    "prompt": row["prompt"],
                    "prompt_type": row["prompt_type"],
                    "target_class": row["target_class"],
                    "target_gray_value": gray_value,
                    "case_prompt_id": prompt_id,
                    "prompt_activate_frame": int(row["prompt_activate_frame"]),

                    "chosen_sam3_obj_id": obj_id if obj_id is not None else np.nan,
                    "activation_score": activation_score,
                    "object_found_in_frame": bool(object_found_in_frame),

                    "gt_exists": gt_exists,
                    "gt_area": gt_area,
                    "pred_area": pred_area,
                    "gt_bbox": bbox_to_str(gt_bbox),
                    "pred_bbox": bbox_to_str(pred_bbox),

                    "dice": float(d),
                    "mask_iou": float(miou),
                    "bbox_iou": float(biou) if not np.isnan(biou) else np.nan,
                    "sam3_score": float(sam3_score) if not np.isnan(sam3_score) else np.nan,

                    "absent_false_positive": bool(absent_false_positive),
                    "absent_empty_correct": bool(absent_empty_correct),
                    "pred_mask_path": pred_mask_path,
                    "pred_overlay_path": pred_overlay_path,
                })

                # Release large per-prompt arrays.
                del gt_mask, pred_mask

            # Release this frame's decoded masks.
            del obj_mask_score

        # Stream propagation outputs frame by frame.
        for local_idx, frame_outputs in tqdm(
            propagate_in_video(predictor, session_id=session_id),
            desc=f"propagate/eval {dataset}/{case_id}",
            leave=False,
        ):
            local_idx = int(local_idx)
            if local_idx not in local_to_orig:
                continue

            frame_idx = int(local_to_orig[local_idx])
            evaluate_prompt_rows_for_frame(frame_idx, frame_outputs)

            # Release raw SAM3 outputs for this frame immediately.
            del frame_outputs

        # If SAM3 did not stream some frames, evaluate them as empty masks.
        for frame_idx in sorted(rows_by_frame.keys()):
            if frame_idx in processed_frame_indices:
                continue
            evaluate_prompt_rows_for_frame(frame_idx, None)

        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finally:
        if session_id is not None:
            close_session(predictor, session_id)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows, activation_rows


# ============================================================
# Evaluation
# ============================================================

def summarize(frame_df):
    metric_df = frame_df[frame_df["gt_exists"] == True].copy() if SUMMARY_ONLY_GT_EXISTS else frame_df.copy()

    if len(metric_df) > 0:
        case_df = (
            metric_df.groupby(
                ["dataset", "case_id", "prompt_type", "prompt", "target_class", "target_gray_value"],
                dropna=False,
            )
            .agg(
                mean_dice=("dice", "mean"),
                std_dice=("dice", "std"),
                mean_mask_iou=("mask_iou", "mean"),
                std_mask_iou=("mask_iou", "std"),
                mean_bbox_iou=("bbox_iou", "mean"),
                mean_sam3_score=("sam3_score", "mean"),
                num_gt_exists_frames=("gt_exists", "count"),
                mean_gt_area=("gt_area", "mean"),
                mean_pred_area=("pred_area", "mean"),
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
                mean_sam3_score=("mean_sam3_score", "mean"),
                num_cases=("case_id", "count"),
                total_gt_exists_frames=("num_gt_exists_frames", "sum"),
                mean_gt_area=("mean_gt_area", "mean"),
                mean_pred_area=("mean_pred_area", "mean"),
            )
            .reset_index()
            .sort_values(["dataset", "prompt_type", "target_gray_value", "prompt"])
        )
    else:
        case_df = pd.DataFrame()
        class_summary = pd.DataFrame()

    absent_df = frame_df[frame_df["gt_exists"] == False].copy()
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
            absent_summary["num_absent_false_positive"] / absent_summary["num_absent_frames"].clip(lower=1)
        )
        absent_summary["absent_empty_correct_rate"] = (
            absent_summary["num_absent_empty_correct"] / absent_summary["num_absent_frames"].clip(lower=1)
        )
    else:
        absent_summary = pd.DataFrame()

    return case_df, class_summary, absent_summary


def evaluate():
    start_time = time.time()

    if not PROMPT_FRAME_CSV.exists():
        raise FileNotFoundError(f"Prompt-frame CSV not found: {PROMPT_FRAME_CSV}")
    if not PROMPT_SCHEDULE_CSV.exists():
        raise FileNotFoundError(f"Prompt schedule CSV not found: {PROMPT_SCHEDULE_CSV}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    FRAME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if SAVE_PRED_MASK:
        PRED_MASK_ROOT.mkdir(parents=True, exist_ok=True)
    if SAVE_PRED_OVERLAY:
        PRED_OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)
    save_config_json()

    print("=" * 120)
    print("SAM3 Video Text-Prompt Evaluation on EgoMed5 Test Online Schedule")
    print("=" * 120)
    print(f"Prompt-frame CSV: {PROMPT_FRAME_CSV}")
    print(f"Schedule CSV:     {PROMPT_SCHEDULE_CSV}")
    print(f"SAM3 checkpoint:  {SAM3_CHECKPOINT}")
    print(f"Device:           {DEVICE}")
    print(f"Run root:         {RUN_ROOT}")
    print(f"Frame cache root: {FRAME_CACHE_ROOT}")
    print(f"Save pred mask:   {SAVE_PRED_MASK}")
    print(f"Pred mask root:   {PRED_MASK_ROOT}")
    print(f"Save overlay:     {SAVE_PRED_OVERLAY}")
    print(f"Overlay root:     {PRED_OVERLAY_ROOT}")
    print(f"Save empty masks: {SAVE_EMPTY_MASKS}")
    print(f"Run log:          {LOG_FILE}")
    print(f"Error log:        {ERROR_LOG_FILE}")
    print("=" * 120)

    prompt_df = pd.read_csv(PROMPT_FRAME_CSV)
    schedule_df = pd.read_csv(PROMPT_SCHEDULE_CSV)

    # Normalize case_id to string for matching.
    prompt_df["case_id"] = prompt_df["case_id"].astype(str)
    schedule_df["case_id"] = schedule_df["case_id"].astype(str)
    prompt_df["dataset"] = prompt_df["dataset"].astype(str)
    schedule_df["dataset"] = schedule_df["dataset"].astype(str)

    required_frame_cols = [
        "dataset", "case_id", "frame_idx", "frame_name",
        "image_path", "label_path", "prompt", "prompt_type",
        "target_class", "target_gray_value", "case_prompt_id",
        "prompt_activate_frame", "gt_exists",
    ]
    for col in required_frame_cols:
        if col not in prompt_df.columns:
            raise ValueError(f"Missing required column in prompt-frame CSV: {col}")

    required_schedule_cols = [
        "dataset", "case_id", "case_prompt_id", "prompt", "prompt_type",
        "target_class", "target_gray_value", "prompt_activate_frame",
    ]
    for col in required_schedule_cols:
        if col not in schedule_df.columns:
            raise ValueError(f"Missing required column in schedule CSV: {col}")

    predictor = build_sam3_video_predictor_local()

    all_rows = []
    all_activation_rows = []

    cases = (
        prompt_df[["dataset", "case_id"]]
        .drop_duplicates()
        .sort_values(["dataset", "case_id"])
        .values
        .tolist()
    )

    for dataset, case_id in tqdm(cases, desc="cases"):
        case_df = prompt_df[
            (prompt_df["dataset"] == str(dataset))
            & (prompt_df["case_id"] == str(case_id))
        ].copy()

        case_schedule = schedule_df[
            (schedule_df["dataset"] == str(dataset))
            & (schedule_df["case_id"] == str(case_id))
        ].copy()

        if len(case_df) == 0 or len(case_schedule) == 0:
            continue

        print("\n" + "-" * 100)
        print(f"Evaluating SAM3 video case: {dataset} / {case_id}")
        print(f"  frames:  {case_df['frame_idx'].nunique()}")
        print(f"  prompts: {case_schedule['case_prompt_id'].nunique()}")
        print("-" * 100)

        rows, activation_rows = evaluate_case(
            predictor=predictor,
            case_df=case_df,
            schedule_df=schedule_df,
            dataset=str(dataset),
            case_id=str(case_id),
        )

        all_rows.extend(rows)
        all_activation_rows.extend(activation_rows)

    frame_df = pd.DataFrame(all_rows)
    activation_df = pd.DataFrame(all_activation_rows)

    case_df, class_summary, absent_summary = summarize(frame_df)

    out_xlsx = RUN_ROOT / "egomed5_sam3_video_text_prompt_eval.xlsx"
    frame_csv = RUN_ROOT / "frame_prompt_metrics.csv"
    activation_csv = RUN_ROOT / "prompt_activation_mapping.csv"
    case_csv = RUN_ROOT / "case_prompt_metrics_gt_exists.csv"
    summary_csv = RUN_ROOT / "class_prompt_summary_gt_exists.csv"
    absent_csv = RUN_ROOT / "absent_frame_summary.csv"

    frame_df.to_csv(frame_csv, index=False)
    activation_df.to_csv(activation_csv, index=False)
    case_df.to_csv(case_csv, index=False)
    class_summary.to_csv(summary_csv, index=False)
    absent_summary.to_csv(absent_csv, index=False)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        frame_df.to_excel(writer, sheet_name="frame_prompt_metrics", index=False)
        activation_df.to_excel(writer, sheet_name="prompt_activation_mapping", index=False)
        case_df.to_excel(writer, sheet_name="case_prompt_metrics", index=False)
        class_summary.to_excel(writer, sheet_name="class_prompt_summary", index=False)
        absent_summary.to_excel(writer, sheet_name="absent_frame_summary", index=False)

    print("\n" + "=" * 120)
    print("SAM3 video evaluation done.")
    print(f"Excel:                {out_xlsx}")
    print(f"Frame CSV:            {frame_csv}")
    print(f"Activation CSV:       {activation_csv}")
    print(f"Case CSV:             {case_csv}")
    print(f"Class summary CSV:    {summary_csv}")
    print(f"Absent summary CSV:   {absent_csv}")
    print(f"Pred mask root:       {PRED_MASK_ROOT if SAVE_PRED_MASK else 'disabled'}")
    print(f"Pred overlay root:    {PRED_OVERLAY_ROOT if SAVE_PRED_OVERLAY else 'disabled'}")
    print(f"Run log:              {LOG_FILE}")
    print(f"Error log:            {ERROR_LOG_FILE}")
    print("=" * 120)

    if len(class_summary) > 0:
        display_cols = [
            "dataset", "prompt_type", "prompt", "target_class",
            "mean_dice", "std_dice", "mean_mask_iou",
            "mean_sam3_score", "num_cases", "total_gt_exists_frames",
        ]
        print("\nClass prompt summary:")
        print(class_summary[display_cols].to_string(index=False))

    if len(absent_summary) > 0:
        display_cols = [
            "dataset", "prompt_type", "prompt", "target_class",
            "num_absent_frames", "absent_false_positive_rate",
            "mean_pred_area_absent",
        ]
        print("\nAbsent-frame summary:")
        print(absent_summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    setup_run_logging()
    evaluate()
