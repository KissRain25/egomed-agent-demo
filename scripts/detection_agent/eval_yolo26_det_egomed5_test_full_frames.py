import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# Config
# ============================================================

DATA_ROOT = Path(f"{REPO_ROOT}/data")
YOLO_RUN_ROOT = Path(f"{REPO_ROOT}/runs/yolo26_det")

DATASETS = [
    "Amos",
    "CAMUS",
    "ACDC",
    "Montgomery-County-CXR-Set",
    "PolypGen2021_MultiCenterData_v3",
]

# Amos only evaluates these 5 targets; other datasets evaluate all classes.
FOCUS_TARGETS_BY_DATASET = {
    "Amos": [
        "liver",
        "right kidney",
        "left kidney",
        "spleen",
        "stomach",
    ],
}

RUN_ROOT = Path(f"{REPO_ROOT}/runs/eval_yolo26_det") / "egomed5_yolo26m_imgsz1024_test_full_frames"

# Use a low threshold for AP calculation. Precision/Recall at 0.5 are computed from the best box per class per frame.
DET_CONF_THRES = 0.001
YOLO_IMGSZ = 1024
TP_IOU_THRES = 0.5
AP_IOU_THRES_LIST = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DEBUG_IMAGES = False


# ============================================================
# Basic utilities
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


def yolo_weight_path(dataset_name):
    return YOLO_RUN_ROOT / f"{dataset_name}_yolo26m_imgsz1024" / "weights" / "best.pt"


# ============================================================
# Excel / class mapping utilities
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
                    f"[Warning] {excel_file.name}: gray value {gray_value} conflict: "
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
        raise ValueError(f"{dataset_name}: focus targets not found in Excel gray mapping: {missing}")

    return filtered


def load_yolo_class_names_from_model(yolo_model):
    names = yolo_model.names
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: v for i, v in enumerate(names)}
    raise ValueError(f"Unsupported yolo_model.names format: {type(names)}")


def build_eval_class_mapping(dataset_name, yolo_model, excel_file):
    all_id_to_name = load_yolo_class_names_from_model(yolo_model)
    full_gray_mapping = build_dataset_gray_mapping(excel_file)
    gray_mapping = filter_gray_mapping(dataset_name, full_gray_mapping)
    allowed_names = set(gray_mapping.values())

    id_to_name = {
        class_id: class_name
        for class_id, class_name in all_id_to_name.items()
        if class_name in allowed_names
    }

    missing = allowed_names - set(id_to_name.values())
    if missing:
        raise ValueError(
            f"{dataset_name}: targets not found in YOLO checkpoint names: {missing}. "
            f"YOLO names are: {all_id_to_name}"
        )

    name_to_gray = {class_name: gray_value for gray_value, class_name in gray_mapping.items()}
    class_id_to_gray = {class_id: name_to_gray[class_name] for class_id, class_name in id_to_name.items()}
    return id_to_name, class_id_to_gray


# ============================================================
# BBox utilities
# ============================================================

def mask_to_bbox(mask):
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_area(box):
    if box is None:
        return 0.0
    return max(0.0, float(box[2] - box[0] + 1.0)) * max(0.0, float(box[3] - box[1] + 1.0))


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
    return float(inter / (bbox_area(box1) + bbox_area(box2) - inter + eps))


def bbox_to_str(box):
    if box is None:
        return ""
    return ",".join([f"{float(x):.2f}" for x in box])


def clip_bbox(box, w, h):
    if box is None:
        return None
    out = np.asarray(box, dtype=np.float32).copy()
    out[0] = np.clip(out[0], 0, w - 1)
    out[2] = np.clip(out[2], 0, w - 1)
    out[1] = np.clip(out[1], 0, h - 1)
    out[3] = np.clip(out[3], 0, h - 1)
    return out


def read_label(path):
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"Cannot read label: {path}")
    return label


def build_gt_boxes_from_label(label, class_id_to_gray):
    gt_by_class = {}
    for class_id, gray_value in class_id_to_gray.items():
        bbox = mask_to_bbox(label == int(gray_value))
        if bbox is not None:
            gt_by_class[class_id] = bbox
    return gt_by_class


def detect_one_frame_all_boxes(yolo_model, image_path, focus_class_ids):
    results = yolo_model.predict(
        source=str(image_path),
        conf=DET_CONF_THRES,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )

    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)

    preds = []
    for box, conf, class_id in zip(boxes, confs, clss):
        if class_id not in focus_class_ids:
            continue
        preds.append({"class_id": int(class_id), "box": box.astype(np.float32), "conf": float(conf)})

    return preds


def keep_best_pred_per_class(preds):
    best = {}
    for pred in preds:
        class_id = pred["class_id"]
        if class_id not in best or pred["conf"] > best[class_id]["conf"]:
            best[class_id] = pred
    return best


# ============================================================
# Metrics
# ============================================================

def compute_prf_from_counts(tp, fp, fn, eps=1e-12):
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float(2 * precision * recall / (precision + recall + eps))
    return precision, recall, f1


def compute_ap_for_class(pred_records, gt_records, iou_thr):
    n_gt = len(gt_records)
    if n_gt == 0:
        return np.nan

    gt_by_frame = {}
    for gt in gt_records:
        gt_by_frame.setdefault(gt["frame_uid"], []).append(gt)

    matched = {
        (frame_uid, idx): False
        for frame_uid, gts in gt_by_frame.items()
        for idx, gt in enumerate(gts)
    }

    preds_sorted = sorted(pred_records, key=lambda x: x["conf"], reverse=True)
    tp = np.zeros(len(preds_sorted), dtype=np.float32)
    fp = np.zeros(len(preds_sorted), dtype=np.float32)

    for i, pred in enumerate(preds_sorted):
        frame_uid = pred["frame_uid"]
        candidate_gts = gt_by_frame.get(frame_uid, [])
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(candidate_gts):
            if matched[(frame_uid, gt_idx)]:
                continue
            iou = bbox_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thr and best_gt_idx is not None:
            tp[i] = 1.0
            matched[(frame_uid, best_gt_idx)] = True
        else:
            fp[i] = 1.0

    if len(preds_sorted) == 0:
        return 0.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(float(n_gt), 1.0)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_class_summary(frame_df, pred_records_all, gt_records_all, id_to_name, class_id_to_gray):
    rows = []
    for class_id, class_name in id_to_name.items():
        class_frame_df = frame_df[frame_df["class_id"] == class_id].copy()
        tp = int((class_frame_df["match_status"] == "TP").sum())
        fp = int((class_frame_df["match_status"] == "FP").sum())
        fn = int((class_frame_df["match_status"] == "FN").sum())
        tn = int((class_frame_df["match_status"] == "TN").sum())
        low_iou = int((class_frame_df["match_status"] == "FP_FN_LOW_IOU").sum())

        # Low-IoU cases count as both FP and FN for PR.
        fp_for_pr = fp + low_iou
        fn_for_pr = fn + low_iou
        precision, recall, f1 = compute_prf_from_counts(tp, fp_for_pr, fn_for_pr)

        ious = class_frame_df.loc[class_frame_df["iou"].notna(), "iou"].values
        mean_iou_present = float(np.mean(ious)) if len(ious) > 0 else np.nan

        pred_records = [r for r in pred_records_all if r["class_id"] == class_id]
        gt_records = [r for r in gt_records_all if r["class_id"] == class_id]

        ap_by_thr = {}
        for thr in AP_IOU_THRES_LIST:
            ap_by_thr[f"AP{int(thr * 100):02d}"] = compute_ap_for_class(pred_records, gt_records, thr)
        valid_aps = [v for v in ap_by_thr.values() if not np.isnan(v)]
        ap50_95 = float(np.mean(valid_aps)) if valid_aps else np.nan

        rows.append({
            "class_id": int(class_id),
            "target_class": class_name,
            "label_gray_value": int(class_id_to_gray[class_id]),
            "num_gt_boxes": len(gt_records),
            "num_pred_boxes": len(pred_records),
            "TP": tp,
            "FP": fp_for_pr,
            "FN": fn_for_pr,
            "TN": tn,
            "low_iou_frames": low_iou,
            "precision_at_0.50": precision,
            "recall_at_0.50": recall,
            "f1_at_0.50": f1,
            "mean_iou_for_gt_pred_present": mean_iou_present,
            "AP50": ap_by_thr.get("AP50", np.nan),
            "AP75": ap_by_thr.get("AP75", np.nan),
            "AP50_95": ap50_95,
            **ap_by_thr,
        })

    return pd.DataFrame(rows)


# ============================================================
# Dataset evaluation
# ============================================================

def evaluate_dataset(dataset_name):
    dataset_root = DATA_ROOT / dataset_name
    img_root = dataset_root / "img"
    label_root = dataset_root / "label"
    excel_file = dataset_root / f"{dataset_name}.xlsx"
    yolo_det_root = DATA_ROOT / "yolo_det" / dataset_name
    split_file = yolo_det_root / "split_cases.txt"
    yolo_weight = yolo_weight_path(dataset_name)
    dataset_run_root = RUN_ROOT / dataset_name
    dataset_run_root.mkdir(parents=True, exist_ok=True)
    debug_img_root = dataset_run_root / "debug_images"
    excel_output = dataset_run_root / f"{dataset_name}_yolo26_det_test_metrics.xlsx"

    print("\n" + "=" * 120)
    print(f"Evaluating YOLO-det dataset: {dataset_name}")
    print("=" * 120)
    print(f"Image root:    {img_root}")
    print(f"Label root:    {label_root}")
    print(f"Excel file:    {excel_file}")
    print(f"Split file:    {split_file}")
    print(f"YOLO weight:   {yolo_weight}")
    print(f"Output Excel:  {excel_output}")
    print("=" * 120)

    for path, desc in [
        (img_root, "Image root"),
        (label_root, "Label root"),
        (excel_file, "Excel file"),
        (split_file, "split_cases.txt"),
        (yolo_weight, "YOLO weight"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    test_cases = read_split_cases(split_file, target_split="test")
    print(f"Number of test cases: {len(test_cases)}")
    print(f"First 10 test cases: {test_cases[:10]}")

    print("\nLoading YOLO detector...")
    yolo_model = YOLO(str(yolo_weight))
    id_to_name, class_id_to_gray = build_eval_class_mapping(dataset_name, yolo_model, excel_file)
    focus_class_ids = set(id_to_name.keys())

    print("\nClass mapping used for evaluation:")
    for class_id, class_name in id_to_name.items():
        print(f"  class_id={class_id}, class_name={class_name}, gray_value={class_id_to_gray[class_id]}")

    frame_rows = []
    case_rows = []
    pred_records_all = []
    gt_records_all = []
    false_positive_rows = []
    false_negative_rows = []

    for case_idx, case_id in enumerate(test_cases):
        print("\n" + "-" * 100)
        print(f"[{dataset_name}] [{case_idx + 1}/{len(test_cases)}] Evaluating case {case_id}")
        print("-" * 100)

        case_img_dir = img_root / case_id
        case_label_dir = label_root / case_id

        if not case_img_dir.exists():
            print(f"[Skip] Missing image dir: {case_img_dir}")
            continue
        if not case_label_dir.exists():
            print(f"[Skip] Missing label dir: {case_label_dir}")
            continue

        image_files = collect_image_files(case_img_dir)
        label_files = collect_label_files(case_label_dir)

        if len(image_files) == 0:
            print(f"[Skip] No images in {case_img_dir}")
            continue
        if len(label_files) == 0:
            print(f"[Skip] No labels in {case_label_dir}")
            continue

        if len(image_files) != len(label_files):
            print(
                f"[Warning] Count mismatch {dataset_name} case {case_id}: "
                f"images={len(image_files)}, labels={len(label_files)}. Using min length."
            )

        n = min(len(image_files), len(label_files))
        image_files = image_files[:n]
        label_files = label_files[:n]

        case_class_counts = {
            class_id: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "LOW_IOU": 0, "gt": 0, "pred": 0, "ious": []}
            for class_id in id_to_name.keys()
        }

        for frame_idx, (image_path, label_path) in enumerate(
            tqdm(list(zip(image_files, label_files)), desc=f"{dataset_name} case {case_id}", leave=False)
        ):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[Warning] Cannot read image: {image_path}")
                continue
            h, w = image.shape[:2]
            label = read_label(label_path)
            if label.shape[:2] != (h, w):
                raise ValueError(
                    f"Image/label size mismatch: image={image_path}, label={label_path}, "
                    f"image_shape={(h, w)}, label_shape={label.shape[:2]}"
                )

            frame_uid = f"{dataset_name}/{case_id}/{image_path.stem}"
            gt_by_class = build_gt_boxes_from_label(label, class_id_to_gray)
            preds_all = detect_one_frame_all_boxes(yolo_model, image_path, focus_class_ids)
            preds_best = keep_best_pred_per_class(preds_all)

            for pred in preds_all:
                class_id = pred["class_id"]
                if class_id not in id_to_name:
                    continue
                pred_box = clip_bbox(pred["box"], w=w, h=h)
                pred_records_all.append({
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "frame_idx": int(frame_idx),
                    "frame_name": image_path.name,
                    "frame_uid": frame_uid,
                    "class_id": int(class_id),
                    "class_name": id_to_name[class_id],
                    "conf": float(pred["conf"]),
                    "box": pred_box,
                    "bbox": bbox_to_str(pred_box),
                })

            for class_id, gt_box in gt_by_class.items():
                gt_records_all.append({
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "frame_idx": int(frame_idx),
                    "frame_name": image_path.name,
                    "frame_uid": frame_uid,
                    "class_id": int(class_id),
                    "class_name": id_to_name[class_id],
                    "box": gt_box,
                    "bbox": bbox_to_str(gt_box),
                })

            for class_id, class_name in id_to_name.items():
                gray_value = int(class_id_to_gray[class_id])
                gt_box = gt_by_class.get(class_id, None)
                pred = preds_best.get(class_id, None)
                pred_box = clip_bbox(pred["box"], w=w, h=h) if pred is not None else None
                pred_conf = float(pred["conf"]) if pred is not None else np.nan
                has_gt = gt_box is not None
                has_pred = pred_box is not None
                iou = bbox_iou(pred_box, gt_box) if has_pred and has_gt else np.nan

                if has_gt:
                    case_class_counts[class_id]["gt"] += 1
                if has_pred:
                    case_class_counts[class_id]["pred"] += 1

                if has_gt and has_pred and iou >= TP_IOU_THRES:
                    match_status = "TP"
                    case_class_counts[class_id]["TP"] += 1
                    case_class_counts[class_id]["ious"].append(float(iou))
                elif (not has_gt) and has_pred:
                    match_status = "FP"
                    case_class_counts[class_id]["FP"] += 1
                    false_positive_rows.append({
                        "dataset": dataset_name,
                        "case_id": case_id,
                        "frame_idx": int(frame_idx),
                        "frame_name": image_path.name,
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "conf": pred_conf,
                        "pred_bbox": bbox_to_str(pred_box),
                        "reason": "prediction_without_gt",
                    })
                elif has_gt and (not has_pred):
                    match_status = "FN"
                    case_class_counts[class_id]["FN"] += 1
                    false_negative_rows.append({
                        "dataset": dataset_name,
                        "case_id": case_id,
                        "frame_idx": int(frame_idx),
                        "frame_name": image_path.name,
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "gt_bbox": bbox_to_str(gt_box),
                        "reason": "gt_without_prediction",
                    })
                elif has_gt and has_pred and iou < TP_IOU_THRES:
                    match_status = "FP_FN_LOW_IOU"
                    case_class_counts[class_id]["LOW_IOU"] += 1
                    case_class_counts[class_id]["ious"].append(float(iou))
                    false_positive_rows.append({
                        "dataset": dataset_name,
                        "case_id": case_id,
                        "frame_idx": int(frame_idx),
                        "frame_name": image_path.name,
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "conf": pred_conf,
                        "pred_bbox": bbox_to_str(pred_box),
                        "gt_bbox": bbox_to_str(gt_box),
                        "iou": float(iou),
                        "reason": "low_iou_prediction",
                    })
                    false_negative_rows.append({
                        "dataset": dataset_name,
                        "case_id": case_id,
                        "frame_idx": int(frame_idx),
                        "frame_name": image_path.name,
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "gt_bbox": bbox_to_str(gt_box),
                        "pred_bbox": bbox_to_str(pred_box),
                        "iou": float(iou),
                        "reason": "low_iou_missed_gt",
                    })
                else:
                    match_status = "TN"
                    case_class_counts[class_id]["TN"] += 1

                frame_rows.append({
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "frame_idx": int(frame_idx),
                    "frame_name": image_path.name,
                    "frame_uid": frame_uid,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "label_gray_value": gray_value,
                    "has_gt": bool(has_gt),
                    "has_pred": bool(has_pred),
                    "match_status": match_status,
                    "iou": float(iou) if not np.isnan(iou) else np.nan,
                    "pred_conf": pred_conf,
                    "gt_bbox": bbox_to_str(gt_box),
                    "pred_bbox": bbox_to_str(pred_box),
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                })

            if SAVE_DEBUG_IMAGES:
                save_debug_detection_image(image, gt_by_class, preds_best, id_to_name, debug_img_root / case_id / f"{image_path.stem}.jpg")

        for class_id, class_name in id_to_name.items():
            counts = case_class_counts[class_id]
            tp = counts["TP"]
            fp = counts["FP"] + counts["LOW_IOU"]
            fn = counts["FN"] + counts["LOW_IOU"]
            tn = counts["TN"]
            precision, recall, f1 = compute_prf_from_counts(tp, fp, fn)
            mean_iou = float(np.mean(counts["ious"])) if len(counts["ious"]) > 0 else np.nan
            case_rows.append({
                "dataset": dataset_name,
                "case_id": case_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "label_gray_value": int(class_id_to_gray[class_id]),
                "num_frames": int(n),
                "num_gt_frames": int(counts["gt"]),
                "num_pred_frames": int(counts["pred"]),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "low_iou_frames": int(counts["LOW_IOU"]),
                "precision_at_0.50": precision,
                "recall_at_0.50": recall,
                "f1_at_0.50": f1,
                "mean_iou": mean_iou,
            })

    frame_df = pd.DataFrame(frame_rows)
    case_df = pd.DataFrame(case_rows)
    false_positive_df = pd.DataFrame(false_positive_rows)
    false_negative_df = pd.DataFrame(false_negative_rows)

    if len(frame_df) > 0:
        class_summary = compute_class_summary(frame_df, pred_records_all, gt_records_all, id_to_name, class_id_to_gray)
        class_summary.insert(0, "dataset", dataset_name)
    else:
        class_summary = pd.DataFrame()

    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        frame_df.to_excel(writer, sheet_name="frame_gt_pred_matches", index=False)
        case_df.to_excel(writer, sheet_name="case_class_summary", index=False)
        class_summary.to_excel(writer, sheet_name="class_summary", index=False)
        false_positive_df.to_excel(writer, sheet_name="false_positives", index=False)
        false_negative_df.to_excel(writer, sheet_name="false_negatives", index=False)

    frame_df.to_csv(dataset_run_root / "frame_gt_pred_matches.csv", index=False)
    case_df.to_csv(dataset_run_root / "case_class_summary.csv", index=False)
    class_summary.to_csv(dataset_run_root / "class_summary.csv", index=False)
    false_positive_df.to_csv(dataset_run_root / "false_positives.csv", index=False)
    false_negative_df.to_csv(dataset_run_root / "false_negatives.csv", index=False)

    print(f"\n[{dataset_name}] Done.")
    print(f"Saved Excel: {excel_output}")

    if len(class_summary) > 0:
        display_cols = ["target_class", "num_gt_boxes", "num_pred_boxes", "precision_at_0.50", "recall_at_0.50", "f1_at_0.50", "AP50", "AP75", "AP50_95"]
        existing = [c for c in display_cols if c in class_summary.columns]
        print("\nDataset class summary:")
        print(class_summary[existing].to_string(index=False))

    return frame_df, case_df, class_summary, false_positive_df, false_negative_df


def save_debug_detection_image(image, gt_by_class, preds_best, id_to_name, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis = image.copy()
    for class_id, gt_box in gt_by_class.items():
        x1, y1, x2, y2 = [int(v) for v in gt_box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"GT {id_to_name[class_id]}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    for class_id, pred in preds_best.items():
        box = pred["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"Pred {id_to_name.get(class_id, class_id)} {pred['conf']:.2f}", (x1, min(vis.shape[0] - 1, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), vis)


# ============================================================
# Main
# ============================================================

def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("YOLO26-det Test Evaluation on EgoMed5 Full Test Frames")
    print("=" * 120)
    print(f"Datasets:          {DATASETS}")
    print(f"Device:            {DEVICE}")
    print(f"YOLO run root:     {YOLO_RUN_ROOT}")
    print(f"DET_CONF_THRES:    {DET_CONF_THRES}")
    print(f"YOLO_IMGSZ:        {YOLO_IMGSZ}")
    print(f"TP_IOU_THRES:      {TP_IOU_THRES}")
    print(f"AP thresholds:     {AP_IOU_THRES_LIST}")
    print(f"Run root:          {RUN_ROOT}")
    print("=" * 120)

    all_frame_dfs = []
    all_case_dfs = []
    all_class_summaries = []
    all_fp_dfs = []
    all_fn_dfs = []

    for dataset_name in DATASETS:
        frame_df, case_df, class_summary, fp_df, fn_df = evaluate_dataset(dataset_name)
        if len(frame_df) > 0:
            all_frame_dfs.append(frame_df)
        if len(case_df) > 0:
            all_case_dfs.append(case_df)
        if len(class_summary) > 0:
            all_class_summaries.append(class_summary)
        if len(fp_df) > 0:
            all_fp_dfs.append(fp_df)
        if len(fn_df) > 0:
            all_fn_dfs.append(fn_df)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_frame_df = pd.concat(all_frame_dfs, ignore_index=True) if all_frame_dfs else pd.DataFrame()
    all_case_df = pd.concat(all_case_dfs, ignore_index=True) if all_case_dfs else pd.DataFrame()
    all_class_summary = pd.concat(all_class_summaries, ignore_index=True) if all_class_summaries else pd.DataFrame()
    all_fp_df = pd.concat(all_fp_dfs, ignore_index=True) if all_fp_dfs else pd.DataFrame()
    all_fn_df = pd.concat(all_fn_dfs, ignore_index=True) if all_fn_dfs else pd.DataFrame()

    combined_excel = RUN_ROOT / "egomed5_all_datasets_yolo26_det_test_metrics.xlsx"
    with pd.ExcelWriter(combined_excel, engine="openpyxl") as writer:
        all_class_summary.to_excel(writer, sheet_name="all_class_summary", index=False)
        all_case_df.to_excel(writer, sheet_name="all_case_summary", index=False)
        all_frame_df.to_excel(writer, sheet_name="all_frame_matches", index=False)
        all_fp_df.to_excel(writer, sheet_name="all_false_positives", index=False)
        all_fn_df.to_excel(writer, sheet_name="all_false_negatives", index=False)

    all_class_summary.to_csv(RUN_ROOT / "egomed5_all_datasets_class_summary.csv", index=False)
    all_case_df.to_csv(RUN_ROOT / "egomed5_all_datasets_case_summary.csv", index=False)
    all_frame_df.to_csv(RUN_ROOT / "egomed5_all_datasets_frame_matches.csv", index=False)
    all_fp_df.to_csv(RUN_ROOT / "egomed5_all_datasets_false_positives.csv", index=False)
    all_fn_df.to_csv(RUN_ROOT / "egomed5_all_datasets_false_negatives.csv", index=False)

    print("\n" + "=" * 120)
    print("All YOLO-det evaluations done.")
    print(f"Combined Excel: {combined_excel}")
    print(f"All outputs saved under: {RUN_ROOT}")
    print("=" * 120)

    if len(all_class_summary) > 0:
        display_cols = ["dataset", "target_class", "num_gt_boxes", "num_pred_boxes", "precision_at_0.50", "recall_at_0.50", "f1_at_0.50", "AP50", "AP75", "AP50_95"]
        existing = [c for c in display_cols if c in all_class_summary.columns]
        print("\nAll dataset class summary:")
        print(all_class_summary[existing].to_string(index=False))


if __name__ == "__main__":
    main()
