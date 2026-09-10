#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoMed-Agent Interactive Segmentation Demo
==========================================
Pick a video (an ACDC frame folder or an mp4), type the structure you want to
segment (LV cavity / RV cavity / myocardium), and get a segmentation video.

The core loop (YOLO detection -> SAM2 propagation -> consistency re-track) is
REUSED VERBATIM from the evaluation engine `scripts/egomed_agent/egomed-agent-iou06.py`
(loaded as a module, no code changes), so the demo behaves exactly like the
validated Table-II reproduction pipeline -- but without needing GT labels.

Designed as a team-extensible base: add one entry to MODALITIES (detector weight
+ class -> gray mapping) to support another imaging modality. Everything else is
modality-agnostic.

Run:
    python scripts/demo/egomed_demo.py            # start the Gradio web UI
    python scripts/demo/egomed_demo.py --cli      # one-shot CLI mode
    python scripts/demo/egomed_demo.py --cli --source <path> --target "LV cavity"
"""

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate repo root and load the evaluation engine as a module.
# The engine's top-level code sets CUDA_VISIBLE_DEVICES="0" before importing
# torch and defines DEVICE/TRACK_IOU_THRES etc., so importing it gives us a
# fully configured, working pipeline.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../scripts/demo -> repo root
ENGINE_PATH = REPO_ROOT / "scripts" / "egomed_agent" / "egomed-agent-iou06.py"
_spec = importlib.util.spec_from_file_location("egomed_engine", ENGINE_PATH)
eng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eng)

import cv2
import numpy as np

from nl_parser import parse_target
from asr_engine import speech_to_target

# ---------------------------------------------------------------------------
# Modality configuration.  Add one dict entry here to support another modality.
#   yolo_weight : detector checkpoint (per-modality, trained by the group)
#   name_to_gray: YOLO class name -> label gray value used when saving masks
# ---------------------------------------------------------------------------
MODALITIES = {
    "ACDC (MRI 心脏)": {
        "yolo_weight": REPO_ROOT / "runs" / "yolo26_det" / "ACDC_yolo26m_imgsz1024" / "weights" / "best.pt",
        "name_to_gray": {"RV cavity": 1, "myocardium": 2, "LV cavity": 3},
    },
    "CAMUS (超声 心脏)": {
        "yolo_weight": REPO_ROOT / "runs" / "yolo26_det" / "CAMUS_yolo26m_imgsz1024" / "weights" / "best.pt",
        "name_to_gray": {
            "Left Ventricular Myocardium": 1,
            "Left Ventricle": 2,
            "Left Atrium": 3,
        },
    },
    "Amos (CT 腹部)": {
        "yolo_weight": REPO_ROOT / "runs" / "yolo26_det" / "Amos_yolo26m_imgsz1024" / "weights" / "best.pt",
        "name_to_gray": {
            "spleen": 1,
            "right kidney": 2,
            "left kidney": 3,
            "gall bladder": 4,
            "esophagus": 5,
            "liver": 6,
            "stomach": 7,
            "arota": 8,
            "postcava": 9,
            "pancreas": 10,
            "right adrenal gland": 11,
            "left adrenal gland": 12,
            "duodenum": 13,
            "bladder": 14,
            "prostate/uterus": 15,
        },
    },
    "Montgomery (X光 胸片)": {
        "yolo_weight": REPO_ROOT / "runs" / "yolo26_det" / "Montgomery-County-CXR-Set_yolo26m_imgsz1024" / "weights" / "best.pt",
        "name_to_gray": {"left lung": 1, "right lung": 2},
    },
    "PolypGen (内窥镜)": {
        "yolo_weight": REPO_ROOT / "runs" / "yolo26_det" / "PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024" / "weights" / "best.pt",
        "name_to_gray": {"polyp": 1},
    },
}

# Models are expensive to load; cache them across calls.
_MODEL_CACHE = {}


def _get_models(modality):
    """Return (sam2_predictor, yolo_model), loading each once."""
    cfg = MODALITIES[modality]
    yolo_weight = cfg["yolo_weight"]
    if not yolo_weight.exists():
        raise FileNotFoundError(f"YOLO weight not found: {yolo_weight}")
    if "sam2" not in _MODEL_CACHE:
        _MODEL_CACHE["sam2"] = eng.init_sam2_predictor()
    key = f"{modality}|{yolo_weight}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = eng.YOLO(str(yolo_weight))
    return _MODEL_CACHE["sam2"], _MODEL_CACHE[key]


def resolve_frames(source_path, work_dir, max_frames=None):
    """Turn a frame folder OR a video file into (sorted image list, frame dir)."""
    src = Path(source_path)
    if src.is_dir():
        files = eng.collect_image_files(src)
        frame_dir = src
    elif src.is_file():
        frame_dir = work_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(src))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(frame_dir / f"{n:06d}.jpg"), frame)
            n += 1
            if max_frames and n >= max_frames:
                break
        cap.release()
        files = eng.collect_image_files(frame_dir)
        if n == 0:
            raise ValueError(f"Cannot read video frames from: {src}")
    else:
        raise FileNotFoundError(f"Path does not exist: {src}")

    if max_frames:
        files = files[:max_frames]
    if len(files) == 0:
        raise ValueError(f"No image frames found in: {src}")
    return files, frame_dir


def _filter_detections(detections, target_id, frame_h, frame_w,
                       min_area_ratio=0.001, min_occurrence=2):
    """过滤检测误检，改善翻拍/低质量视频的分割效果（对所有模态通用）。

    1) 面积过滤：丢弃面积 < frame_h*frame_w*min_area_ratio 的检测框。
       （手机拍屏幕时，鼠标光标 / UI 元素 / 工具栏的框通常极小）
    2) 连续性过滤：只保留连续出现 >= min_occurrence 帧的目标。
       （真实器官在视频里会连续可见；零散 1~2 帧的"检测"大概率是误检）

    两个规则都用**相对帧面积 / 连续帧数**来衡量，不依赖具体器官尺寸，
    因此对小器官（息肉、肾上腺等）和正常干净影像同样安全。
    """
    n = len(detections)
    if n == 0:
        return detections, 0
    min_area = float(frame_h) * frame_w * min_area_ratio
    valid = [False] * n
    for i, d in enumerate(detections):
        info = d.get(target_id)
        if info is None:
            continue
        x1, y1, x2, y2 = info["box"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        valid[i] = area >= min_area

    kept = valid[:]
    if min_occurrence > 1:
        i = 0
        while i < n:
            if valid[i]:
                j = i
                while j < n and valid[j]:
                    j += 1
                if (j - i) < min_occurrence:
                    for k in range(i, j):
                        kept[k] = False
                i = j
            else:
                i += 1

    removed = sum(1 for a, b in zip(valid, kept) if a and not b)
    filtered = [dict(d) if kept[i] else {} for i, d in enumerate(detections)]
    return filtered, removed


def _segment_target(predictor, yolo_model, image_files, frame_dir, target_id,
                    id_to_name, class_id_to_gray, output_dir, target_text,
                    conf_thres=None, min_area_ratio=0.001, min_occurrence=2,
                    filter_enabled=True, drift_area_ratio=3.0,
                    drift_center_ratio=0.15, fill_bgr=None,
                    max_prop_frames=None):
    """Single-target online segmentation: detect -> propagate -> re-track.

    Mirrors `evaluate_case` from the engine but tracks only ONE class and needs
    no GT labels / prompt schedule (the chosen target is active from frame 0).

    fill_bgr: 分割区域叠加颜色 (BGR)。None 表示用引擎默认红色。
              内窥镜息肉用绿色 (0,255,0) 对比更明显。
    max_prop_frames: 单段 SAM2 传播的最大帧数上限。屏幕翻拍等低质量视频里
              SAM2 长程传播会持续漂移（几帧就偏出目标），设一个短上限
              （如 10 帧）强制周期性地用 YOLO 检测框重新定位，可显著
              减少"标注跳到错误位置"的问题。None 表示不限制。
    """
    overlay_dir = output_dir / "overlay"
    mask_dir = output_dir / "masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    num_frames = len(image_files)
    first = cv2.imread(str(image_files[0]), cv2.IMREAD_COLOR)
    frame_h, frame_w = first.shape[:2]

    # 1. Precompute YOLO detections for every frame (only the target class).
    #    Patch the engine's confidence threshold so it applies to every modality.
    old_conf = eng.DET_CONF_THRES
    if conf_thres is not None and conf_thres > 0:
        eng.DET_CONF_THRES = float(conf_thres)
    try:
        detections = eng.precompute_yolo_detections(
            yolo_model, image_files, focus_class_ids={target_id}
        )
    finally:
        eng.DET_CONF_THRES = old_conf

    # 1b. False-positive filtering (small boxes / isolated frames).
    # Keep the RAW detections for drift-correction checks: filtering must only
    # affect where we INITIALIZE, not whether we can CORRECT (fewer filtered
    # frames = fewer chances for YOLO to pull the mask back onto the organ).
    detections_raw = detections
    removed_fp = 0
    if filter_enabled and (min_area_ratio > 0 or min_occurrence > 1):
        detections, removed_fp = _filter_detections(
            detections, target_id, frame_h, frame_w,
            min_area_ratio, min_occurrence,
        )
        if removed_fp > 0:
            print(f"[demo] 误检过滤：剔除了 {removed_fp} 帧的疑似误检"
                  f"（面积<{min_area_ratio*100:.2f}% 帧面积，或孤立出现<{min_occurrence}帧）")

    # 2. Find the first frame where the target is detected.
    start = next((i for i, d in enumerate(detections) if target_id in d), None)
    if start is None:
        return {
            "status": f"未检测到目标 {target_text}（可能是视频模态不符，或目标未出现在画面中）",
            "masks": {}, "overlay_files": [], "events": [], "detection_frames": 0,
        }

    # 3. Load the video once into SAM2, then run segments.
    print(f"[demo] Loading SAM2 frames once ({num_frames} frames)...")
    inference_state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(inference_state)

    masks = {}        # frame_idx -> bool (h, w)
    track_boxes = {}  # frame_idx -> bbox
    events = []
    segment_id = 0
    current = start
    last_mask = None

    while current < num_frames:
        segment_id += 1

        # No detection at this frame: carry the last propagated mask forward
        # until the next frame that detects the target.
        if target_id not in detections[current]:
            nxt = current + 1
            while nxt < num_frames and target_id not in detections[nxt]:
                nxt += 1
            for i in range(current, min(nxt, num_frames)):
                if last_mask is not None:
                    masks[i] = last_mask
            current = nxt
            continue

        predictor.reset_state(inference_state)
        obj_id = 1
        eng.sam2_add_box(
            predictor, inference_state, current, obj_id,
            detections[current][target_id]["box"],
        )
        events.append(f"segment {segment_id}: 初始化 @帧 {current}")

        # 4. Propagate until a re-track is triggered.
        #    Drift correction: THREE independent triggers, so the YOLO
        #    detection agent keeps correcting the mask even when detections
        #    are sparse:
        #      (a) det-track IoU < TRACK_IOU_THRES  (classic, needs a det)
        #      (b) mask area jump > drift_area_ratio (needs no det)
        #      (c) mask center jump > drift_center_ratio (needs no det)
        #    When the current frame has no detection, fall back to the most
        #    recent high-confidence detection box.
        next_reinit = None
        last_valid_box = None
        prev_mask = None
        prev_center = None
        frames_in_seg = 0
        frame_diag = float((frame_w ** 2 + frame_h ** 2) ** 0.5)
        for out_idx, out_obj_ids, out_mask_logits in eng.propagate_video(
            predictor, inference_state, current
        ):
            out_idx = int(out_idx)
            if out_idx >= num_frames:
                break
            m = eng.sam2_masks_from_output(
                out_obj_ids, out_mask_logits, {obj_id: target_id}
            )
            mask = m.get(target_id)
            # "丢失" = 无 mask 输出，或 mask 全空（SAM2 跟丢时常返回
            # 全空 mask 而非 None，两种都要视为丢失，否则该帧之后的
            # 所有帧都会一路空转，导致只有前几帧有标注）。
            lost = (mask is None or int(mask.sum()) == 0)
            if lost:
                # 若 YOLO 在这一帧仍检测到目标，立即中断本段并用它的框
                # 重新初始化（否则从丢帧开始到视频结尾都不会再有 mask，
                # 出现"YOLO 框一直在、分割标注只有前几帧"的现象）。
                if out_idx > current and target_id in detections[out_idx]:
                    next_reinit = out_idx
                    events.append(
                        f"segment {segment_id}: 丢失重捕 @帧 {out_idx}"
                    )
                    print(f"[demo] 丢失重捕 @帧 {out_idx}")
                    break
                continue

            masks[out_idx] = mask
            last_mask = mask
            track_boxes[out_idx] = eng.mask_to_bbox(mask)

            det = detections_raw[out_idx].get(target_id)
            if det is not None:
                last_valid_box = det["box"]

            if out_idx <= current:
                prev_mask = mask
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    prev_center = (float(xs.mean()), float(ys.mean()))
                continue

            # ---- 段长限制：SAM2 在低质量（屏幕翻拍）视频上长程传播会
            # 缓慢漂移出目标。限制单段传播长度，到期就跳到下一个有 YOLO
            # 检测的帧重新定位，把漂移切断在可接受范围内。----
            frames_in_seg += 1
            if (max_prop_frames and frames_in_seg >= max_prop_frames
                    and out_idx + 1 < num_frames):
                nxt = out_idx + 1
                while nxt < num_frames and target_id not in detections[nxt]:
                    nxt += 1
                if nxt < num_frames:
                    next_reinit = nxt
                    events.append(
                        f"segment {segment_id}: 周期重捕 @帧 {nxt} "
                        f"(本段已传{frames_in_seg}帧)"
                    )
                    print(f"[demo] 周期重捕 @帧 {nxt} (本段已传{frames_in_seg}帧)")
                break

            tb = track_boxes[out_idx]
            reason = None
            if det is not None and tb is not None:
                iou = eng.bbox_iou(det["box"], tb)
                if iou < eng.TRACK_IOU_THRES:
                    reason = f"det-track IoU={iou:.2f}<{eng.TRACK_IOU_THRES:.1f}"
            if reason is None and prev_mask is not None:
                a_now = int(mask.sum())
                a_prev = int(prev_mask.sum())
                if a_prev > 50 and a_now > 0:
                    r = a_now / a_prev
                    if r > drift_area_ratio or r < 1.0 / drift_area_ratio:
                        reason = f"面积突变 x{r:.1f}"
            if reason is None and prev_center is not None:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    c_now = (float(xs.mean()), float(ys.mean()))
                    d = ((c_now[0] - prev_center[0]) ** 2 +
                         (c_now[1] - prev_center[1]) ** 2) ** 0.5
                    if d > drift_center_ratio * frame_diag:
                        reason = f"中心跳变 {d:.0f}px"

            if reason is not None:
                box = det["box"] if det is not None else last_valid_box
                if box is not None:
                    next_reinit = out_idx
                    events.append(
                        f"segment {segment_id}: YOLO修正 @帧 {out_idx} ({reason})"
                    )
                    print(f"[demo] YOLO修正 @帧 {out_idx}: {reason}")
                    break

            prev_mask = mask
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                prev_center = (float(xs.mean()), float(ys.mean()))

        current = next_reinit if next_reinit is not None else num_frames

    # 5. Render per-frame prediction + overlay.
    #    fill_bgr=None 表示"用引擎默认红色"：只有息肉用绿色，其余模态
    #    （X光/超声/CT/MRI）保持引擎原来的行为，不能把 None 传下去。
    if fill_bgr is None:
        fill_bgr = (0, 0, 255)
    overlay_files = []
    for i, image_path in enumerate(image_files):
        pred_label = np.zeros((frame_h, frame_w), dtype=np.uint8)
        mask = masks.get(i)
        if mask is not None:
            if mask.shape != (frame_h, frame_w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            pred_label[mask] = int(class_id_to_gray[target_id])

        eng.save_pred_label(pred_label, mask_dir / f"{Path(image_path).stem}.png")

        of = overlay_dir / f"{i:06d}.jpg"
        eng.overlay_prediction(
            image_rgb=eng.read_rgb_image(image_path),
            pred_label=pred_label,
            gt_label=None,
            class_id_to_gray=class_id_to_gray,
            id_to_name=id_to_name,
            active_dets=detections[i] if target_id in detections[i] else {},
            track_boxes=({target_id: track_boxes[i]}
                         if track_boxes.get(i) is not None else {}),
            output_path=of,
            fill_bgr=fill_bgr,
        )
        overlay_files.append(of)

    try:
        del inference_state
    except Exception:
        pass
    if eng.torch.cuda.is_available():
        eng.torch.cuda.empty_cache()

    return {
        "status": "完成",
        "masks": masks,
        "overlay_files": overlay_files,
        "events": events,
        "detection_frames": sum(1 for d in detections if target_id in d),
        "retrack_count": sum(1 for e in events
                              if "YOLO修正" in e or "re-track" in e),
        "removed_fp": removed_fp,
    }


def _encode_video(overlay_files, out_path, fps, max_width):
    """Stitch overlay frames into an mp4 (H.264 so browsers can play it)."""
    if not overlay_files:
        return
    first = cv2.imread(str(overlay_files[0]))
    h, w = first.shape[:2]
    if max_width and w > max_width:
        size = (max_width, int(h * max_width / w))
    else:
        size = (w, h)
    # 1. Write a temporary mp4v file (fast, reliable everywhere).
    tmp_path = out_path.with_name(out_path.stem + "_tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    for f in overlay_files:
        img = cv2.imread(str(f))
        if img.shape[1] != size[0]:
            img = cv2.resize(img, size)
        writer.write(img)
    writer.release()
    # 2. Transcode to H.264 (avc1) so the Gradio <video> can play it in a browser.
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(tmp_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-movflags", "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def _resolve_modality_key(modality):
    """把用户传入的模态标识解析为 MODALITIES 的正式键名。

    支持：完整键名（如 'CAMUS (超声 心脏)'）、英文关键词
    （如 'CAMUS'、'Amos'、'ACDC'、'Montgomery'、'PolypGen'，
    不区分大小写）。找不到返回 None。
    """
    if not modality:
        return None
    if modality in MODALITIES:
        return modality
    low = modality.strip().lower()
    for m in MODALITIES:
        if low == m.lower() or m.lower().startswith(low) or low in m.lower():
            return m
    return None


def resolve_modality(target_text, modality=None):
    """根据目标名（或显式指定的模态）确定要用的模态配置。

    规则：若显式给了 modality 且目标在其中 -> 用它；
    否则在全部模态里找能识别该目标的模态（支持跨模态自动匹配）。
    返回 (modality_name, name_to_gray) 或 (None, None)。
    """
    if modality:
        key = _resolve_modality_key(modality)
        if key is None:
            # 给了未知模态名：提示可选值，不要静默走跨模态匹配
            raise ValueError(
                f"未知模态「{modality}」。可选：{list(MODALITIES)}"
            )
        cfg = MODALITIES[key]
        if target_text in cfg["name_to_gray"]:
            return key, cfg["name_to_gray"]
    # 跨模态自动匹配：目标属于哪个模态就用哪个
    for m, cfg in MODALITIES.items():
        if target_text in cfg["name_to_gray"]:
            return m, cfg["name_to_gray"]
    return None, None


def segment_video(target_text, source_path, modality=None,
                  work_dir=None, fps=10, max_width=1280, max_frames=None,
                  conf_thres=None, min_area_ratio=0.001, min_occurrence=2,
                  filter_enabled=True):
    """Full pipeline: resolve frames -> detect -> propagate -> overlay -> mp4.

    conf_thres       : 覆盖 YOLO 检测置信度阈值（默认 None=用引擎的 0.3）
    min_area_ratio   : 过滤面积 < 帧面积*该比例的检测框（默认 0.001=0.1%）
    min_occurrence   : 只保留连续出现 >= N 帧的目标（默认 2）
    filter_enabled   : 是否启用误检过滤（对正常干净影像也安全，可关闭）

    Returns (mp4_path or None, human-readable summary).
    """
    modality, name_to_gray = resolve_modality(target_text, modality)
    if modality is None:
        return None, (
            f"目标 '{target_text}' 不在任何已配置模态中。\n"
            f"当前支持的模态：{list(MODALITIES)}"
        )

    predictor, yolo_model = _get_models(modality)

    # YOLO names -> filtered class mapping consistent with the config.
    id_to_name = {
        k: v for k, v in eng.load_yolo_class_names_from_model(yolo_model).items()
        if v in name_to_gray
    }
    class_id_to_gray = {k: name_to_gray[v] for k, v in id_to_name.items()}
    target_id = next((k for k, v in id_to_name.items() if v == target_text), None)
    if target_id is None:
        return None, f"YOLO 权重中找不到目标 '{target_text}'（类别: {list(id_to_name.values())}）"

    base = work_dir or (REPO_ROOT / "runs" / "demo_output" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)

    try:
        image_files, frame_dir = resolve_frames(source_path, base, max_frames)
    except Exception as e:
        return None, f"读取视频失败: {e}"

    num_frames = len(image_files)
    print(f"[demo] modality={modality} target={target_text} frames={num_frames}")

    # 内窥镜息肉：用绿色叠加，避免红色息肉在红色覆盖下看不清。
    # （其他模态保持引擎默认的红色预测叠加）
    is_polyp = "PolypGen" in modality
    fill_bgr = (0, 255, 0) if is_polyp else None
    result = _segment_target(
        predictor, yolo_model, image_files, frame_dir, target_id,
        id_to_name, class_id_to_gray, base, target_text,
        conf_thres=conf_thres, min_area_ratio=min_area_ratio,
        min_occurrence=min_occurrence, filter_enabled=filter_enabled,
        # 内窥镜多是屏幕翻拍视频，SAM2 长程传播容易漂移：限长 10 帧
        # 周期性用 YOLO 框重新定位，并把面积突变阈值收紧到 2 倍。
        drift_area_ratio=2.0 if is_polyp else 3.0,
        max_prop_frames=10 if is_polyp else None,
        fill_bgr=fill_bgr,
    )

    if result["status"].startswith("未检测"):
        return None, result["status"]

    mp4_path = base / "result.mp4"
    _encode_video(result["overlay_files"], mp4_path, fps, max_width)

    n_det = result["detection_frames"]
    n_retrack = result["retrack_count"]
    n_mask = len(result["masks"])
    removed = result.get("removed_fp", 0)
    summary = (
        f"[完成] 目标: {target_text} | 总帧数: {num_frames} | "
        f"检测到目标的帧: {n_det} | 有分割结果的帧: {n_mask} | "
        f"YOLO修正/重追踪次数: {n_retrack}\n"
        f"误检过滤: 剔除 {removed} 帧（面积过滤+孤立帧过滤，可关闭）\n"
        f"输出: {base}\n"
        f"分割动画: {mp4_path}\n"
        f"分割 mask: {base / 'masks'}\n"
        f"效果图: {base / 'overlay'}"
    )
    if result["events"]:
        summary += "\n运行过程:\n  " + "\n  ".join(result["events"][:20])
    return str(mp4_path), summary


# ---------------------------------------------------------------------------
# Gradio web UI
# ---------------------------------------------------------------------------
def run_ui():
    import gradio as gr

    def _handle(source_path, upload, modality_sel, target_text, fps,
                conf_val, use_filter):
        src = upload if upload else source_path
        if not src:
            return None, "请先填写视频/帧文件夹路径，或上传一个视频文件。"
        # 自然语言 -> 分割目标（可能返回多个并列候选，需结合模态消歧）
        parsed = parse_target(target_text)
        if not parsed.ok:
            return None, parsed.message
        # 用所选模态过滤候选：优先保留该模态支持的目标
        modality = modality_sel if modality_sel else None
        chosen = None
        if modality and len(parsed.targets) > 1:
            supported = set(MODALITIES[modality]["name_to_gray"].keys())
            filtered = [t for t in parsed.targets if t in supported]
            if len(filtered) == 1:
                chosen = filtered[0]
            elif len(filtered) > 1:
                return None, (
                    f"模态「{modality}」下这些目标仍存在歧义：{filtered}。\n"
                    f"请说得更具体些，例如明确输入完整目标名。"
                )
        if chosen is None:
            if len(parsed.targets) > 1:
                return None, (
                    f"识别到多个目标：{parsed.targets}。\n"
                    f"请在上方选择对应的模态，或说得更具体（如把模态区分开）。"
                )
            chosen = parsed.targets[0]
        target = chosen
        print(f"[demo] 模态={modality} 自然语言「{target_text}」-> 目标 {target}")
        try:
            return segment_video(
                target, src, modality=modality, fps=fps,
                conf_thres=float(conf_val) if conf_val else None,
                filter_enabled=bool(use_filter),
            )
        except Exception as e:
            import traceback
            return None, f"出错:\n{traceback.format_exc()}"

    with gr.Blocks(title="EgoMed 交互式分割 Demo") as demo:
        gr.Markdown(
            "## 🫀 EgoMed-Agent 交互式医学影像分割 Demo\n"
            "选择或上传一段医学影像（帧文件夹或 mp4），用**自然语言**告诉系统要分割什么，"
            "模型输出逐帧分割动画（红色 = 预测 mask，蓝色 = YOLO 检测框，黄色 = SAM2 追踪框）。"
            "\n\n💬 **支持自然语言指令**，例如：\"帮我分割左心室\"、\"把肝标出来\"、\"分割右肺\"。"
            "\n\n🫀🫁🩻🦠 **支持 5 种模态**："
            "MRI 心脏（左/右心室腔、心肌）、超声心脏（左室/左房）、CT 腹部（肝、脾、肾、胰…）、"
            "X光胸片（左/右肺）、内窥镜（息肉）。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                modality = gr.Dropdown(
                    choices=list(MODALITIES.keys()),
                    value=None, label="影像模态（可选，辅助消歧）",
                    info="不选也能自动匹配；但当同一器官存在于多种模态时，建议选择",
                )
                source = gr.Textbox(
                    label="视频 / 帧文件夹路径",
                    placeholder=r"例如 D:\EgoMed-Agent\data\ACDC\img\14 （或直接粘一个 mp4 路径）",
                )
                upload = gr.File(label="或上传视频文件 (mp4/avi)")
                target = gr.Textbox(
                    label="自然语言指令 / 目标",
                    placeholder='例如："帮我分割左心室"、"把肝标出来"、"分割右肺"',
                )
                with gr.Accordion("🎤 语音输入（实验功能）", open=False):
                    gr.Markdown("点击下方按钮录音（授权麦克风），说完点**停止**，再点**识别语音**。")
                    voice_audio = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="录音：说目标器官，如“分割肝脏”",
                    )
                    voice_btn = gr.Button("🎙 识别语音 -> 填入目标", variant="secondary")
                    voice_status = gr.Markdown("")
                fps = gr.Slider(5, 30, value=10, step=1, label="导出帧率 (fps)")
                conf_slider = gr.Slider(
                    0.1, 0.9, value=0.3, step=0.05,
                    label="检测灵敏度 (conf)",
                    info="越低越敏感（误检可能增加），越高越严格（可能漏检）。翻拍屏幕的视频建议 0.4~0.5",
                )
                use_filter = gr.Checkbox(
                    value=True, label="误检过滤（推荐）",
                    info="剔除面积过小 / 孤立出现的检测框，对翻拍或低质量视频提升明显，正常影像也安全",
                )
                btn = gr.Button("开始分割", variant="primary")
            with gr.Column(scale=1):
                out_video = gr.Video(label="分割动画")
                out_status = gr.Textbox(label="运行日志", lines=10)

        btn.click(
            _handle,
            inputs=[source, upload, modality, target, fps, conf_slider, use_filter],
            outputs=[out_video, out_status],
        )

        # ---- 语音识别：录音 -> 转文字 -> 填入目标框（复用 NLU）----
        def _on_voice(audio):
            if audio is None:
                return "", "⚠️ 请先录音（点麦克风按钮，说目标器官）"
            text, r = speech_to_target(audio)
            if not text:
                return "", "⚠️ 没识别到语音，请靠近麦克风重试"
            if r and r.ok:
                msg = f"✅ 识别为：**{text}** → 目标 **{r.message}**\n请点**开始分割**。"
                return text, msg
            hint = r.message if r and r.message else "未能解析目标"
            return text, f"🗣 识别为：**{text}**\n{hint}"

        voice_btn.click(
            _on_voice,
            inputs=[voice_audio],
            outputs=[target, voice_status],
        )

    # 系统开启了代理时，gradio 的健康检查(httpx)会走代理导致 localhost 请求 500。
    # 这里显式禁用对本地回环的代理。
    import os as _os
    for _k in ("NO_PROXY", "no_proxy"):
        _os.environ[_k] = "127.0.0.1,localhost"
    _os.environ.pop("HTTP_PROXY", None)
    _os.environ.pop("HTTPS_PROXY", None)
    _os.environ.pop("http_proxy", None)
    _os.environ.pop("https_proxy", None)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )


# ---------------------------------------------------------------------------
# CLI mode (also used for automated testing)
# ---------------------------------------------------------------------------
def run_cli(args):
    if not args.source or not args.target:
        print("CLI 模式需要 --source 和 --target。例如：")
        print("  python scripts/demo/egomed_demo.py --cli --source <路径> --target \"帮我分割左心室\"")
        return 1
    # 自然语言 -> 分割目标（CLI 也支持）
    parsed = parse_target(args.target)
    if not parsed.ok:
        print(parsed.message)
        return 1

    # 可选显式指定模态，用于跨模态歧义（如左心室同时存在于 ACDC / CAMUS）
    modality = args.modality if args.modality else None
    if modality:
        key = _resolve_modality_key(modality)
        if key is None:
            print(f"未知模态「{modality}」。可选：{list(MODALITIES)}")
            return 1
        # 用模态过滤多候选目标，消歧
        if len(parsed.targets) > 1:
            supported = set(MODALITIES[key]["name_to_gray"].keys())
            filtered = [t for t in parsed.targets if t in supported]
            if len(filtered) == 1:
                parsed.targets = filtered
            elif len(filtered) > 1:
                print(f"模态「{key}」下这些目标仍有歧义：{filtered}。请说得更具体些。")
                return 1
        target = parsed.targets[0]
        if target not in MODALITIES[key]["name_to_gray"]:
            print(f"模态「{key}」不支持目标「{target}」。支持：{list(MODALITIES[key]['name_to_gray'])}")
            return 1
        print(f"[demo] 显式模态「{key}」-> 目标 {target}")
    else:
        if len(parsed.targets) > 1:
            print(parsed.message)
            return 1
        target = parsed.targets[0]
        print(f"[demo] 自然语言「{args.target}」-> 目标 {target}")
    mp4, summary = segment_video(
        target, args.source, modality=modality,
        fps=args.fps, max_frames=args.max_frames,
        conf_thres=args.conf, filter_enabled=not args.no_filter,
    )
    print(summary)
    if mp4 is None:
        return 1
    print(f"\nMP4: {mp4}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EgoMed 交互式分割 Demo")
    parser.add_argument("--cli", action="store_true", help="命令行一次性模式（不启动界面）")
    parser.add_argument("--source", help="视频文件或帧文件夹路径")
    parser.add_argument("--target", help="要分割的结构（LV cavity / RV cavity / myocardium）")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--modality", default=None, help="显式指定影像模态（跨模态歧义时使用）")
    parser.add_argument("--max-frames", type=int, default=None, help="只处理前 N 帧（调试用）")
    parser.add_argument("--conf", type=float, default=None,
                        help="YOLO 检测置信度阈值（默认 0.3；翻拍屏幕的视频建议 0.4~0.5）")
    parser.add_argument("--no-filter", action="store_true",
                        help="关闭误检过滤（面积过滤 + 孤立帧过滤）")
    args = parser.parse_args()

    if args.cli:
        sys.exit(run_cli(args))
    run_ui()
