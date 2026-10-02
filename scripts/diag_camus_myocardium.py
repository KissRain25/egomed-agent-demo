"""更严谨的判据：心肌框与 LV 框给出的掩膜是不是"同一块整体结构"？

测量（同一批帧，同一 YOLO + 同一 SAM2 图像预测器）：
  D1 = Dice(心肌掩膜, GT 心肌)                 —— 单类重叠（易被面积巧合抬高）
  D2 = Dice(心肌掩膜, GT 心肌 ∪ GT LV 腔)       —— 若高，说明它给的是"整体截面"而非"环"
  D3 = Dice(LV 掩膜,   GT 心肌 ∪ GT LV 腔)
  D4 = Dice(心肌掩膜, LV 掩膜)                  —— 两类掩膜是否同一块
  面积比：心肌掩膜 / GT心肌；心肌掩膜 / (GT心肌∪GT LV)
"""
from __future__ import annotations

import csv
import pathlib
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = pathlib.Path(r"D:\EgoMed-Agent")
YOLO_W = ROOT / "runs/yolo26_det/CAMUS_yolo26m_imgsz1024/weights/best.pt"
SAM2_CKPT = ROOT / "checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CASES = ["4", "5"]
N_FRAMES = 6
DET_CONF, IMGSZ = 0.3, 1024


def _2d(x):
    return np.squeeze(x).astype(bool)


def dice(a, b) -> float:
    a, b = _2d(a), _2d(b)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    s = int(a.sum() + b.sum())
    return float(2 * int((a & b).sum()) / s) if s else 1.0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO(str(YOLO_W))
    names = det.names
    names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    myo_id = next(i for i, n in names.items() if n.lower().startswith("left ventricular myocardium"))
    lv_id = next(i for i, n in names.items() if n == "Left Ventricle")
    predictor = SAM2ImagePredictor(build_sam2(SAM2_CFG, str(SAM2_CKPT), device=device))

    print("%-6s %-9s %7s %7s %7s %7s %10s %10s" % (
        "病例", "帧", "D1", "D2", "D3", "D4", "心肌掩膜/GT心肌", "心肌掩膜/整体"))
    rows = []
    for case in CASES:
        for fp in sorted((ROOT / "data/CAMUS/img" / case).glob("*.jpg"))[:N_FRAMES]:
            gt = cv2.imread(str(ROOT / "data/CAMUS/label" / case / f"{fp.stem}.png"), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
            det_res = det.predict(source=img, conf=DET_CONF, imgsz=IMGSZ, verbose=False)[0]
            boxes = {int(b.cls): [] for b in det_res.boxes}
            for b in det_res.boxes:
                boxes[int(b.cls)].append([float(v) for v in b.xyxy[0].tolist()])
            if not boxes.get(myo_id) or not boxes.get(lv_id):
                continue
            predictor.set_image(img)
            masks = {}
            for cid, key in [(myo_id, "myo"), (lv_id, "lv")]:
                best, best_area = None, -1
                for box in boxes[cid]:
                    m, _, _ = predictor.predict(box=np.array(box), multimask_output=False)
                    m = _2d(m[0])
                    if m.sum() > best_area:
                        best, best_area = m, int(m.sum())
                masks[key] = best

            myo, lv = masks["myo"], masks["lv"]
            gt_myo, gt_lv = _2d(gt == 1), _2d(gt == 2)
            whole = gt_myo | gt_lv
            D1 = dice(myo, gt_myo)                       # 心肌掩膜单类重叠
            D2 = dice(myo, whole)                        # 心肌掩膜 vs 整体（是否"整块"）
            D3 = dice(lv, whole)
            D4 = dice(myo, lv)                           # 两类掩膜是否同一块
            D5 = dice(lv & ~myo, gt_lv)                  # 修复方案：LV 优先减心肌 → LV 得分
            D6 = dice(lv, gt_lv)                         # 线上方案（LV 后写=LV 掩膜原样）→ LV 得分
            r1 = myo.sum() / max(1, gt_myo.sum())
            r2 = myo.sum() / max(1, whole.sum())
            rows.append((case, fp.stem, D1, D2, D3, D4, D5, D6, r1, r2))
            print("%-6s %-9s %7.3f %7.3f %7.3f %7.3f %8.3f %8.3f %10.2f %10.2f" % (
                case, fp.stem, D1, D2, D3, D4, D5, D6, r1, r2))

    if rows:
        a = np.array([r[2:] for r in rows], dtype=float)
        print("-" * 96)
        print("均值（n=%d）：D1=%.3f  D2=%.3f  D3=%.3f  D4=%.3f ｜ 心肌掩膜/GT心肌=%.2f 倍  心肌掩膜/整体=%.2f 倍"
              % (len(rows), a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(),
                 a[:, 4].mean(), a[:, 5].mean()))
        out = ROOT / "runs" / "ab_myo_result2.csv"
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case", "frame", "D1_myo_vs_gtmyo", "D2_myo_vs_whole", "D3_lv_vs_whole",
                        "D4_myo_vs_lv", "ratio_myo_over_gtmyo", "ratio_myo_over_whole"])
            w.writerows(rows)
        print("明细:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
