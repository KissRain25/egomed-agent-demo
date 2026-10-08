#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 ACDC 是否也存在「标签写序」问题（LV 腔最后写 → 覆盖心肌）。

用途（2026-10-03 首次执行，结论 = 否证）
--------------------------------------
CAMUS 的心肌被"后写的 LV 腔"抹掉（预测 LV 掩膜面积达 GT 的 2.55 倍），
本脚本用**同一口径**检查 ACDC（它的类别顺序同样是 LV 腔最后写）：
实测 **预测心肌面积 = GT 的 93.2%、预测 LV 掩膜 = GT 的 0.99 倍** → 未被覆盖，
即 ACDC 的 myocardium 0.518 是真实难度（薄环分割 + 跟踪漂移），与写序无关。
=> 把「写序致命」的结论收紧为**条件性**：前提是预测掩膜严重外扩。

判据（与 CAMUS 同口径）
----------------------
  1) 预测心肌像素数 vs GT 心肌像素数（若预测心肌 ≈ 0 → 被覆盖）
  2) GT 心肌像素落在「预测 LV 腔」内的比例（被 LV 吃掉的量）
  3) 预测 LV 腔 / GT LV 腔 面积比（外扩倍数；CAMUS 2.55 倍命中、ACDC 0.99 倍不命中）

用法
----
    python scripts/diag_acdc_order.py      # 需要已跑过 ACDC 评测（pred_mask 存在）

归属：闫（诊断脚本）｜ 关联：docs/复现报告.md §六 表 A3「反向验证」、docs/工作日志.md 2026-10-03
"""
from __future__ import annotations

import csv
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN = (ROOT / "runs/eval_yolo26_original_sam2_video_schedule_reset_retrack_correction"
       / "egomed5_yolo26m_original_sam2_online_schedule_reset_retrack_iou06" / "ACDC")
# ACDC 灰度：1=RV cavity, 2=myocardium, 3=LV cavity
G_RV, G_MYO, G_LV = 1, 2, 3
SCHEDULE = ROOT / "data/text_prompt_eval/egomed5_test_prompt_schedule.csv"


def test_cases() -> list[str]:
    rows = list(csv.DictReader(SCHEDULE.open(encoding="utf-8")))
    out, seen = [], set()
    for r in rows:
        if r["dataset"] == "ACDC" and r["prompt_type"] == "exact":
            c = str(r["case_id"])
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cases = test_cases()[:6]
    print("=== ACDC 写序诊断（前 %d 个测试病例）===" % len(cases))
    print("%-6s %-6s %10s %10s %10s %10s" % (
        "病例", "帧", "GT心肌px", "预测心肌px", "GT心肌∩预测LV", "预测LVpx/GT LVpx"))
    tot = dict(gt_myo=0, pd_myo=0, gtmyo_in_pdlv=0, pd_lv=0, gt_lv=0, n=0)
    for case in cases:
        pm = RUN / "pred_mask" / case
        gt_dir = ROOT / "data/ACDC/label" / case
        if not pm.exists():
            print("%-6s 无预测目录（%s）" % (case, pm))
            continue
        files = sorted(pm.glob("*.png"))
        # 均匀抽样 6 帧
        picked = files[:: max(1, len(files) // 6)][:6]
        for p in picked:
            gt_p = gt_dir / (p.stem + ".png")
            if not gt_p.exists():
                continue
            pred = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE)
            if pred is None or gt is None or pred.shape != gt.shape:
                continue
            gt_myo, gt_lv = (gt == G_MYO), (gt == G_LV)
            pd_myo, pd_lv = (pred == G_MYO), (pred == G_LV)
            if gt_myo.sum() == 0:
                continue
            inter = int(np.logical_and(gt_myo, pd_lv).sum())
            print("%-6s %-6s %10d %10d %10d %10s" % (
                case, p.stem, int(gt_myo.sum()), int(pd_myo.sum()), inter,
                "%.2f" % (pd_lv.sum() / max(1, gt_lv.sum()))))
            tot["gt_myo"] += int(gt_myo.sum())
            tot["pd_myo"] += int(pd_myo.sum())
            tot["gtmyo_in_pdlv"] += inter
            tot["pd_lv"] += int(pd_lv.sum())
            tot["gt_lv"] += int(gt_lv.sum())
            tot["n"] += 1
    if tot["n"]:
        print()
        print("=== 合计（%d 帧）===" % tot["n"])
        print("  GT 心肌像素             : %d" % tot["gt_myo"])
        print("  预测心肌像素            : %d  （占 GT 的 %.1f%%）" % (
            tot["pd_myo"], tot["pd_myo"] / tot["gt_myo"] * 100))
        print("  GT 心肌 ∩ 预测 LV 腔     : %d  （占 GT 心肌 %.1f%%）" % (
            tot["gtmyo_in_pdlv"], tot["gtmyo_in_pdlv"] / tot["gt_myo"] * 100))
        print("  预测 LV 腔 / GT LV 腔    : %.2f 倍" % (tot["pd_lv"] / max(1, tot["gt_lv"])))
        print()
        ratio = tot["pd_myo"] / tot["gt_myo"]
        if ratio < 0.15:
            print("→ 判定：**存在同类写序问题**（预测心肌远小于 GT），值得跑修复版重跑。")
        elif ratio < 0.6:
            print("→ 判定：**部分受影响**（预测心肌明显偏小），修复可能小幅提升，可跑可不跑。")
        else:
            print("→ 判定：**写序问题不显著**（预测心肌面积接近 GT），"
                  "ACDC 的 0.518 主要来自薄环分割/跟踪漂移，修复收益有限。")
    else:
        print("未取到可比帧（检查 pred_mask 是否存在）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
