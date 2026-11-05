# -*- coding: utf-8 -*-
"""补齐 Amos 的 21 个测试病例：1:1 抽帧（来自原始录屏）+ 拷贝逐帧标注（来自网盘完整副本）。

背景（2026-11-05 核查）：
- 评测计划表要 49 个 Amos 测试病例；仓库副本只有 24 例帧+标注齐。
- 另有 21 例：标注在网盘的完整副本里（`Amos/label/<case>/*.png`），但仓库里只有旧的
  "每 3 帧取 1" 子采样帧（148~151 帧），与标注（443~452 帧）不匹配。
- 已验证：录屏 1:1 抽出的第 3k 帧与旧第 k 帧逐像素相同 → 旧帧就是子采样，可安全归档后重抽。
- 仍缺 4 例（186/197/229/260）逐帧标注，两处都没有，不在本脚本范围。

做法：
1. 旧的部分帧目录整体移到 `runs/_amos_img_partial_backup/`（不删除，留证）；
2. 从 `rawdata/<case>/*.mp4` 按 1:1 抽帧，命名 `<case>_%04d.jpg`（与既有约定一致）；
3. 从网盘 `Amos/label/<case>/` 拷贝 `<case>_%04d.png`，过滤百度网盘重复下载产生的 `*(1).png`；
4. 逐例校验 img 数 == label 数 == 视频帧数，任何不一致都报错退出（不静默放过）。

用法：
    python scripts/prep_amos_extra21.py --dry-run     # 只报告将做什么
    python scripts/prep_amos_extra21.py               # 实际执行
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
NET_AMOS = Path(r"D:\BaiduNetdiskDownload\EgoMed-Screen\Amos")
IMG_ROOT = ROOT / "data" / "Amos" / "img"
LABEL_ROOT = ROOT / "data" / "Amos" / "label"
BACKUP_ROOT = ROOT / "runs" / "_amos_img_partial_backup"
PLAN = ROOT / "data" / "text_prompt_eval" / "Amos_test_prompt_schedule.csv"

# 有标注（网盘）但帧不足的 21 例
CASES = ["174", "188", "191", "193", "201", "202", "205", "207", "214", "220",
         "237", "244", "248", "249", "250", "252", "256", "258", "261", "267", "268"]


def plan_cases() -> list[str]:
    if not PLAN.exists():
        return []
    with PLAN.open(encoding="utf-8") as fh:
        return sorted({r["case_id"] for r in csv.DictReader(fh)})


def video_path(case: str) -> Path | None:
    hits = sorted(glob.glob(str(NET_AMOS / "rawdata" / case / "*.mp4")) +
                  glob.glob(str(NET_AMOS / "rawdata" / case / "*.MP4")))
    return Path(hits[0]) if hits else None


def extract_frames(video: Path, out_dir: Path, case: str) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开录屏：{video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{case}_{i:04d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    cap.release()
    return i


def copy_labels(case: str) -> int:
    src_dir = NET_AMOS / "label" / case
    dst_dir = LABEL_ROOT / case
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.glob("*.png")):
        if "(1)" in p.name:          # 百度网盘重复下载残留，跳过
            continue
        shutil.copy2(p, dst_dir / p.name)
        n += 1
    return n


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="补齐 Amos 21 例的帧与标注")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写盘")
    args = ap.parse_args()

    planned = plan_cases()
    extra = [c for c in CASES if c in planned]
    print(f"计划表测试病例 {len(planned)} 个；本次补齐 {len(extra)} 个：{', '.join(extra)}")

    results = []
    for case in extra:
        vid = video_path(case)
        if vid is None:
            raise SystemExit(f"[{case}] 没有录屏，终止")
        cap = cv2.VideoCapture(str(vid))
        vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        lab_src = len([p for p in (NET_AMOS / "label" / case).glob("*.png") if "(1)" not in p.name])

        if args.dry_run:
            old = len(list((IMG_ROOT / case).glob("*.jpg"))) if (IMG_ROOT / case).exists() else 0
            print(f"[dry] {case}: 视频 {vframes} 帧 / 标注 {lab_src} / 现有帧 {old}（将归档后重抽）")
            continue

        old_dir = IMG_ROOT / case
        if old_dir.exists():
            BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
            dst = BACKUP_ROOT / case
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(old_dir), str(dst))

        n_img = extract_frames(vid, IMG_ROOT / case, case)
        n_lab = copy_labels(case)
        ok = (n_img == n_lab == vframes)
        results.append((case, vframes, n_img, n_lab, ok))
        print(f"{case}: 视频 {vframes} / 抽出 {n_img} / 标注 {n_lab} → {'OK' if ok else '不一致!'}")

    if args.dry_run:
        return 0

    bad = [r for r in results if not r[4]]
    print("-" * 68)
    print(f"完成 {len(results)} 例；不一致 {len(bad)} 例")
    if bad:
        print("不一致清单：", ", ".join(f"{c}(视频{v}/帧{i}/标注{l})" for c, v, i, l, _ in bad))
        return 1
    print("全部一致（帧数 == 标注数 == 视频帧数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
