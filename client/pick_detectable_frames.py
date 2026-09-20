#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挑"可检出帧"生成测试样例（N14 样例集整改工具）

背景：`data/samples/polyp` 的 12 帧在 YOLO 下 0 检出（帧内无可见息肉），
导致息肉模态的联调/演示必然"没反应"；`data/samples/acdc` 前几帧同样无目标。
本工具用与引擎相同的 YOLO 权重筛帧，只保留**目标可被检出**的帧，
生成新的样例集（保留原样例到 `*_screenrec/` 作为"屏幕翻拍难样本"）。

用法：
    # 息肉样例整改（默认输出 data/samples/polyp，旧样例备份到 data/samples/polyp_screenrec）
    python client/pick_detectable_frames.py --modality "PolypGen (内窥镜)" \
        --raw data/PolypGen2021_MultiCenterData_v3/img --out data/samples/polyp --count 12

    # ACDC 样例整改（让前几帧就有目标，便于首帧/实时链路验收）
    python client/pick_detectable_frames.py --modality "ACDC (MRI 心脏)" \
        --raw data/ACDC/img --out data/samples/acdc --count 12

    # 只统计不写文件
    python client/pick_detectable_frames.py --modality "PolypGen (内窥镜)" \
        --raw data/PolypGen2021_MultiCenterData_v3/img --dry-run

归属：闫（测试工具）｜ 关联：docs/联调发现_20260920_真引擎实时链路.md 发现 3
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "demo"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def rel(p: Path) -> str:
    """尽量输出项目内相对路径（用户可能传相对/绝对路径）"""
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def parse_args():
    p = argparse.ArgumentParser(description="挑可检出帧生成测试样例（N14）")
    p.add_argument("--modality", required=True,
                   help='模态名（egomed_demo.MODALITIES 的键），如 "PolypGen (内窥镜)"')
    p.add_argument("--raw", required=True, help="原始数据根目录（其下按病例分子目录）")
    p.add_argument("--out", default="", help="输出样例目录（默认只统计）")
    p.add_argument("--count", type=int, default=12, help="输出帧数（默认 12）")
    p.add_argument("--conf", type=float, default=0.3, help="检出置信度阈值（默认 0.3）")
    p.add_argument("--max-scan", type=int, default=400, help="最多扫描多少帧（防止全库遍历）")
    p.add_argument("--contiguous", action="store_true",
                   help="优先挑最长连续帧段（SAM2 窗口跟踪更稳，推荐给实时链路用例）")
    p.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    return p.parse_args()


def longest_run(picked: list[tuple[Path, str, float]], count: int):
    """从命中帧里找最长连续段（同病例 + 文件名数字连续），返回落脚点 list。"""
    best: list = []
    i = 0
    while i < len(picked):
        j = i
        while j + 1 < len(picked):
            a, b = picked[j][0], picked[j + 1][0]
            if a.parent == b.parent and _num(b) == _num(a) + 1:
                j += 1
            else:
                break
        run = picked[i:j + 1]
        if len(run) > len(best):
            best = run
        i = j + 1
    return best[:count] if best else picked[:count]


def _num(p: Path) -> int:
    try:
        return int(p.stem.split("_")[-1])
    except ValueError:
        return -1


def main() -> int:
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    import cv2
    import egomed_demo as demo

    if args.modality not in demo.MODALITIES:
        print(f"未知模态：{args.modality}\n可选：{list(demo.MODALITIES)}")
        return 2

    raw_root = Path(args.raw).resolve()
    if not raw_root.is_dir():
        print(f"原始数据目录不存在：{raw_root}")
        return 2

    _pred, yolo = demo._get_models(args.modality)
    names = demo.eng.load_yolo_class_names_from_model(yolo)
    expected = set(demo.MODALITIES[args.modality]["name_to_gray"].keys())
    print(f"模态：{args.modality}")
    print(f"模型类别：{list(names.values())}")
    print(f"配置目标：{sorted(expected)}")
    print(f"扫描 {raw_root}（阈值 conf≥{args.conf}，最多 {args.max_scan} 帧）…")

    cases = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    scanned = 0
    picked: list[tuple[Path, str, float]] = []      # (路径, 类别, 置信度)
    # --contiguous 需要更多候选来挑连续段，扫描上限放宽到 count 的 10 倍
    need = args.count if not args.contiguous else max(args.count * 10, args.count + 20)
    for case in cases:
        frames = sorted([f for f in case.iterdir() if f.suffix.lower() in IMG_EXTS])
        for f in frames:
            if scanned >= args.max_scan or len(picked) >= need:
                break
            scanned += 1
            img = cv2.imread(str(f))
            if img is None:
                continue
            res = yolo.predict(f, conf=args.conf, verbose=False)
            best = None
            for r in res:
                for b in r.boxes:
                    cname = names.get(int(b.cls))
                    if cname in expected:
                        c = float(b.conf)
                        if best is None or c > best[1]:
                            best = (cname, c)
            if best:
                picked.append((f, best[0], best[1]))
        if scanned >= args.max_scan or len(picked) >= need:
            break

    print(f"共扫描 {scanned} 帧，命中 {len(picked)} 帧（conf≥{args.conf}）")
    if not picked:
        print("未找到任何可检出帧：请降低 --conf 或扩大 --max-scan/换 --raw 目录")
        return 1

    if args.contiguous:
        run = longest_run(picked, args.count)
        print(f"[--contiguous] 最长连续段 {len(longest_run(picked, 10 ** 6))} 帧，"
              f"采用前 {len(run)} 帧")
        picked = run
    else:
        picked = picked[:args.count]

    for i, (f, cname, c) in enumerate(picked[:12], 1):
        print(f"  {i:2d}. {rel(f)}  → {cname} {c:.3f}")

    if args.dry_run or not args.out:
        print("\n[dry-run] 未写文件。")
        return 0

    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        backup = out_dir.parent / f"{out_dir.name}_screenrec"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(out_dir, backup)
        print(f"\n原样例已备份到：{backup.relative_to(PROJECT_ROOT)}（作为屏幕翻拍难样本）")
        for f in out_dir.iterdir():
            if f.is_file():
                f.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (f, _cname, _c) in enumerate(picked, 1):
        shutil.copy(f, out_dir / f"frame_{i:04d}.jpg")
    print(f"已写入 {len(picked)} 帧 → {out_dir.relative_to(PROJECT_ROOT)}"
          f"（命名 frame_0001.jpg … 保持沿用原约定）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
