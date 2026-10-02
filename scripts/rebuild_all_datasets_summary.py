#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重建"五数据集汇总"文件（某一集单独运行把总表覆盖时用）。

背景
----
`scripts/egomed_agent/egomed-agent-iou06.py` 每轮运行结束时，会按 `DATASETS` 顺序
把各数据集的 per-dataset 结果 concat 起来，写入：

    RUN_ROOT/egomed5_all_datasets_class_summary.csv / .xlsx
    RUN_ROOT/egomed5_all_datasets_reinit_correction_log.csv

若本轮只跑了 1 个数据集（例如"只重跑 CAMUS"），这三个文件就会被**冲成单集内容**
（2026-10-02 实际发生过：14 行总表被覆盖为 CAMUS 3 行，靠本工具重建）。

该脚本现已加 SAFETY 保护：非"5 集齐跑"时改写 `*_partial_<n>ds.*`，不再碰 canonical 文件名。
本工具用于两种场合：① 修补历史上被覆盖的总表；② 按指定数据集集合手工合并。

用法
----
    python scripts/rebuild_all_datasets_summary.py --dry-run          # 只看不写
    python scripts/rebuild_all_datasets_summary.py                    # 用默认 5 集顺序重建
    python scripts/rebuild_all_datasets_summary.py --datasets ACDC CAMUS
    python scripts/rebuild_all_datasets_summary.py --verify-macro 0.5453   # 用宏平均校验重建正确性
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (REPO_ROOT / "runs" / "eval_yolo26_original_sam2_video_schedule_reset_retrack_correction"
                    / "egomed5_yolo26m_original_sam2_online_schedule_reset_retrack_iou06")

# 冻结版脚本里 DATASETS 的顺序（重建时应保持一致，行序才与原表相同）
DEFAULT_DATASETS = ["Amos", "CAMUS", "ACDC", "Montgomery-County-CXR-Set", "PolypGen2021_MultiCenterData_v3"]

SUMMARY_CSV = "egomed5_all_datasets_class_summary.csv"
SUMMARY_XLSX = "egomed5_all_datasets_class_summary.xlsx"
CORR_CSV = "egomed5_all_datasets_reinit_correction_log.csv"


def rebuild(run_root: pathlib.Path, datasets: list[str], dry_run: bool,
            verify_macro: float | None) -> int:
    summaries, corrections, missing = [], [], []
    for ds in datasets:
        cls = run_root / ds / "class_summary.csv"
        cor = run_root / ds / "reinit_correction_log.csv"
        if not cls.exists():
            missing.append(str(cls))
            continue
        summaries.append(pd.read_csv(cls))
        if cor.exists():
            corrections.append(pd.read_csv(cor))

    if missing:
        print("[错误] 缺少各数据集结果，无法重建：")
        for p in missing:
            print("   -", p)
        return 1

    all_summary = pd.concat(summaries, ignore_index=True)
    print("=== 重建内容（%d 个数据集 / %d 行）===" % (len(datasets), len(all_summary)))
    print("%-32s %-34s %8s" % ("数据集", "类别", "Dice"))
    for _, r in all_summary.iterrows():
        print("%-32s %-34s %8.4f" % (r["dataset"], r["target_class"], r["mean_dice"]))

    macro = float(all_summary["mean_dice"].mean())
    print()
    print("类别宏平均 = %.4f" % macro)
    ok = True
    if verify_macro is not None:
        ok = abs(macro - verify_macro) < 0.0002
        print("校验：期望 %.4f → %s" % (verify_macro, "✓ 一致" if ok else "✗ 不一致（请检查数据集清单/顺序）"))

    if dry_run:
        print("\n[dry-run] 未写文件。")
        return 0 if ok else 2

    all_summary.to_csv(run_root / SUMMARY_CSV, index=False)
    all_summary.to_excel(run_root / SUMMARY_XLSX, index=False)
    if corrections:
        pd.concat(corrections, ignore_index=True).to_csv(run_root / CORR_CSV, index=False)
    print("\n已写入：")
    print("  ", run_root / SUMMARY_CSV)
    print("  ", run_root / SUMMARY_XLSX)
    if corrections:
        print("  ", run_root / CORR_CSV)
    return 0 if ok else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="重建五数据集汇总（class_summary + 重初始化日志）")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_ROOT), help="评测产物根目录")
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS,
                    help="按此顺序拼接（默认冻结版 5 集顺序）")
    ap.add_argument("--dry-run", action="store_true", help="只打印不写文件")
    ap.add_argument("--verify-macro", type=float, default=None,
                    help="用已知的类别宏平均校验重建结果（如 0.5453）")
    args = ap.parse_args(argv)
    return rebuild(pathlib.Path(args.run_dir), args.datasets, args.dry_run, args.verify_macro)


if __name__ == "__main__":
    sys.exit(main())
