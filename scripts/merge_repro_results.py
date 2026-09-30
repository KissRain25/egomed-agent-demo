"""合并五集复现结果，生成完整 Table II 对照数据（含总体平均）。

背景：评测脚本按数据集分次运行，每集把结果写到 `<RUN_ROOT>/<DS>/class_summary.csv`；
但分次运行的"全数据集汇总"只包含当次跑过的数据集。本脚本把五集的 class_summary
合并成一份完整表，并给出两种口径的总体平均 Dice：

    - **类别宏平均**：所有 (数据集, 类别) 的 mean_dice 直接平均（与"Table II 逐行平均"一致）
    - **帧加权平均**：按 total_valid_gt_frames 加权（更偏向帧多的数据集）

用法：python scripts/merge_repro_results.py [--write]
      --write 时写出 `runs/eval_.../egomed5_all_datasets_class_summary.MERGED.csv` 与 `docs/复现结果_TableII.md`
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = (ROOT / "runs" / "eval_yolo26_original_sam2_video_schedule_reset_retrack_correction"
            / "egomed5_yolo26m_original_sam2_online_schedule_reset_retrack_iou06")
ORDER = ["ACDC", "Montgomery-County-CXR-Set", "CAMUS", "PolypGen2021_MultiCenterData_v3", "Amos"]


def read_summary(ds: str) -> list[dict]:
    p = RUN_ROOT / ds / "class_summary.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="合并五集复现结果")
    ap.add_argument("--write", action="store_true", help="写出合并 CSV 与 Markdown 结果表")
    args = ap.parse_args()

    rows: list[dict] = []
    missing: list[str] = []
    for ds in ORDER:
        got = read_summary(ds)
        if not got:
            missing.append(ds)
            continue
        rows.extend(got)

    print("=" * 104)
    print("五集复现结果合并（论文 Table II 口径：τ₂=0.6，全帧序列）")
    print("=" * 104)
    print("%-38s %-30s %8s %8s %8s %10s" % ("数据集", "类别", "均值", "标准差", "病例数", "有效帧"))
    for r in rows:
        print("%-38s %-30s %8.4f %8.4f %8s %10s" % (
            r["dataset"], r["target_class"], float(r["mean_dice"]), float(r["std_dice"]),
            r["num_cases"], r["total_valid_gt_frames"]))

    if rows:
        macro = sum(float(r["mean_dice"]) for r in rows) / len(rows)
        frames = sum(float(r["total_valid_gt_frames"]) for r in rows)
        weighted = sum(float(r["mean_dice"]) * float(r["total_valid_gt_frames"]) for r in rows) / frames
        print("-" * 104)
        print("类别宏平均 Dice: %.4f（%d 个 (数据集,类别) 行）" % (macro, len(rows)))
        print("帧加权平均 Dice: %.4f（合计 %d 帧）" % (weighted, int(frames)))
        per_ds: dict[str, list[float]] = {}
        for r in rows:
            per_ds.setdefault(r["dataset"], []).append(float(r["mean_dice"]))
        print("\n各数据集平均（该类内宏平均）：")
        for ds in ORDER:
            if ds in per_ds:
                print("   %-38s %.4f（%d 类）" % (ds, sum(per_ds[ds]) / len(per_ds[ds]), len(per_ds[ds])))
    if missing:
        print("\n⚠ 尚缺数据集（未跑完/未产出 class_summary.csv）：", ", ".join(missing))

    if args.write and rows:
        out_csv = RUN_ROOT / "egomed5_all_datasets_class_summary.MERGED.csv"
        fieldnames = list(rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        md = ROOT / "docs" / "复现结果_TableII.md"
        lines = [
            "# 复现结果 · Table II 对照数据（五集 · 全帧序列 · τ₂=0.6）",
            "",
            f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} ｜ 由 `scripts/merge_repro_results.py` 从各集 `class_summary.csv` 合并",
            "",
            "| 数据集 | 类别 | Dice 均值 | 标准差 | 病例数 | 有效帧 |",
            "|--------|------|----------|--------|--------|--------|",
        ]
        for r in rows:
            lines.append("| %s | %s | **%.4f** | %.4f | %s | %s |" % (
                r["dataset"], r["target_class"], float(r["mean_dice"]), float(r["std_dice"]),
                r["num_cases"], r["total_valid_gt_frames"]))
        if rows:
            lines += ["", "**总体**：类别宏平均 **%.4f**；帧加权平均 **%.4f**（合计 %d 帧）。"
                      % (macro, weighted, int(frames))]
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n已写出：\n  %s\n  %s" % (out_csv.relative_to(ROOT), md.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
