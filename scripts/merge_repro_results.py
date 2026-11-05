"""合并五集复现结果，生成完整主结果对照数据（含总体平均与论文值对照）。

背景：评测脚本按数据集分次运行，每集把结果写到 `<RUN_ROOT>/<DS>/class_summary.csv`；
但分次运行的"全数据集汇总"只包含当次跑过的数据集。本脚本把五集的 class_summary
合并成一份完整表，并给出两种口径的总体平均 Dice：

    - **类别宏平均**：所有 (数据集, 类别) 的 mean_dice 直接平均
    - **帧加权平均**：按 total_valid_gt_frames 加权（更偏向帧多的数据集）
    - **论文同口径（12 目标）**：只取论文 Table I 报过的目标，与论文 71.34% 直接可比

论文值来源：`2607.17341v1.pdf` 第 6 页 **Table I**（2026-11-05 抄录）。
注意：论文主结果表是 **Table I**，**Table II 是模块消融表**（此前文档把主结果写成 Table II 属命名错误）。

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

# 论文 Table I 的 12 个目标（%）。CAMUS 论文只报 Left Atrium，心肌/LV 无论文值。
PAPER_TABLE1 = {
    ("Amos", "liver"): 71.69,
    ("Amos", "left kidney"): 76.23,
    ("Amos", "right kidney"): 59.68,
    ("Amos", "spleen"): 69.00,
    ("Amos", "stomach"): 49.88,
    ("ACDC", "RV cavity"): 57.87,
    ("ACDC", "LV cavity"): 77.82,
    ("ACDC", "myocardium"): 51.95,
    ("CAMUS", "Left Atrium"): 81.43,
    ("Montgomery-County-CXR-Set", "left lung"): 92.13,
    ("Montgomery-County-CXR-Set", "right lung"): 90.32,
    ("PolypGen2021_MultiCenterData_v3", "polyp"): 78.02,
}
PAPER_AVG_12 = 71.34


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
    print("五集复现结果合并（论文 Table I 口径：τ₂=0.6，全帧序列）")
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
            "# 复现结果 · 主结果表对照数据（五集 · 全帧序列 · τ₂=0.6）",
            "",
            f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} ｜ 由 `scripts/merge_repro_results.py` 从各集 `class_summary.csv` 合并",
            "> 论文值来源：`2607.17341v1.pdf` p.6 **Table I**（主结果表；**Table II 是模块消融表**）。逐行分析见 `docs/复现报告.md` §六。",
            "",
            "| 数据集 | 类别 | **论文值（%）** | **我们实测（%）** | 差异（pp） | 标准差 | 病例数 | 有效帧 |",
            "|--------|------|---------------|------------------|-----------|--------|--------|--------|",
        ]
        matched: list[float] = []
        for r in rows:
            key = (r["dataset"], r["target_class"])
            ours = float(r["mean_dice"]) * 100
            paper = PAPER_TABLE1.get(key)
            if paper is None:
                paper_s, diff_s = "**论文未报告**", "—"
            else:
                matched.append(ours)
                paper_s, diff_s = "%.2f" % paper, "**%+.2f**" % (ours - paper)
            lines.append("| %s | %s | %s | **%.2f** | %s | %.4f | %s | %s |" % (
                r["dataset"], r["target_class"], paper_s, ours, diff_s,
                float(r["std_dice"]), r["num_cases"], r["total_valid_gt_frames"]))
        lines += [
            "",
            "**总体**：",
            "- 我方全测口径（%d 行，含论文未报告的类别）：类别宏平均 **%.4f**；帧加权平均 **%.4f**（合计 %d 帧）。"
            % (len(rows), macro, weighted, int(frames)),
        ]
        if matched:
            lines.append("- **论文同口径（%d 个目标，与论文 %.2f%% 可比）：我们 %.2f%%，差 %+.2fpp**。"
                         % (len(matched), PAPER_AVG_12, sum(matched) / len(matched),
                            sum(matched) / len(matched) - PAPER_AVG_12))
        lines += [
            "",
            "**差距分布（一句话）**：ACDC 三行几乎逐位重合（≤0.23pp）→ 评测口径已对齐；其余差距按大小依次为 "
            "X 光 −23.0pp > 息肉 −14.5pp > CAMUS LA −12.9pp > Amos −12.0pp，**归因均未查到底**"
            "（X 光优先，见 `docs/复现报告.md` §7 第 1b 项）。",
        ]
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n已写出：\n  %s\n  %s" % (out_csv.relative_to(ROOT), md.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
