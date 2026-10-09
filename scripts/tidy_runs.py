#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""产物卫生（N45）—— 把 `runs/` 根目录的临时文件**归档**，并盯着体积

为什么用"归档"而不是删除
------------------------
1. 本机宿主环境有「批量删除守卫」：一次删除累计到阈值会抛 `SystemExit`，
   连评测/服务端进程都可能被牵连（见 `docs/工作日志.md` 2026-10-08、派工单 T11）。
2. 归档 = **移动**，既不触发守卫，也不会误删你还想看的东西；
   真要大扫除时再用 `--purge-archive N --yes` 显式删 N 天前的归档。

做什么
------
① **归档**：`runs/` **根目录下**的临时文件（`tmp_*` / `_eval_*` / `_soak_*` / `_console_*`
   / `_reg_*` / `_mock_*` / `_mixed_*` / `_parse_*` / `_desktop_*` / 单下划线开头的散件）
   → 移到 `runs/_archive/<YYYYMMDD>/`（按文件修改日期分桶，保留原名）
   注：**只动根目录的文件**，`runs/` 下的子目录（报告/帧/评测产物）一律不碰。
② **体积**：列出 `runs/` 各一级子目录体积 Top N + 总量，超阈值给告警。
③ （可选）`--purge-archive N --yes`：删除 N 天前的归档目录。

用法
----
    python scripts/tidy_runs.py --dry-run          # 只打印将要归档什么（先看再动）
    python scripts/tidy_runs.py                    # 执行归档
    python scripts/tidy_runs.py --keep-days 1      # 最近 1 天内的临时文件先留着
    python scripts/tidy_runs.py --top 15           # 体积榜显示 15 项
    python scripts/tidy_runs.py --purge-archive 30 --yes   # 删 30 天前的归档

归属：闫（运维工具）｜ 关联：docs/总计划.md 阶段五 N45
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
ARCHIVE = RUNS / "_archive"
SIZE_WARN_GB = 20.0
TEMP_PREFIXES = ("tmp_", "_eval_", "_soak_", "_console_", "_reg_", "_mock_",
                 "_mixed_", "_parse_", "_desktop_", "_mock_demo")
KEEP_ALWAYS = {"_lock"}          # 绝不动的东西


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def is_temp(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.name in KEEP_ALWAYS:
        return False
    return p.name.startswith(TEMP_PREFIXES) or (
        p.name.startswith("_") and p.suffix.lower() in (".py", ".txt", ".json", ".out", ".err",
                                                        ".log", ".md", ".ps1", ".bat"))


def dir_size(p: Path) -> int:
    total = 0
    try:
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except Exception:  # noqa: BLE001
        pass
    return total


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="runs/ 产物卫生（N45）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不实际移动")
    ap.add_argument("--keep-days", type=float, default=0.0,
                    help="最近 N 天内的临时文件先留着（默认 0=都归档）")
    ap.add_argument("--top", type=int, default=10, help="体积榜显示前 N 项（默认 10）")
    ap.add_argument("--size-warn-gb", type=float, default=SIZE_WARN_GB,
                    help=f"体积告警阈值（GB，默认 {SIZE_WARN_GB:g}）")
    ap.add_argument("--purge-archive", type=float, default=0.0, metavar="DAYS",
                    help="删除 N 天前的归档目录（需同时给 --yes）")
    ap.add_argument("--yes", action="store_true", help="确认执行 --purge-archive")
    args = ap.parse_args(argv)

    if not RUNS.exists():
        print(f"[错误] 找不到 {RUNS}")
        return 1

    print("=" * 76)
    print("runs/ 产物卫生（N45）" + ("　[dry-run 只打印]" if args.dry_run else ""))
    print("=" * 76)

    # ---------- ① 归档根目录临时文件 ----------
    cutoff = time.time() - args.keep_days * 86400 if args.keep_days > 0 else None
    moves: list[tuple[Path, Path]] = []
    for p in sorted(RUNS.iterdir()):
        if not is_temp(p):
            continue
        if cutoff and p.stat().st_mtime >= cutoff:
            continue
        bucket = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%d")
        dst = ARCHIVE / bucket / p.name
        if dst.exists():
            dst = ARCHIVE / bucket / f"{p.stem}_{int(p.stat().st_mtime)}{p.suffix}"
        moves.append((p, dst))

    total_bytes = sum(p.stat().st_size for p, _ in moves)
    print(f"[归档] 命中根目录临时文件 {len(moves)} 个（{human(total_bytes)}）")
    if moves:
        buckets = sorted({d.parent.name for _, d in moves})
        print(f"       目标：runs/_archive/<{', '.join(buckets)}>/")
        for src, dst in moves[:6]:
            print(f"       · {src.name} → _archive/{dst.parent.name}/")
        if len(moves) > 6:
            print(f"       · …另有 {len(moves) - 6} 个")
    if moves and not args.dry_run:
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dst))
            except Exception as exc:  # noqa: BLE001
                print(f"       [警告] 移动失败 {src.name}：{exc}")
        print(f"       ✔ 已归档 {len(moves)} 个")

    # ---------- ② 体积榜 ----------
    print()
    subs: list[tuple[int, str]] = []
    root_files = 0
    for p in RUNS.iterdir():
        if p.is_dir():
            subs.append((dir_size(p), p.name))
        elif p.is_file():
            root_files += p.stat().st_size
    subs.sort(reverse=True)
    total = sum(s for s, _ in subs) + root_files
    print(f"[体积] runs/ 合计 {human(total)}（其中根目录散件 {human(root_files)}）")
    print(f"       一级子目录 Top {args.top}：")
    for size, name in subs[:args.top]:
        flag = "  ← 大头" if total and size / max(total, 1) > 0.3 else ""
        print(f"       {human(size):>10}　{name}{flag}")
    warn = args.size_warn_gb * 1024**3
    if total > warn:
        print(f"       [告警] 超过阈值 {args.size_warn_gb:g} GB —— 大头通常是 pred_mask / "
              f"soak|console 的 frames；确认无用后分批清理（勿一次删上万文件）")
    else:
        print(f"       体积正常（阈值 {args.size_warn_gb:g} GB）")

    # ---------- ③ 清理归档 ----------
    if args.purge_archive > 0:
        if not args.yes:
            print()
            print("[跳过] --purge-archive 需要同时给 --yes")
            return 0
        limit = time.time() - args.purge_archive * 86400
        victims = [d for d in ARCHIVE.iterdir()
                   if d.is_dir() and d.stat().st_mtime < limit] if ARCHIVE.exists() else []
        freed = sum(dir_size(d) for d in victims)
        print()
        print(f"[清理] 准备删除 {len(victims)} 个 {args.purge_archive:g} 天前的归档"
              f"（{human(freed)}）")
        for d in victims:
            try:
                shutil.rmtree(d)
                print(f"       · 已删 _archive/{d.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"       [警告] 删除失败 {d.name}：{exc}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
