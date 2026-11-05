#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务完成率统计（唯一口径来源）。

背景：`docs/总计划.md` 里同一个 N 编号可能出现两次（主表 + 阶段/专项细分表），
人工数一遍容易出错（不同人不同时间数的数还不一样）。本脚本按固定规则解析：

  1. 分母 = 总计划里出现过的**去重后** N 任务数；
  2. 同一 N 的多行状态合并：任一处有 ☑/✅ -> 已完成；否则任一处有 ◐ -> 进行中；否则未开始；
  3. 百分比保留一位小数。

用法：
  python scripts/task_stats.py            # 打印统计
  python scripts/task_stats.py --md       # 打印可直接贴进文档的 Markdown 表格
  python scripts/task_stats.py --list    # 列出未开始/进行中的任务编号
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "总计划.md"

ROW_RE = re.compile(r"^\|\s*(N\d+)\s*\|")
DONE_MARKS = ("☑", "✅")
PROG_MARK = "◐"


def parse_plan(path: Path = PLAN) -> dict[str, list[str]]:
    """返回 {N编号: [该编号所有出现处的状态标记]}。"""
    if not path.exists():
        raise SystemExit(f"找不到总计划：{path}")
    marks: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        nid = m.group(1)
        cells = [c.strip() for c in line.split("|")]
        last = cells[-2] if len(cells) >= 3 else ""
        sym = "".join(ch for ch in last if ch in DONE_MARKS + (PROG_MARK,))
        marks.setdefault(nid, []).append(sym)
    return marks


def classify(marks: dict[str, list[str]]) -> dict[str, list[str]]:
    total = len(marks)
    done, prog, todo = [], [], []
    for nid, syms in marks.items():
        if any(s in DONE_MARKS for s in syms):
            done.append(nid)
        elif any(PROG_MARK in s for s in syms):
            prog.append(nid)
        else:
            todo.append(nid)
    key = lambda n: int(n[1:])
    return {
        "total": total,
        "done": sorted(done, key=key),
        "prog": sorted(prog, key=key),
        "todo": sorted(todo, key=key),
    }


def main() -> int:
    try:  # Windows 控制台默认 GBK，输出 ☑/✅ 会 UnicodeEncodeError
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="输出 Markdown 表格")
    ap.add_argument("--list", action="store_true", help="列出进行中/未开始编号")
    args = ap.parse_args()

    r = classify(parse_plan())
    total = r["total"]
    pct = lambda n: f"{n / total * 100:.1f}%"
    done, prog, todo = r["done"], r["prog"], r["todo"]

    if args.md:
        print("| 项 | 数量 | 占比 |")
        print("|---|---:|---:|")
        print(f"| N 任务总数（去重） | {total} | 100% |")
        print(f"| **已完成 ☑/✅** | **{len(done)}** | **{pct(len(done))}** |")
        print(f"| 进行中 ◐ | {len(prog)} | {pct(len(prog))} |")
        print(f"| 未开始（空） | {len(todo)} | {pct(len(todo))} |")
    else:
        print(f"总数 {total} | 已完成 {len(done)}（{pct(len(done))}） | "
              f"进行中 {len(prog)}（{pct(len(prog))}） | 未开始 {len(todo)}（{pct(len(todo))}）")
        print(f"含进行中完成率：{pct(len(done) + len(prog))}")

    if args.list:
        print("进行中：" + ("、".join(prog) or "（无）"))
        print("未开始：" + ("、".join(todo) or "（无）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
