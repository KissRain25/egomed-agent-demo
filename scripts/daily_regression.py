#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""每日自动回归（N44）—— 定时把"一键回归"跑一遍，归档、并与上一次对比

为什么要它
----------
接下来一个月会频繁改代码，靠人记得跑回归不现实。本脚本把这件事自动化：

  跑 `client/run_regression.py`（mock + real 两套）
    → 从 `runs/acceptance/<时间戳>/report.json` 汇总每套通过数
    → 追加一行到 `runs/daily/history.csv`（历史趋势）
    → 生成 `runs/daily/latest.md`（本次摘要 + **与上次对比**）
    → **如果出现"上次通过、这次失败"的用例 → 判为回归，退出码 1**

用法
----
    python scripts/daily_regression.py                # 完整回归（含真引擎，约 5~8 分钟）
    python scripts/daily_regression.py --quick        # 只跑 mock 套件（秒级，适合高频）
    python scripts/daily_regression.py --no-run       # 不跑，只汇总已有报告（验证/补录）
    python scripts/daily_regression.py --task-hint    # 打印注册 Windows 计划任务的命令

输出
----
    runs/daily/history.csv     历史：日期 / 套件通过数 / 结论 / 耗时 / 失败用例
    runs/daily/latest.md       本次摘要 + 与上次对比（新增失败会显著标出）
    runs/daily/<时间戳>.log    本次回归的完整终端输出

归属：闫（自动化 / 运维工具）｜ 关联：docs/总计划.md 阶段五 N44
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "client"))
import mock_client as mc  # noqa: E402

PROJECT_ROOT = mc.PROJECT_ROOT
RUNS = PROJECT_ROOT / "runs"
ACC_ROOT = RUNS / "acceptance"
DAILY = RUNS / "daily"
HISTORY = DAILY / "history.csv"
LOCK = DAILY / "_lock"
REGRESSION = PROJECT_ROOT / "client" / "run_regression.py"
LOCK_STALE_SEC = 3600.0        # 锁超过 1 小时视为残留（上次崩溃）
FIELDS = ["ts", "suites", "mock_pass", "mock_total", "real_pass", "real_total",
          "verdict", "seconds", "new_failures", "failed_cases"]


def log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("gbk", errors="replace").decode("gbk"), flush=True)


def acquire_lock() -> bool:
    DAILY.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < LOCK_STALE_SEC:
            log(f"[跳过] 已有回归在跑（锁 {LOCK.name}，{age/60:.1f} 分钟前）")
            return False
        log(f"[提示] 清理残留锁（{age/60:.1f} 分钟前）")
    LOCK.write_text(str(time.time()), encoding="utf-8")
    return True


def release_lock() -> None:
    try:
        LOCK.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def run_regression(quick: bool, log_path: Path) -> int:
    """跑一键回归；完整输出同时落盘。"""
    args = ["--suite", "mock" if quick else "all"]
    cmd = [sys.executable, str(REGRESSION), *args]
    log(f"[跑] {' '.join(args)} → {log_path.name}")
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=fh,
                              stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="ignore")
    return proc.returncode


def collect_reports(since: float | None) -> dict[str, dict]:
    """汇总验收报告。

    since=None → **每个套件取最新一份**（--no-run 用）；
    since=t    → 只取 t 之后新建的（本次跑出来的）。
    返回 {套件: {pass,total,failed[],dir}}；同名套件以更新的目录覆盖。
    """
    out: dict[str, dict] = {}
    if not ACC_ROOT.exists():
        return out
    dirs = sorted([d for d in ACC_ROOT.iterdir() if d.is_dir()], key=lambda d: d.name)
    for d in dirs:
        if since is not None and d.stat().st_mtime < since:
            continue
        f = d / "report.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        suite = str(data.get("head", {}).get("suite") or "?")
        items = data.get("items", [])
        out[suite] = {
            "pass": sum(1 for i in items if i.get("pass")),
            "total": len(items),
            "failed": [f"{i.get('case')} {i.get('name')}" for i in items if not i.get("pass")],
            "dir": str(d.relative_to(PROJECT_ROOT)),
        }
    return out


def read_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    return list(csv.DictReader(HISTORY.open(encoding="utf-8-sig")))


def has_data(r: dict) -> bool:
    """有真实数据的行（mock/real 至少一套跑到了用例）；空行不参与趋势与对比。"""
    try:
        return (int(r.get("mock_total") or 0) + int(r.get("real_total") or 0)) > 0
    except (TypeError, ValueError):
        return False


def prune_history() -> int:
    """把"没取到报告"的空行从 history.csv 里剔掉（自愈，保持趋势干净）。"""
    rows = read_history()
    keep = [r for r in rows if has_data(r)]
    if len(keep) == len(rows):
        return 0
    with HISTORY.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(keep)
    return len(rows) - len(keep)


def latest_before(t0: float) -> dict | None:
    """读 history.csv 里本次之前最近一行有数据的（用于对比）。"""
    rows = [r for r in read_history() if has_data(r)]
    return rows[-1] if rows else None


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="每日自动回归（N44）")
    ap.add_argument("--quick", action="store_true", help="只跑 mock 套件（秒级）")
    ap.add_argument("--no-run", action="store_true", help="不跑，只汇总最近一次已有报告")
    ap.add_argument("--task-hint", action="store_true", help="打印注册计划任务的命令后退出")
    args = ap.parse_args(argv)

    if args.task_hint:
        bat = PROJECT_ROOT / "每日回归.bat"
        log("注册每日定时回归（Windows 计划任务，管理员 CMD 执行一次即可）：")
        log("")
        log(f'  schtasks /create /tn "EgoMed每日回归" /tr "\\"{bat}\\"" /sc daily /st 09:30')
        log("")
        log("  查看：  schtasks /query /tn \"EgoMed每日回归\" /v /fo LIST")
        log("  立即跑：schtasks /run    /tn \"EgoMed每日回归\"")
        log("  删除：  schtasks /delete /tn \"EgoMed每日回归\" /f")
        log("")
        log("建议：日常用 --quick（秒级）；每周一次完整（含真引擎）。")
        return 0

    t0 = time.time()                 # 用于"与上次对比"的时间基准
    since = None if args.no_run else t0
    if not args.no_run:
        if not acquire_lock():
            return 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = DAILY / f"{stamp}.log"
    DAILY.mkdir(parents=True, exist_ok=True)

    rc = 0
    try:
        dropped = prune_history()
        if dropped:
            log(f"[自愈] 历史里剔掉 {dropped} 行空记录（没取到报告的行不参与趋势）")
        started = time.time()
        if not args.no_run:
            rc = run_regression(args.quick, log_path)
        took = time.time() - started

        suites = collect_reports(since)
        if not suites:
            log("[警告] 没找到验收报告（本次跑得太快？启动阶段就失败？或 runs/acceptance 被清过）")
        mock = suites.get("mock", {"pass": 0, "total": 0, "failed": []})
        real = suites.get("real", {"pass": 0, "total": 0, "failed": []})
        all_failed = sorted(mock.get("failed", []) + real.get("failed", []))
        missing = sorted({"mock", "real"} - set(suites))
        if not suites:
            verdict = "未取到报告 ⚠"
        elif all_failed:
            verdict = "存在失败 ❌"
        elif rc == 0 and missing:
            verdict = f"通过（未跑 {'/'.join(missing)}）"
        else:
            verdict = "全部通过 ✅"

        # 与上次对比：找出"上次通过（不在失败列表）、这次失败"的用例 = 回归
        prev = latest_before(t0)
        prev_failed = set(filter(None, (prev or {}).get("failed_cases", "").split("；")))
        new_failures = [c for c in all_failed if c not in prev_failed]

        row = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "suites": "+".join(sorted(suites)) or "-",
            "mock_pass": mock.get("pass", 0), "mock_total": mock.get("total", 0),
            "real_pass": real.get("pass", 0), "real_total": real.get("total", 0),
            "verdict": verdict, "seconds": round(took, 1),
            "new_failures": len(new_failures),
            "failed_cases": "；".join(all_failed),
        }
        if args.no_run:
            log("[提示] --no-run 只做查看，不写入趋势历史")
        else:
            new_file = not HISTORY.exists()
            with HISTORY.open("a", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                w.writerow(row)

        lines = [
            f"# 每日回归 {row['ts']}", "",
            f"- 套件：`{row['suites']}`　耗时 {row['seconds']}s　判定：**{verdict}**",
            f"- mock **{row['mock_pass']}/{row['mock_total']}**　"
            f"real **{row['real_pass']}/{row['real_total']}**", "",
        ]
        if prev:
            lines += ["## 与上次对比", "",
                      f"- 上次：{prev['ts']}　mock {prev['mock_pass']}/{prev['mock_total']}　"
                      f"real {prev['real_pass']}/{prev['real_total']}　{prev['verdict']}"]
        if new_failures:
            lines += ["", "## ⚠ 新增失败（疑似回归）", "",
                      *[f"- `{c}`" for c in new_failures]]
        elif all_failed:
            lines += ["", "## 本次失败（与上次相同）", "", *[f"- `{c}`" for c in all_failed]]
        if suites:
            lines += ["", "## 报告位置", "",
                      *[f"- {k}: `{v['dir']}`" for k, v in sorted(suites.items())]]
        (DAILY / "latest.md").write_text("\n".join(lines), encoding="utf-8")

        log("")
        log("=" * 74)
        log(f"每日回归　{row['ts']}　耗时 {row['seconds']}s")
        log("-" * 74)
        log(f"  mock {row['mock_pass']}/{row['mock_total']}　"
            f"real {row['real_pass']}/{row['real_total']}　{verdict}")
        if new_failures:
            log(f"  [回归] 新增失败 {len(new_failures)} 项：")
            for c in new_failures:
                log(f"        - {c}")
        elif all_failed:
            log(f"  失败 {len(all_failed)} 项（与上次相同，非新回归）")
        log(f"  历史：{HISTORY.relative_to(PROJECT_ROOT)}　摘要："
            f"{(DAILY / 'latest.md').relative_to(PROJECT_ROOT)}")
        log("=" * 74)
        return 1 if (rc != 0 or new_failures or not suites) else 0
    finally:
        if not args.no_run:
            release_lock()


if __name__ == "__main__":
    sys.exit(main())
