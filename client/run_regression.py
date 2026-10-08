#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EgoMed-Agent 一键回归（N22）

一条命令跑完两套验收，并**自动管理服务端进程**（起 → 等就绪 → 跑 → 关）：

  mock 套件 —— 服务端【假实现】（秒级启动，不加载模型）：验协议/字段/坏帧分支（P0.3/P0.4）
  real 套件 —— 服务端【真引擎 --real】（预热 60~90 秒）：验首帧延迟 / 并发丢帧 / 连续帧出结果 / 语音（P0.1/P0.2/A.11）

用法
----
    python client/run_regression.py                 # mock + real（默认：完整回归）
    python client/run_regression.py --suite mock    # 只跑快的（改代码后的即时回归）
    python client/run_regression.py --suite real --concurrency 5
    python client/run_regression.py --no-start      # 服务端已手动起好，只跑用例
    python client/run_regression.py --keep-server   # 跑完不关服务端（接着做别的）

输出
----
    终端   ：两套件 PASS/FAIL 汇总表 + 报告路径
    报告   ：runs/acceptance/<时间戳>/report.md|json（沿用 acceptance_test 的产物）
    服务端日志：runs/regression/<时间戳>/server_out.txt|server_err.txt

归属：闫（客户端 / 测试工具）｜ 关联：docs/总计划.md N22
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = PROJECT_ROOT / "server" / "api.py"
ACC_SCRIPT = PROJECT_ROOT / "client" / "acceptance_test.py"
ACC_ROOT = PROJECT_ROOT / "runs" / "acceptance"
REG_ROOT = PROJECT_ROOT / "runs" / "regression"

WARMUP_TIMEOUT_REAL = 300.0    # 真引擎预热（含 ASR medium），给足余量
READY_TIMEOUT_MOCK = 40.0      # 假实现秒级启动


def log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("gbk", errors="replace").decode("gbk"), flush=True)


def child_env() -> dict:
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def health(url: str) -> dict | None:
    """探活；服务端没起或还没就绪都返回 None。"""
    try:
        import requests
        r = requests.get(f"{url.rstrip('/')}/healthz", timeout=1.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def start_server(port: int, real: bool, log_dir: Path) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = (log_dir / "server_out.txt").open("wb")
    err = (log_dir / "server_err.txt").open("wb")
    cmd = [sys.executable, str(SERVER_SCRIPT), "--port", str(port)]
    if real:
        cmd.append("--real")
    log(f"    · 启动服务端（{'真引擎 --real' if real else '假实现'}，秒级/预热两种节奏）…")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=out, stderr=err, env=child_env())
    log(f"      PID={proc.pid}  日志：{log_dir.relative_to(PROJECT_ROOT)}")
    return proc


def wait_ready(url: str, proc: subprocess.Popen, real: bool) -> bool:
    """轮询 /healthz 直到就绪（真引擎额外要求 real_engine=true）。"""
    timeout = WARMUP_TIMEOUT_REAL if real else READY_TIMEOUT_MOCK
    t0 = time.time()
    last = -1
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            log(f"      [!!] 服务端已退出（returncode={proc.returncode}），看 server_err.txt")
            return False
        h = health(url)
        if h and (h.get("real_engine") if real else True):
            log(f"      [OK] 服务端就绪（{time.time() - t0:.0f}s，sessions={h.get('sessions')}）")
            return True
        if real:
            spent = int(time.time() - t0)
            if spent // 10 != last:
                last = spent // 10
                log(f"      … 预热中 {spent}s")
        time.sleep(1.5)
    log(f"      [!!] 等待就绪超时（{timeout:.0f}s）")
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log("    · 服务端已关闭")


def latest_reports(before: set[Path]) -> list[Path]:
    if not ACC_ROOT.exists():
        return []
    return sorted(p for p in ACC_ROOT.iterdir() if p.is_dir() and p not in before)


def read_report(d: Path) -> tuple[str, int, int]:
    """返回 (结论, 通过数, 总数)。"""
    f = d / "report.json"
    if not f.exists():
        return ("(无 report.json)", 0, 0)
    data = json.loads(f.read_text(encoding="utf-8"))
    items = data.get("items", [])
    ok = sum(1 for i in items if i.get("pass"))
    return (data.get("head", {}).get("result", "?"), ok, len(items))


def run_suite(suite: str, url: str, args) -> tuple[bool, Path | None, str]:
    """跑一个套件；返回 (是否全通过, 报告目录, "通过/总数")。"""
    before = set(ACC_ROOT.iterdir()) if ACC_ROOT.exists() else set()
    log("")
    log(f"  ┌─ 套件 [{suite}] " + "─" * 50)
    cmd = [sys.executable, str(ACC_SCRIPT), "--suite", suite, "--url", url,
           "--budget-ms", str(args.budget_ms)]
    if args.concurrency and suite != "mock":
        cmd += ["--concurrency", str(args.concurrency)]
    if args.seg_frames:
        cmd += ["--seg-frames", str(args.seg_frames)]
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=child_env()).returncode
    log(f"  └─ " + "─" * 60)

    new_dirs = latest_reports(before)
    if new_dirs:
        rep_dir = new_dirs[-1]
        result, ok, total = read_report(rep_dir)
        return (rc == 0, rep_dir, f"{ok}/{total} PASS" if total else result)
    return (rc == 0, None, f"exit={rc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="EgoMed-Agent 一键回归（mock + real 两套验收）")
    ap.add_argument("--suite", choices=["mock", "real", "all"], default="all",
                    help="mock=协议/坏帧（快）；real=延迟/并发/语音（需预热）；all=两套（默认）")
    ap.add_argument("--port", type=int, default=8000, help="服务端端口，默认 8000")
    ap.add_argument("--url", default="", help="服务器地址（默认按 --port 拼 127.0.0.1）")
    ap.add_argument("--budget-ms", type=float, default=3000.0, help="首帧延迟预算（毫秒）")
    ap.add_argument("--concurrency", type=int, default=0, help="real 套件并发帧数（0=跳过）")
    ap.add_argument("--seg-frames", type=int, default=12, help="R4 连续发帧数")
    ap.add_argument("--no-start", action="store_true", help="服务端已手动起好，不自动起/关")
    ap.add_argument("--keep-server", action="store_true", help="跑完不关服务端")
    args = ap.parse_args()

    url = args.url or f"http://127.0.0.1:{args.port}"
    suites = ["mock", "real"] if args.suite == "all" else [args.suite]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log("=" * 72)
    log(f"EgoMed-Agent 一键回归  套件={args.suite}  服务器={url}  {stamp}")
    log("=" * 72)

    proc: subprocess.Popen | None = None
    started_by_us = False
    h = health(url)
    if args.no_start:
        if not h:
            log(f"[错误] --no-start 但 {url} 探活失败：请先启动服务端")
            return 2
        log(f"[1/3] 使用已有服务端（real_engine={h.get('real_engine')}）")
    elif h:
        log(f"[1/3] 检测到已有服务端（real_engine={h.get('real_engine')}），直接使用；本次不会关它")
    else:
        need_real = "real" in suites
        log(f"[1/3] 未发现服务端 → 自动启动（{'真引擎' if need_real else '假实现'}）")
        proc = start_server(args.port, need_real, REG_ROOT / stamp)
        started_by_us = True
        log("[2/3] 等待服务端就绪 …")
        if not wait_ready(url, proc, need_real):
            stop_server(proc)
            return 2

    # 已有服务端但套件不匹配时给出明确提示（避免 real 套件打到假实现上）
    h2 = health(url)
    if h2 and "real" in suites and not h2.get("real_engine"):
        log("[警告] 当前服务端是【假实现】，real 套件需 `python server/api.py --real`")
        log("       已自动跳过 real 套件")
        suites = [s for s in suites if s != "real"]

    log("[3/3] 开始跑用例 …")
    results: list[tuple[str, bool, Path | None, str]] = []
    try:
        for s in suites:
            ok, rep_dir, summary = run_suite(s, url, args)
            results.append((s, ok, rep_dir, summary))
    finally:
        if started_by_us and not args.keep_server:
            stop_server(proc)

    log("")
    log("=" * 72)
    log("回归汇总")
    log("-" * 72)
    log(f"  {'套件':<8}{'结果':<8}{'通过':<12}报告")
    all_ok = True
    for s, ok, rep_dir, summary in results:
        all_ok = all_ok and ok
        mark = "PASS" if ok else "FAIL"
        rel = rep_dir.relative_to(PROJECT_ROOT) if rep_dir else "-"
        log(f"  {s:<8}{mark:<8}{summary:<12}{rel}")
    log("-" * 72)
    log(f"  总判定：{'全部通过 ✅' if all_ok else '存在失败 ❌'}")
    log("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
