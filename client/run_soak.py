#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长稳基线（N24）—— 客户端 + 服务端连续跑，周期采样 fps / P95 / 内存 / 显存

干什么
------
起真引擎服务端 → 设好分割目标 → 以固定节奏连续发帧（默认 **1 小时 @2fps**），
每 `--sample-sec` 秒记一个采样桶：

  · 客户端：提交数、实际 fps、客户端耗时 P50/P95、错误数
  · 服务端：自报耗时 P50/P95、`frames_processed` 累计
  · 资源  ：服务端进程 RSS（psutil 或 tasklist 兜底）、GPU 显存（nvidia-smi）

跑完自动判定（N24 验收标准）
---------------------------
  1. **无崩溃**：服务端进程全程存活
  2. **显存零增长**：末段显存 − 首段显存 ≤ 50 MB（且 ≤ 首段的 20%）
  3. **不退化**：末段 P95 相对首段恶化 **< 10%**
  4. 错误 0、实际 fps ≥ 目标的 90%

用法
----
    # 标准（真引擎，1 小时）
    python client/run_soak.py

    # 冒烟自测（假服务端，1 分钟，秒级起）
    python client/run_soak.py --fake --minutes 1 --fps 5 --sample-sec 15

    # 服务端已手动起好
    python client/run_soak.py --no-start --minutes 30

输出
----
    终端：每桶一行的进度 + 末尾判定
    文件：runs/soak/<时间戳>/report.md + report.json + server_out.txt/server_err.txt

归属：闫（客户端 / 测试工具）｜ 关联：docs/总计划.md 阶段四 N24、派工单 T3b/T10
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_client as mc  # noqa: E402

PROJECT_ROOT = mc.PROJECT_ROOT
SERVER_SCRIPT = PROJECT_ROOT / "server" / "api.py"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "samples" / "acdc"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "samples" / "audio"
OUT_ROOT = PROJECT_ROOT / "runs" / "soak"

GPU_GROWTH_MB = 50.0      # 显存允许增长上限（MB）
P95_DEGRADE_PCT = 10.0    # P95 允许恶化上限（%）


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
    try:
        import requests
        r = requests.get(f"{url.rstrip('/')}/healthz", timeout=1.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def gpu_mem_mb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="ignore", timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def rss_mb(pid: int) -> float | None:
    """进程常驻内存（MB）：优先 psutil，退回 tasklist。"""
    try:
        import psutil  # type: ignore
        return round(psutil.Process(pid).memory_info().rss / 1048576, 1)
    except Exception:
        pass
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="ignore", timeout=10)
        line = out.stdout.strip().splitlines()[0]
        kb = int(line.split(",")[-1].strip().strip('"').replace(" K", "").replace(",", ""))
        return round(kb / 1024, 1)
    except Exception:
        return None


def find_server_pid() -> int | None:
    """--no-start 时，从进程表里找出服务端 PID（否则采不到 RSS）。"""
    try:
        import psutil  # type: ignore
        for pr in psutil.process_iter(["pid", "cmdline"]):
            cl = " ".join(pr.info.get("cmdline") or [])
            if "api.py" in cl and "server" in cl:
                return int(pr.info["pid"])
    except Exception:
        pass
    return None


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[k]


def start_server(port: int, real: bool, out_dir: Path) -> subprocess.Popen:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir / "server_out.txt").open("wb")
    err = (out_dir / "server_err.txt").open("wb")
    cmd = [sys.executable, str(SERVER_SCRIPT), "--port", str(port)]
    if real:
        cmd.append("--real")
    log(f"  · 启动服务端（{'真引擎 --real（预热 60~90s）' if real else '假实现'}）…")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=out, stderr=err, env=child_env())
    log(f"    PID={proc.pid}")
    return proc


def wait_ready(url: str, proc: subprocess.Popen, real: bool, timeout: float) -> bool:
    t0 = time.time()
    last = -1
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            log(f"    [!!] 服务端已退出（returncode={proc.returncode}）")
            return False
        h = health(url)
        if h and (h.get("real_engine") if real else True):
            log(f"    [OK] 就绪（{time.time() - t0:.0f}s，sessions={h.get('sessions')}）")
            return True
        spent = int(time.time() - t0)
        if real and spent // 10 != last:
            last = spent // 10
            log(f"    … 预热中 {spent}s")
        time.sleep(1.5)
    log("    [!!] 等待就绪超时")
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log("  · 服务端已关闭")


def main() -> int:
    ap = argparse.ArgumentParser(description="长稳基线（N24）")
    ap.add_argument("--minutes", type=float, default=60.0, help="总时长（分钟），默认 60")
    ap.add_argument("--fps", type=float, default=2.0, help="目标提交帧率，默认 2")
    ap.add_argument("--sample-sec", type=float, default=60.0, help="采样桶长度（秒），默认 60")
    ap.add_argument("--url", default="", help="服务器地址（默认按 --port 拼）")
    ap.add_argument("--port", type=int, default=8000, help="服务端端口，默认 8000")
    ap.add_argument("--fake", action="store_true", help="用假实现（冒烟自测，秒级）")
    ap.add_argument("--no-start", action="store_true", help="服务端已手动起好，不自动起/关")
    ap.add_argument("--timeout", type=float, default=10.0, help="单请求超时秒数，默认 10")
    ap.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="样例帧目录")
    ap.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR), help="样例语音目录（设目标用）")
    ap.add_argument("--out", default="", help="输出目录（默认 runs/soak/<时间戳>）")
    args = ap.parse_args()

    url = args.url or f"http://127.0.0.1:{args.port}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (OUT_ROOT / stamp)
    if not out_dir.is_absolute():          # 允许传相对路径（相对项目根）
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    real = not args.fake

    try:
        shown = out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = out_dir
    log("=" * 78)
    log(f"N24 长稳基线　{args.minutes:g} 分钟 @ {args.fps:g} fps　服务器 {url}")
    log(f"输出目录：{shown}")
    log("=" * 78)

    jpgs = [mc.compress_image(p) for p in
            sorted([q for q in Path(args.image_dir).iterdir() if q.suffix.lower() in mc.IMG_EXTS],
                   key=mc.natural_key)]
    wavs = sorted(Path(args.audio_dir).glob("*.wav"), key=mc.natural_key)
    log(f"  · 样例帧 {len(jpgs)} 张；语音 {len(wavs)} 段")

    proc: subprocess.Popen | None = None
    started_by_us = False
    sid = mc.make_session_id("soak")
    if args.no_start:
        if not health(url):
            log(f"[错误] --no-start 但 {url} 探活失败")
            return 2
        log("[1/3] 使用已有服务端")
    else:
        log("[1/3] 启动服务端并等待就绪 …")
        proc = start_server(args.port, real, out_dir)
        started_by_us = True
        if not wait_ready(url, proc, real, 300.0 if real else 40.0):
            stop_server(proc)
            return 2

    # 设目标（没 target 服务端不跑推理 → 长稳会变成无意义的空跑，必须拿到）
    log("[2/3] 设置分割目标（逐个样例语音试，跳过歧义句）…")
    target, used_wav, _, terr = mc.set_target(url, sid, Path(args.audio_dir), 15.0)
    if not target:
        log(f"  [错误] 没能设上 target（err={terr}）—— 没有目标时服务端不跑推理，"
            f"长稳会空跑，已中止")
        if started_by_us and proc is not None:
            stop_server(proc)
        return 2
    log(f"  [OK] target = {target!r}（来自 {used_wav}）")

    # ---------- 主循环 ----------
    log(f"[3/3] 开始连续压测：{args.minutes:g} 分钟 @ {args.fps:g} fps，"
        f"每 {args.sample_sec:g}s 记一桶 …")
    rows: list[dict] = []
    buckets: list[dict] = []
    t_start = time.perf_counter()
    t_end = t_start + args.minutes * 60.0
    interval = 1.0 / max(args.fps, 0.01)
    next_t = time.perf_counter()
    last_sample = next_t
    n_bucket = 0
    server_pid = proc.pid if proc else find_server_pid()
    if server_pid:
        log(f"  · 被采样的服务端 PID = {server_pid}（用于 RSS 曲线）")
    else:
        log("  · [警告] 没找到服务端 PID，本次不采 RSS")
    crashed = False

    try:
        i = 0
        while time.perf_counter() < t_end:
            payload, elapsed, err = mc.post_frame(
                url, sid, jpgs[i % len(jpgs)], f"soak_{i:06d}.jpg", args.timeout)
            s_ms = None
            if isinstance(payload, dict):
                try:
                    s_ms = int(payload.get("elapsed_ms"))
                except (TypeError, ValueError):
                    s_ms = None
            rows.append({"i": i, "c": round(elapsed, 1), "s": s_ms, "err": err,
                         "t": round(time.perf_counter() - t_start, 1)})
            i += 1

            if proc is not None and proc.poll() is not None:
                crashed = True
                log(f"  [!!] 服务端在 {time.perf_counter() - t_start:.0f}s 退出"
                    f"（returncode={proc.returncode}）→ 停止压测")
                break

            now = time.perf_counter()
            if now - last_sample >= args.sample_sec:
                n_bucket += 1
                rows_b = [r for r in rows if r["t"] > last_sample - t_start]
                ok_b = [r for r in rows_b if not r["err"]]
                b = {
                    "bucket": n_bucket,
                    "min": round((now - t_start) / 60.0, 1),
                    "n": len(rows_b),
                    "fps": round(len(rows_b) / max(now - last_sample, 1e-6), 2),
                    "c_p50": pct([r["c"] for r in ok_b], 50),
                    "c_p95": pct([r["c"] for r in ok_b], 95),
                    "s_p50": pct([r["s"] for r in ok_b if r["s"] is not None], 50),
                    "s_p95": pct([r["s"] for r in ok_b if r["s"] is not None], 95),
                    "err": sum(1 for r in rows_b if r["err"]),
                    "gpu_mb": gpu_mem_mb(),
                    "rss_mb": rss_mb(server_pid) if server_pid else None,
                }
                buckets.append(b)
                log(f"  [{b['min']:>5.1f} min] 桶{b['bucket']:<3} n={b['n']:<5} fps={b['fps']:<5} "
                    f"客户端 P50/P95={b['c_p50']:.0f}/{b['c_p95']:.0f}ms "
                    f"服务端 P50/P95={b['s_p50']:.0f}/{b['s_p95']:.0f}ms "
                    f"err={b['err']} | RSS={b['rss_mb']}MB GPU={b['gpu_mb']}MB")
                last_sample = now
                # 边跑边落盘（中途被打断也有数据）
                _dump(out_dir, args, url, target, rows, buckets, None)

            next_t += interval
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        log("\n  [中断] 用户终止，保留已有数据")

    wall = time.perf_counter() - t_start
    if started_by_us and proc is not None:
        stop_server(proc)

    # ---------- 判定 ----------
    verdict, checks = _judge(args, rows, buckets, crashed, wall)
    log("")
    log("=" * 78)
    log("长稳结论")
    log("-" * 78)
    for name_, ok, detail in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name_}：{detail}")
    log("-" * 78)
    log(f"  总判定：{verdict}")
    log("=" * 78)
    _dump(out_dir, args, url, target, rows, buckets, (verdict, checks))
    log(f"报告已保存：{out_dir / 'report.md'}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def _judge(args, rows, buckets, crashed, wall):
    checks: list[tuple[str, bool, str]] = []
    total = len(rows)
    errs = sum(1 for r in rows if r["err"])
    fps_avg = total / max(wall, 1e-6)
    checks.append(("无崩溃", not crashed, "服务端全程存活" if not crashed else "服务端中途退出"))
    checks.append(("错误 0", errs == 0, f"{errs}/{total} 请求出错"))
    checks.append((f"实际 fps ≥ 目标 90%", fps_avg >= args.fps * 0.9,
                   f"实测 {fps_avg:.2f} fps（目标 {args.fps:g}）"))
    if len(buckets) >= 2:
        head = buckets[: max(1, len(buckets) // 6)]
        tail = buckets[-max(1, len(buckets) // 6):]
        h95 = max(b["c_p95"] for b in head)
        t95 = max(b["c_p95"] for b in tail)
        deg = (t95 - h95) / max(h95, 1e-6) * 100
        checks.append(("P95 退化 <10%", deg < P95_DEGRADE_PCT,
                       f"首段 {h95:.0f}ms → 末段 {t95:.0f}ms（{deg:+.1f}%）"))
        gs = [b["gpu_mb"] for b in buckets if b["gpu_mb"] is not None]
        if gs:
            grow = gs[-1] - gs[0]
            limit = min(GPU_GROWTH_MB, max(gs[0], 1.0) * 0.2)
            checks.append(("显存零增长", grow <= limit,
                           f"{gs[0]:.0f}MB → {gs[-1]:.0f}MB（{grow:+.0f}MB，上限 {limit:.0f}MB）"))
        rs = [b["rss_mb"] for b in buckets if b["rss_mb"] is not None]
        if rs:
            checks.append(("服务端内存无明显增长（观察项）", rs[-1] - rs[0] <= 200.0,
                           f"{rs[0]:.0f}MB → {rs[-1]:.0f}MB（{rs[-1] - rs[0]:+.0f}MB）"))
    else:
        checks.append(("采样桶 ≥2", False, f"只有 {len(buckets)} 个桶，无法判退化"))
    return ("全部通过 ✅" if all(ok for _, ok, _ in checks) else "存在未通过 ❌"), checks


def _dump(out_dir: Path, args, url, target, rows, buckets, final) -> None:
    bad = [r for r in rows if r["err"]]
    hist: dict[str, int] = {}
    for r in bad:
        hist[r["err"]] = hist.get(r["err"], 0) + 1
    data = {
        "head": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "url": url,
                 "minutes": args.minutes, "fps": args.fps, "fake": args.fake,
                 "target": target, "requests": len(rows), "errors": len(bad),
                 "error_hist": hist, "rows": rows},
        "buckets": buckets, "final": ({"verdict": final[0],
                                       "checks": [{"name": n, "pass": o, "detail": d}
                                                  for n, o, d in final[1]]} if final else None),
    }
    (out_dir / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    lines = [f"# 长稳基线报告 {data['head']['date']}", "",
             f"- 服务器 `{url}`　时长 {args.minutes:g} 分钟 @ {args.fps:g} fps"
             f"（{'假实现' if args.fake else '真引擎'}）　分割目标 `{target}`",
             f"- 总请求 {len(rows)}"]
    if final:
        lines += ["", f"## 结论：{final[0]}", "",
                  "| 检查项 | 结果 | 说明 |", "|--------|------|------|"]
        for n, ok, d in final[1]:
            lines.append(f"| {n} | {'PASS' if ok else '**FAIL**'} | {d} |")
    if buckets:
        lines += ["", "## 采样桶（每桶约 " + f"{args.sample_sec:g}s）", "",
                  "| 桶 | 分钟 | 请求 | fps | 客户端 P50/P95 | 服务端 P50/P95 | err | 服务端 RSS | GPU |",
                  "|---:|-----:|-----:|----:|--------------:|--------------:|----:|-----------:|----:|"]
        for b in buckets:
            lines.append(
                f"| {b['bucket']} | {b['min']} | {b['n']} | {b['fps']} | "
                f"{b['c_p50']:.0f}/{b['c_p95']:.0f} ms | {b['s_p50']:.0f}/{b['s_p95']:.0f} ms | "
                f"{b['err']} | {b['rss_mb']} MB | {b['gpu_mb']} MB |")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
