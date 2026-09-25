#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键启动「模拟眼镜」演示（EgoMed-Agent）

干什么（不用记任何命令）：
  1) 检查环境（解释器 / 服务端 / 样例图）
  2) 后台起服务端 --real，并**等它预热完成**（约 60~90 秒，实时打印进度）
  3) 可选：自动打开一张 MRI 样例图（给摄像头拍）
  4) 起实时客户端（摄像头 / 手机网络流 / 视频样例，可选开麦克风）
  5) 你按 q 退出后，自动关闭服务端（不留僵尸进程）

用法
----
交互式（双击 启动模拟眼镜.bat）：
    python client/launch_sim_glasses.py

非交互式（自动化 / 复现）：
    python client/launch_sim_glasses.py --source video --input data/samples/acdc \
        --max-frames 6 --no-mic --fps 2 --no-show
    python client/launch_sim_glasses.py --source url --input http://192.168.1.23:8080/video
    python client/launch_sim_glasses.py --dry-run          # 只做环境检查，不启动任何东西

归属：闫（客户端 / 测试工具）｜ 关联：docs/子计划_N4_摄像头实时采集.md
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = PROJECT_ROOT / "server" / "api.py"
CLIENT_SCRIPT = PROJECT_ROOT / "client" / "live_client.py"
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "samples" / "acdc" / "frame_0001.jpg"
LOG_DIR = PROJECT_ROOT / "runs" / "launch"
SERVER_OUT = LOG_DIR / "server_out.txt"
SERVER_ERR = LOG_DIR / "server_err.txt"
URL = "http://127.0.0.1:8000"
WARMUP_TIMEOUT = 240.0        # 预热最长等待（秒）：真引擎含 ASR medium，约 60~90s
READY_MARKERS = ("真引擎启动", "Uvicorn running")


def log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # 控制台是 GBK 时，退化为可替换字符输出，避免直接崩掉
        print(msg.encode("gbk", errors="replace").decode("gbk"), flush=True)


def _init_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def check_env() -> bool:
    ok = True
    log("=" * 72)
    log("环境检查")
    log(f"  解释器：{sys.executable}")
    if not Path(sys.executable).exists():
        log("  [!!] 解释器不存在（应使用 egomed 环境的 python）")
        ok = False
    for p, name in ((SERVER_SCRIPT, "服务端 server/api.py"), (CLIENT_SCRIPT, "客户端 client/live_client.py")):
        if p.exists():
            log(f"  [OK] {name}")
        else:
            log(f"  [!!] 缺少 {name}：{p}")
            ok = False
    log(f"  {'[OK]' if SAMPLE_IMAGE.exists() else '[--]'} 样例图：{SAMPLE_IMAGE.name}"
        f"{'' if SAMPLE_IMAGE.exists() else '（缺失，不影响启动，只是无法自动打开）'}")
    try:
        import requests  # noqa: F401
        log("  [OK] 依赖 requests")
    except ImportError:
        log("  [!!] 缺少 requests（pip install requests）")
        ok = False
    log("=" * 72)
    return ok


def choose_source(args) -> tuple[str, str]:
    """返回 (source, input)；非交互模式下直接用参数"""
    if args.source:
        return args.source, args.input or ""
    log("请选择采集源：")
    log("  1) 手机网络流（IP Webcam，推荐：分辨率高、可手持移动）")
    log("  2) 本机摄像头")
    log("  3) 视频样例（data/samples/acdc，不需要硬件）")
    log("  4) 退出")
    while True:
        c = input("输入 1/2/3/4 后回车：").strip()
        if c in ("1", "2", "3", "4"):
            break
    if c == "4":
        sys.exit(0)
    if c == "1":
        url = input("粘贴手机流地址（形如 http://192.168.1.23:8080/video）：").strip()
        return "url", url
    if c == "2":
        return "cam", ""
    return "video", str(Path("data") / "samples" / "acdc")


def start_server() -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = open(SERVER_OUT, "wb")
    err = open(SERVER_ERR, "wb")
    log("[1/4] 启动服务端（真引擎 --real）…")
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT), "--real"],
        cwd=str(PROJECT_ROOT), stdout=out, stderr=err,
    )
    log(f"      服务端进程 PID={proc.pid}，日志：{SERVER_OUT.relative_to(PROJECT_ROOT)}")
    return proc


def wait_warmup(proc: subprocess.Popen, timeout: float = WARMUP_TIMEOUT) -> bool:
    """轮询日志 + /healthz，等到“真引擎启动”或超时"""
    log("[2/4] 等待预热（首次约 60~90 秒，含 ASR medium 加载）…")
    t0 = time.time()
    last = -1
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            log(f"      [!!] 服务端进程已退出（returncode={proc.returncode}），看日志：{SERVER_ERR}")
            return False
        # 日志标记
        try:
            text = SERVER_OUT.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        if any(m in text for m in READY_MARKERS):
            log(f"      [OK] 预热完成（用时 {time.time() - t0:.0f}s）")
            return True
        # 健康检查兜底（日志被缓冲时也能判断）
        try:
            import requests
            r = requests.get(f"{URL}/healthz", timeout=1.0)
            if r.status_code == 200 and r.json().get("real_engine"):
                log(f"      [OK] 预热完成（健康检查确认，用时 {time.time() - t0:.0f}s）")
                return True
        except Exception:
            pass
        spent = int(time.time() - t0)
        if spent // 10 != last:
            last = spent // 10
            log(f"      … {spent}s")
        time.sleep(2)
    log(f"      [!!] 预热超时（{timeout:.0f}s）：请查看 {SERVER_OUT}")
    return False


def open_sample() -> None:
    if not SAMPLE_IMAGE.exists():
        log(f"[3/4] 样例图缺失，跳过自动打开（可手动打开 {SAMPLE_IMAGE}）")
        return
    log(f"[3/4] 打开样例图：{SAMPLE_IMAGE.name}（用系统看图程序，尽量全屏 F11；给摄像头拍）")
    try:
        os.startfile(str(SAMPLE_IMAGE))       # noqa: S606  Windows 专用
    except Exception as exc:
        log(f"      自动打开失败（{exc}），请手动打开")


def run_client(args, source: str, src_input: str) -> int:
    cmd = [sys.executable, str(CLIENT_SCRIPT), "--source", source,
           "--fps", str(args.fps), "--url", URL]
    if src_input:
        cmd += ["--input", src_input]
    if not args.no_mic:
        cmd += ["--mic"]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.no_show:
        cmd += ["--no-show"]
    if args.audio:
        cmd += ["--audio", args.audio]
        if not args.no_expect_overlay:
            cmd += ["--expect-overlay"]
    elif not args.no_expect_overlay:
        cmd += ["--expect-overlay"]
    if args.save_frames:
        cmd += ["--save-frames"]
    log("[4/4] 启动客户端（窗口中按 q 退出；退出后会自动关闭服务端）")
    log("      命令：" + " ".join(cmd[1:]))
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        log("服务端已自行退出")
        return
    log("关闭服务端 …")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log("已关闭")


def parse_args():
    p = argparse.ArgumentParser(description="一键启动「模拟眼镜」演示")
    p.add_argument("--source", choices=["url", "cam", "video"], default="",
                   help="采集源；不给则进入交互菜单")
    p.add_argument("--input", default="", help="url 源填手机流地址；video 源填帧目录/mp4")
    p.add_argument("--fps", type=float, default=5.0, help="发帧节奏（默认 5）")
    p.add_argument("--max-frames", type=int, default=0, help="发多少帧后停止（0=手动 q 退出）")
    p.add_argument("--no-mic", action="store_true", help="不开麦克风（用 --audio 播预录 wav 时选它）")
    p.add_argument("--audio", default="", help="用预录 wav 设目标（自动化用）")
    p.add_argument("--save-frames", action="store_true", help="落盘抽样帧")
    p.add_argument("--no-show", action="store_true", help="无头模式（不弹窗）")
    p.add_argument("--no-sample", action="store_true", help="不自动打开样例图")
    p.add_argument("--no-expect-overlay", action="store_true", help="不要求必须出分割结果")
    p.add_argument("--dry-run", action="store_true", help="只做环境检查，不启动服务端/客户端")
    return p.parse_args()


def main() -> int:
    _init_stdout()
    args = parse_args()
    log("EgoMed-Agent 模拟眼镜 · 一键启动")
    if not check_env():
        return 2
    source, src_input = choose_source(args) if not args.dry_run else ("(dry-run)", "")
    log(f"采集源：{source} {src_input}")
    if args.dry_run:
        log("dry-run：环境检查通过，未启动任何进程。")
        return 0

    server = start_server()
    try:
        if not wait_warmup(server):
            return 3
        if not args.no_sample and source in ("cam", "url"):
            open_sample()
        rc = run_client(args, source, src_input)
        log(f"客户端退出（returncode={rc}）")
        return rc
    except KeyboardInterrupt:
        log("\n手动中断")
        return 130
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
