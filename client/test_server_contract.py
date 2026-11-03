#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务端契约一致性检查（2026-11-03 新增）

为什么需要它
------------
此前服务端**错误分支零覆盖**：验收套件只打「正常帧 / 坏帧 / 语音」三条路，
1001（参数缺失）、2001（画面不可用）、5000（内部错误）、503（ASR 预热中）
这些分支**没有任何脚本触达** —— 契约写没写、代码实现成什么样，全靠人读代码。

2026-11-03 的服务端修复（ASR 异常被吞成 200、2001 出现在响应 JSON、
预热失败仍报 ready、门控 fail-open 架空 A.12……）都是靠人工读代码发现的。
这个脚本把这些分支固化成**可重复执行的护栏**：以后再改服务端，跑一遍就知道
有没有破坏契约。

覆盖（假实现即可，秒级起服）
    K1  A1 缺 session_id        → 400 + code 1001
    K2  A1 缺 timestamp        → 400 + code 1001
    K3  A1 timestamp 非整数    → 400 + code 1001
    K4  A1 缺 image 文件       → 400 + code 1001
    K5  A1 空文件              → 200 + status ok + 三字段全空 + 边界 1 文案
                                （**响应 JSON 不得出现 2001**，契约决议 #13）
    K6  A1 超大文件（>2MB）    → 400 + code 1001
    K7  A1 正常帧              → 200 + A1 字段齐全
    K8  A2 缺 session_id       → 400 + code 1001
    K9  A2 缺 audio 文件       → 400 + code 1001
    K10 A2 正常（假实现）      → 200 + A2 字段齐全
    K11 /healthz               → 200 + state / real_engine 存在
    K12 未知路径               → 404

用法
    python client/test_server_contract.py                              # 自己起服，跑完关掉
    python client/test_server_contract.py --url http://127.0.0.1:8000   # 复用已有服务端
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = PROJECT_ROOT / "server" / "api.py"
OUT_ROOT = PROJECT_ROOT / "runs" / "contract"
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"

# 契约 A1 边界 1 的统一文案（决议 #15）
BOUNDARY1_MESSAGE = "画面里没检测到目标，请对准屏幕"
# A1 响应必须齐的字段（契约 §4）
A1_FIELDS = ("status", "session_id", "timestamp", "modality", "target", "overlay",
             "message", "need_confirm", "candidates", "elapsed_ms", "frames_processed")
# A2 响应必须齐的字段（契约 §4）
A2_FIELDS = ("status", "session_id", "text", "target", "modality_hint",
             "need_confirm", "candidates", "message")


def log(msg: str = "") -> None:
    """写 stdout。

    注意：本机控制台默认 GBK，直接 print 中文/emoji 会抛 UnicodeEncodeError
    （其它脚本用 try/except + ascii 兜底，但中文照样乱码）。
    这里直接写字节流（UTF-8），由 IDE / 终端自己按 UTF-8 解码 —— 中文与 emoji 都不丢。
    """
    try:
        sys.stdout.buffer.write(msg.encode("utf-8", "replace") + b"\n")
        sys.stdout.buffer.flush()
    except Exception:  # noqa: BLE001  极端情况下退回 print
        print(msg.encode("utf-8", "replace").decode("ascii", "replace"), flush=True)


class Checker:
    """收集用例结果并生成报告。"""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, cid: str, name: str, ok: bool, detail: str, ms: float = 0.0) -> None:
        self.items.append({"id": cid, "name": name, "pass": bool(ok),
                           "detail": detail, "ms": round(ms, 1)})
        log(f"  [{'PASS' if ok else 'FAIL'}] {cid} {name:<28} {detail}")

    @property
    def passed(self) -> int:
        return sum(1 for i in self.items if i["pass"])

    @property
    def total(self) -> int:
        return len(self.items)

    def write_report(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        all_ok = self.passed == self.total and self.total > 0
        lines = [
            "# 服务端契约一致性检查",
            "",
            f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 结论：**{self.passed}/{self.total} {'PASS ✅' if all_ok else 'FAIL ❌'}**",
            "",
            "| # | 用例 | 结果 | 耗时 | 说明 |",
            "|---|------|:--:|-----:|------|",
        ]
        for i in self.items:
            lines.append(f"| {i['id']} | {i['name']} | {'PASS' if i['pass'] else '**FAIL**'} "
                         f"| {i['ms']:.0f} ms | {i['detail']} |")
        (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out_dir / "report.json").write_text(
            json.dumps({"head": {"result": "PASS" if all_ok else "FAIL"},
                        "items": self.items}, ensure_ascii=False, indent=2),
            encoding="utf-8")


# ---------------------------------------------------------------- HTTP 小工具
def post_frame_raw(url: str, session_id: str | None, timestamp: str | None,
                   image: tuple | None, timeout: float = 10.0):
    """直接打 A1，不做任何自动补全 —— 就是要构造"缺字段"这类非法请求。"""
    data = {}
    if session_id is not None:
        data["session_id"] = session_id
    if timestamp is not None:
        data["timestamp"] = timestamp
    files = {}
    if image is not None:
        files["image"] = image
    t0 = time.perf_counter()
    r = requests.post(f"{url.rstrip('/')}/api/v1/frame", data=data, files=files, timeout=timeout)
    return r, (time.perf_counter() - t0) * 1000


def post_audio_raw(url: str, session_id: str | None, timestamp: str | None,
                   audio: tuple | None, timeout: float = 10.0):
    data = {}
    if session_id is not None:
        data["session_id"] = session_id
    if timestamp is not None:
        data["timestamp"] = timestamp
    files = {}
    if audio is not None:
        files["audio"] = audio
    t0 = time.perf_counter()
    r = requests.post(f"{url.rstrip('/')}/api/v1/audio", data=data, files=files, timeout=timeout)
    return r, (time.perf_counter() - t0) * 1000


def now_ts() -> str:
    return str(int(time.time() * 1000))


def jbody(r):
    try:
        return r.json()
    except ValueError:
        return {}


def find_sample_image() -> Path | None:
    """找一张可用的正常样例帧（mock 套件同源）。"""
    for sub in ("acdc", "bad"):
        d = SAMPLE_DIR / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                return p
    return None


# ---------------------------------------------------------------- 用例
def run_checks(url: str, ck: Checker) -> None:
    sid = f"doc-contract-{int(time.time())}"
    ts = now_ts()

    # K1 缺 session_id
    r, ms = post_frame_raw(url, None, ts, ("a.jpg", b"\xff\xd8\xff", "image/jpeg"))
    b = jbody(r)
    ck.add("K1", "A1 缺 session_id", r.status_code == 400 and b.get("code") == 1001,
           f"HTTP {r.status_code} code={b.get('code')}（期望 400/1001）", ms)

    # K2 缺 timestamp
    r, ms = post_frame_raw(url, sid, None, ("a.jpg", b"\xff\xd8\xff", "image/jpeg"))
    b = jbody(r)
    ck.add("K2", "A1 缺 timestamp", r.status_code == 400 and b.get("code") == 1001,
           f"HTTP {r.status_code} code={b.get('code')}（期望 400/1001）", ms)

    # K3 timestamp 非整数
    r, ms = post_frame_raw(url, sid, "not-a-number", ("a.jpg", b"\xff\xd8\xff", "image/jpeg"))
    b = jbody(r)
    ck.add("K3", "A1 timestamp 非整数", r.status_code == 400 and b.get("code") == 1001,
           f"HTTP {r.status_code} code={b.get('code')}（期望 400/1001）", ms)

    # K4 缺 image 文件
    r, ms = post_frame_raw(url, sid, ts, None)
    b = jbody(r)
    ck.add("K4", "A1 缺 image 文件", r.status_code == 400 and b.get("code") == 1001,
           f"HTTP {r.status_code} code={b.get('code')}（期望 400/1001）", ms)

    # K5 空文件 → 契约 A1 边界 1（status ok + 空字段 + 统一文案；响应不得出现 2001）
    r, ms = post_frame_raw(url, sid, ts, ("empty.jpg", b"", "image/jpeg"))
    b = jbody(r)
    ok5 = (r.status_code == 200 and b.get("status") == "ok"
           and b.get("modality") == "" and b.get("target") == "" and b.get("overlay") == ""
           and b.get("message") == BOUNDARY1_MESSAGE
           and "2001" not in json.dumps(b, ensure_ascii=False))
    ck.add("K5", "A1 空文件 → 边界 1", ok5,
           f"HTTP {r.status_code} status={b.get('status')} message={b.get('message')!r} "
           f"（期望 200/ok/空字段/统一文案，且不含 2001）", ms)

    # K6 超大文件（>2MB 硬上限）
    r, ms = post_frame_raw(url, sid, ts, ("big.jpg", b"\x00" * (2 * 1024 * 1024 + 1024),
                                          "image/jpeg"))
    b = jbody(r)
    ck.add("K6", "A1 超大文件被拒", r.status_code == 400 and b.get("code") == 1001,
           f"HTTP {r.status_code} code={b.get('code')}（期望 400/1001）", ms)

    # K7 正常帧 → A1 字段齐全
    sample = find_sample_image()
    if sample is None:
        ck.add("K7", "A1 正常帧字段齐全", False, f"找不到样例帧（{SAMPLE_DIR} 下没有可用图片）")
    else:
        data = sample.read_bytes()
        r, ms = post_frame_raw(url, sid, ts, (sample.name, data, "image/jpeg"))
        b = jbody(r)
        missing = [f for f in A1_FIELDS if f not in b]
        ck.add("K7", "A1 正常帧字段齐全",
               r.status_code == 200 and b.get("status") == "ok" and not missing,
               f"HTTP {r.status_code} status={b.get('status')} "
               f"缺字段={missing or '无'}", ms)

    # K8 A2 缺 session_id
    r, ms = post_audio_raw(url, None, ts, ("a.wav", b"RIFF", "audio/wav"))
    b = jbody(r)
    ck.add("K8", "A2 缺 session_id", r.status_code == 400 and b.get("code") == 1001,
           f"HTTP {r.status_code} code={b.get('code')}（期望 400/1001）", ms)

    # K9 A2 缺 audio 文件
    r, ms = post_audio_raw(url, sid, ts, None)
    b = jbody(r)
    ck.add("K9", "A2 缺 audio 文件", r.status_code == 400 and b.get("code") == 1001,
           f"HTTP {r.status_code} code={b.get('code')}（期望 400/1001）", ms)

    # K10 A2 正常（假实现固定返回）
    r, ms = post_audio_raw(url, sid, ts, ("a.wav", b"RIFF0000WAVEfmt ", "audio/wav"))
    b = jbody(r)
    missing = [f for f in A2_FIELDS if f not in b]
    ck.add("K10", "A2 正常响应字段齐全",
           r.status_code == 200 and b.get("status") == "ok" and not missing,
           f"HTTP {r.status_code} status={b.get('status')} 缺字段={missing or '无'}", ms)

    # K11 /healthz
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{url.rstrip('/')}/healthz", timeout=5)
        b = jbody(r)
        ms = (time.perf_counter() - t0) * 1000
        ck.add("K11", "/healthz 探活",
               r.status_code == 200 and "state" in b and "real_engine" in b,
               f"HTTP {r.status_code} state={b.get('state')} "
               f"real_engine={b.get('real_engine')}", ms)
    except Exception as exc:
        ck.add("K11", "/healthz 探活", False, f"请求失败：{exc}")

    # K12 未知路径 → 404
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{url.rstrip('/')}/api/v1/not-exist", timeout=5)
        ms = (time.perf_counter() - t0) * 1000
        ck.add("K12", "未知路径 404", r.status_code == 404,
               f"HTTP {r.status_code}（期望 404）", ms)
    except Exception as exc:
        ck.add("K12", "未知路径 404", False, f"请求失败：{exc}")


# ---------------------------------------------------------------- 起服
def start_server(port: int) -> subprocess.Popen:
    out = (OUT_ROOT / "server_out.txt").open("wb")
    err = (OUT_ROOT / "server_err.txt").open("wb")
    cmd = [sys.executable, str(SERVER_SCRIPT), "--port", str(port)]
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=out, stderr=err)


def wait_ready(url: str, proc: subprocess.Popen, timeout: float = 40.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(f"{url.rstrip('/')}/healthz", timeout=1.5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="服务端契约一致性检查")
    ap.add_argument("--url", default="", help="复用已有服务端（默认自己起一个假实现）")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    url = args.url or f"http://127.0.0.1:{args.port}"
    proc = None
    if not args.url:
        log("· 启动假实现服务端…")
        proc = start_server(args.port)
        if not wait_ready(url, proc):
            log("[!!] 服务端未就绪，看 runs/contract/server_err.txt")
            if proc.poll() is None:
                proc.terminate()
            return 2
    try:
        ck = Checker()
        log("")
        log("── 服务端契约一致性检查 ──")
        run_checks(url, ck)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = OUT_ROOT / stamp
        ck.write_report(out_dir)
        log("")
        log(f"结论：{ck.passed}/{ck.total} {'PASS' if ck.passed == ck.total else 'FAIL'}")
        log(f"报告：{out_dir}")
        return 0 if ck.passed == ck.total else 1
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log("· 服务端已关闭")


if __name__ == "__main__":
    sys.exit(main())
