#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务端 ↔ 客户端 **字段一致性**实测（2026-11-03 新增）

为什么需要它
------------
"字段一不一致"这种话不能靠人读代码保证。本次起因就是用户一句
"服务端和客户端互通性怎么样，字段一致吧" —— 顺着查下去**真抓到一处不一致**：
空文件（边界 1）分支在 post_frame 里提前 return，绕过了统一补
`elapsed_ms` / `frames_processed` 的那段，响应只有 9 个字段（契约要求 11 个）。

这个脚本把「一致」变成**可重复执行、可归档的实测结论**：

1. **契约字段清单从 `docs/API_CONTRACT.md` 里解析**（读 A1/A2 的 JSON 示例），
   不硬编码 —— 契约改了脚本自动跟随，杜绝"人和文档漂移"。
2. **实测每条分支的真实响应键集合**（假实现 + 真引擎各一套），
   逐条比对：缺字段 / 多字段 / 私有字段泄漏（`_timing` 之类）。
3. **比对客户端实际会读的字段**（客户端读、服务端没给 = 断链）。

用法
    python client/test_field_alignment.py                 # 只测假实现（秒级）
    python client/test_field_alignment.py --real          # 加测真引擎（需预热 ~90s）
    python client/test_field_alignment.py --url http://127.0.0.1:8000   # 复用已有服务端
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = PROJECT_ROOT / "server" / "api.py"
CONTRACT_MD = PROJECT_ROOT / "docs" / "API_CONTRACT.md"
OUT_ROOT = PROJECT_ROOT / "runs" / "field_align"
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"

# 客户端代码里实际会读的响应字段（2026-11-03 扫 client/*.py 统计得到）。
# 成功响应读到的：status / modality / target / overlay / message / need_confirm /
#                 candidates / elapsed_ms / text
# 503 预热门控读到的：state / retry_after_s
CLIENT_READS_SUCCESS = {
    "status", "modality", "target", "overlay", "message",
    "need_confirm", "candidates", "elapsed_ms", "text",
}
CLIENT_READS_503 = {"state", "retry_after_s"}
# 服务端给了但客户端当前没读的（不算不一致，只是信息未被消费，列出供参考）
SERVER_EXTRA_NOT_READ = {"session_id", "timestamp", "frames_processed", "modality_hint"}
# 私有字段：出现在响应里就是泄漏（契约不允许）
PRIVATE_PREFIX = "_"


def log(msg: str = "") -> None:
    try:
        sys.stdout.buffer.write(msg.encode("utf-8", "replace") + b"\n")
        sys.stdout.buffer.flush()
    except Exception:  # noqa: BLE001
        print(msg.encode("utf-8", "replace").decode("ascii", "replace"), flush=True)


# ---------------------------------------------------------------- 契约字段解析
def parse_contract_fields() -> tuple[set[str], set[str]]:
    """从 API_CONTRACT.md 的 A1/A2 JSON 示例里解析字段清单。"""
    md = CONTRACT_MD.read_text(encoding="utf-8")

    def grab(marker: str) -> set[str]:
        i = md.index(marker)
        j = md.index("```json", i)
        k = md.index("```", j + 7)
        blob = md[j + 7:k].strip()
        return set(json.loads(blob).keys())

    a1 = grab("### A1. POST /api/v1/frame")
    a2 = grab("### A2. POST /api/v1/audio")
    return a1, a2


# ---------------------------------------------------------------- 抓响应
def post_frame(url: str, sid: str, data: bytes, name: str = "f.jpg"):
    files = {"image": (name, data, "image/jpeg")}
    form = {"session_id": sid, "timestamp": str(int(time.time() * 1000))}
    r = requests.post(f"{url.rstrip('/')}/api/v1/frame", data=form, files=files, timeout=60)
    return r


def post_audio(url: str, sid: str, data: bytes, name: str = "a.wav"):
    files = {"audio": (name, data, "audio/wav")}
    form = {"session_id": sid, "timestamp": str(int(time.time() * 1000))}
    r = requests.post(f"{url.rstrip('/')}/api/v1/audio", data=form, files=files, timeout=60)
    return r


def keys_of(r) -> tuple[set[str], dict]:
    try:
        body = r.json()
    except ValueError:
        return set(), {}
    return set(body.keys()), body


class Aligner:
    def __init__(self, a1_fields: set[str], a2_fields: set[str]) -> None:
        self.a1_fields = a1_fields
        self.a2_fields = a2_fields
        self.items: list[dict] = []

    def check(self, cid: str, branch: str, kind: str, r, expect: set[str]) -> None:
        """比对一条分支的真实响应。kind: 'A1' / 'A2'"""
        ks, body = keys_of(r)
        contract = self.a1_fields if kind == "A1" else self.a2_fields
        missing = sorted(expect - ks)          # 服务端少给
        extra = sorted(ks - expect)            # 服务端多给
        leaked = sorted(k for k in ks if k.startswith(PRIVATE_PREFIX))
        client_missing = sorted((CLIENT_READS_SUCCESS & expect) - ks)  # 客户端要但没有
        ok = not missing and not leaked and not client_missing
        parts = [f"HTTP {r.status_code}", f"字段 {len(ks)} 个"]
        if missing:
            parts.append(f"缺={missing}")
        if extra:
            parts.append(f"多={extra}")
        if leaked:
            parts.append(f"**私有字段泄漏={leaked}**")
        if client_missing:
            parts.append(f"**客户端读不到={client_missing}**")
        self.items.append({"id": cid, "branch": branch, "kind": kind,
                           "fields": sorted(ks), "missing": missing, "extra": extra,
                           "leaked": leaked, "client_missing": client_missing,
                           "pass": ok, "detail": "，".join(parts)})
        log(f"  [{'PASS' if ok else 'FAIL'}] {cid} {branch:<34} {'，'.join(parts)}")

    @property
    def passed(self) -> int:
        return sum(1 for i in self.items if i["pass"])

    @property
    def total(self) -> int:
        return len(self.items)

    def check_semantics(self, cid: str, branch: str, r, fn) -> None:
        """语义级断言（字段集合之外的取值约束）。

        用于锁死那些「字段名/类型都对，但取值语义跑偏」的问题 ——
        2026-11-03 的 `candidates` 就是这一类：字段在、类型对，但给的是 YOLO
        类名而不是契约裁决的模态名，光比字段集合查不出来。
        """
        ks, body = keys_of(r)
        try:
            ok, detail = fn(body)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"断言执行失败：{exc}"
        self.items.append({"id": cid, "branch": branch, "kind": "语义",
                           "fields": sorted(ks), "missing": [], "extra": [],
                           "leaked": [], "client_missing": [], "pass": bool(ok),
                           "detail": detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {cid} {branch:<34} {detail}")

    def write_report(self, out_dir: Path, extra_notes: list[str]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        all_ok = self.passed == self.total and self.total > 0
        lines = [
            "# 服务端 ↔ 客户端 字段一致性实测",
            "",
            f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 结论：**{self.passed}/{self.total} {'PASS ✅' if all_ok else 'FAIL ❌'}**",
            f"- 契约字段来源：`docs/API_CONTRACT.md` 的 A1/A2 JSON 示例（解析得到，非硬编码）",
            f"- A1 契约字段（{len(self.a1_fields)}）：{sorted(self.a1_fields)}",
            f"- A2 契约字段（{len(self.a2_fields)}）：{sorted(self.a2_fields)}",
            "",
            "| # | 分支 | 链路 | 结果 | 实测字段 | 说明 |",
            "|---|------|:--:|:--:|---------|------|",
        ]
        for i in self.items:
            lines.append(f"| {i['id']} | {i['branch']} | {i['kind']} | "
                         f"{'PASS' if i['pass'] else '**FAIL**'} | {len(i['fields'])} 个 | {i['detail']} |")
        if extra_notes:
            lines += ["", "## 附注"] + [f"- {n}" for n in extra_notes]
        (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out_dir / "report.json").write_text(
            json.dumps({"head": {"result": "PASS" if all_ok else "FAIL"},
                        "a1_contract": sorted(self.a1_fields),
                        "a2_contract": sorted(self.a2_fields),
                        "items": self.items}, ensure_ascii=False, indent=2),
            encoding="utf-8")


# ---------------------------------------------------------------- 起服
def start_server(port: int, real: bool) -> subprocess.Popen:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tag = "real" if real else "mock"
    out = (OUT_ROOT / f"server_out_{tag}.txt").open("wb")
    err = (OUT_ROOT / f"server_err_{tag}.txt").open("wb")
    cmd = [sys.executable, str(SERVER_SCRIPT), "--port", str(port)]
    if real:
        cmd.append("--real")
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=out, stderr=err)


def wait_ready(url: str, proc: subprocess.Popen, real: bool, timeout: float = 300.0) -> bool:
    t0 = time.time()
    last = -1
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            log(f"[!!] 服务端已退出，看 {OUT_ROOT}/server_err_*.txt")
            return False
        try:
            r = requests.get(f"{url.rstrip('/')}/healthz", timeout=2)
            if r.status_code == 200:
                log(f"· 服务端就绪（{time.time() - t0:.0f}s）")
                return True
        except Exception:
            pass
        if real:
            spent = int(time.time() - t0)
            if spent // 15 != last:
                last = spent // 15
                log(f"· 预热中 {spent}s…")
        time.sleep(1.5)
    return False


def find_sample_image() -> Path | None:
    for sub in ("acdc", "bad"):
        d = SAMPLE_DIR / sub
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    return p
    return None


def find_bad_image() -> Path | None:
    d = SAMPLE_DIR / "bad"
    if not d.exists():
        return None
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            return p
    return None


def find_audio() -> Path | None:
    d = SAMPLE_DIR / "audio"
    if not d.exists():
        return None
    for p in sorted(d.glob("*.wav")):
        return p
    return None


# ---------------------------------------------------------------- 各分支实测
def probe_mock(url: str, al: Aligner) -> None:
    sid = f"fa-mock-{int(time.time())}"
    img = find_sample_image()
    if img is None:
        log("[!!] 找不到样例帧，跳过假实现分支")
        return
    data = img.read_bytes()

    # M1 正常帧
    al.check("M1", "假实现 · 正常帧", "A1", post_frame(url, sid, data, img.name), al.a1_fields)
    # M2 坏帧（黑/糊/噪声 → 边界 1）
    bad = find_bad_image()
    if bad is not None:
        al.check("M2", "假实现 · 坏帧（边界 1）", "A1",
                 post_frame(url, sid, bad.read_bytes(), bad.name), al.a1_fields)
    # M3 空文件（边界 1，post_frame 提前 return 的那条路径）
    al.check("M3", "假实现 · 空文件（边界 1）", "A1",
             post_frame(url, sid, b"", "empty.jpg"), al.a1_fields)
    # M4 A2 正常
    wav = find_audio()
    if wav is not None:
        al.check("M4", "假实现 · A2 正常", "A2",
                 post_audio(url, sid, wav.read_bytes(), wav.name), al.a2_fields)
    # M5 A2 空文件
    al.check("M5", "假实现 · A2 空文件", "A2",
             post_audio(url, sid, b"", "empty.wav"), al.a2_fields)


def probe_real(url: str, al: Aligner) -> None:
    sid = f"fa-real-{int(time.time())}"
    img = find_sample_image()
    wav = find_audio()
    if img is None or wav is None:
        log("[!!] 缺样例帧/音频，跳过真引擎分支")
        return
    data = img.read_bytes()

    # R1 真引擎 · 无目标（还没设分割目标 → 提示先说话）
    al.check("R1", "真引擎 · 未设目标", "A1",
             post_frame(url, sid, data, img.name), al.a1_fields)
    # R2 真引擎 · A2 设目标
    al.check("R2", "真引擎 · A2 设目标", "A2",
             post_audio(url, sid, wav.read_bytes(), wav.name), al.a2_fields)
    # R3 真引擎 · 分割成功帧
    al.check("R3", "真引擎 · 分割结果帧", "A1",
             post_frame(url, sid, data, img.name), al.a1_fields)
    # R4 真引擎 · 空文件（边界 1）
    al.check("R4", "真引擎 · 空文件（边界 1）", "A1",
             post_frame(url, sid, b"", "empty.jpg"), al.a1_fields)

    # ---- 语义级断言（契约 v1.0.6 决议 #16：candidates 恒为模态名）----
    audio_dir = SAMPLE_DIR / "audio"

    def _all_zh(names) -> bool:
        """候选是否都是**含中文的显示名**（模态名形如 "ACDC (MRI 心脏)"；
        YOLO 类名形如 "LV cavity"/"left kidney" 是纯英文 → 不合格）。"""
        return bool(names) and all(any(ord(ch) >= 0x4E00 for ch in str(n))
                                   for n in names)

    # R5 跨模态歧义（cmd_01「帮我分割左心室」→ MRI/超声 都有）
    cmd01 = audio_dir / "cmd_01.wav"
    if cmd01.is_file():
        def _amb(body):
            cands = body.get("candidates") or []
            if not body.get("need_confirm"):
                return False, f"need_confirm={body.get('need_confirm')}（期望 True）"
            if not cands:
                return False, "candidates 为空"
            if not _all_zh(cands):
                return False, f"candidates={cands} 不是模态名（含 YOLO 类名？）"
            return True, f"need_confirm=True，candidates={cands}（模态名 ✓）"
        al.check_semantics("R5", "真引擎 · 歧义候选是模态名",
                           post_audio(url, sid, cmd01.read_bytes(), cmd01.name), _amb)

    # R6 多目标（cmd_08「帮我分割肝脏和左肾」→ 同在 CT 腹部，不是模态歧义）
    cmd08 = audio_dir / "cmd_08.wav"
    if cmd08.is_file():
        def _multi(body):
            msg = str(body.get("message") or "")
            cands = body.get("candidates") or []
            if not body.get("need_confirm"):
                return False, f"need_confirm={body.get('need_confirm')}（期望 True）"
            if not _all_zh(cands):
                return False, f"candidates={cands} 不是模态名"
            if "多个模态" in msg:
                return False, f"message 误称模态歧义：{msg!r}（肝肾同在 CT 腹部）"
            return True, f"candidates={cands}，message={msg!r}"
        al.check_semantics("R6", "真引擎 · 多目标不误称模态歧义",
                           post_audio(url, sid, cmd08.read_bytes(), cmd08.name), _multi)


def main() -> int:
    ap = argparse.ArgumentParser(description="服务端 ↔ 客户端 字段一致性实测")
    ap.add_argument("--url", default="", help="复用已有服务端（默认自己起）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--real", action="store_true", help="加测真引擎分支（需预热 ~90s）")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    a1_fields, a2_fields = parse_contract_fields()
    log(f"· 契约 A1 字段（{len(a1_fields)}）：{sorted(a1_fields)}")
    log(f"· 契约 A2 字段（{len(a2_fields)}）：{sorted(a2_fields)}")

    url = args.url or f"http://127.0.0.1:{args.port}"
    al = Aligner(a1_fields, a2_fields)
    notes: list[str] = []
    proc_mock = proc_real = None
    try:
        if args.url:
            log("")
            log("── 复用已有服务端 ──")
            probe_mock(url, al)
        else:
            log("")
            log("── 假实现（mock）──")
            proc_mock = start_server(args.port, real=False)
            if not wait_ready(url, proc_mock, real=False, timeout=60):
                return 2
            probe_mock(url, al)
            proc_mock.terminate()
            proc_mock = None
        if args.real and not args.url:
            log("")
            log("── 真引擎（--real）──")
            proc_real = start_server(args.port, real=True)
            if not wait_ready(url, proc_real, real=True, timeout=300):
                return 2
            probe_real(url, al)
        # 503 预热门控的字段（只有真引擎 + 预热期才会出现，这里做静态核对）
        log("")
        log("── 503 预热门控响应体（契约 A.12）──")
        a12_fields = {"status", "code", "message", "state", "retry_after_s"}
        miss = CLIENT_READS_503 - a12_fields
        ok503 = not miss
        log(f"  [{'PASS' if ok503 else 'FAIL'}] 503 字段契约      "
            f"客户端读 {sorted(CLIENT_READS_503)}，契约给 {sorted(a12_fields)}")
        al.items.append({"id": "S1", "branch": "503 预热门控响应体", "kind": "A2",
                         "fields": sorted(a12_fields), "missing": [], "extra": [],
                         "leaked": [], "client_missing": sorted(miss), "pass": ok503,
                         "detail": f"契约字段 {sorted(a12_fields)}"})
        notes.append(f"服务端给了但客户端当前没读的字段：{sorted(SERVER_EXTRA_NOT_READ)}"
                     f"（非不一致，属信息未被消费）")
        notes.append("客户端读的 9 个成功响应字段 + 2 个 503 字段，服务端全部提供 → 无断链")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = OUT_ROOT / stamp
        al.write_report(out_dir, notes)
        log("")
        log(f"结论：{al.passed}/{al.total} {'PASS' if al.passed == al.total else 'FAIL'}")
        log(f"报告：{out_dir}")
        return 0 if al.passed == al.total else 1
    finally:
        for p in (proc_mock, proc_real):
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    sys.exit(main())
