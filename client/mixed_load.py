#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1/A2 混合负载复现工具（N23 用）—— 定位"A2/A1 并发偶发超时 1.7%"

干什么
------
对**同一个 session** 同时打两种请求，看超时会不会复现、超的是谁、卡在哪：

  A1 发帧（image）   —— 走 YOLO 检测 + SAM2 分割/跟踪（吃 GPU）
  A2 发语音（audio） —— 走 ASR 识别（也吃 GPU/CPU）

关键设计（否则复现不出来）
--------------------------
1. **先设目标再压测**：服务端在"没有 target"时直接返提示、不跑推理 → 压不出争抢。
   本工具先用一段预录语音把 target 设好，确认 `target` 非空才开始压测。
2. **先测单流基线，再测混合**：预热阶段单发几帧，记下 **A1 服务端自报 elapsed 的 P95**；
   混合阶段再记一次。两者一对比就能判断"A2 是否把 A1 拖慢了"（服务端慢 vs 客户端超时）。
3. **区分两种耗时**：客户端 elapsed（含网络+排队+推理）与响应里的 `elapsed_ms`（服务端自报）。

用法
----
    # 标准复现（需服务端以真引擎启动：python server/api.py --real）
    python client/mixed_load.py --frames 400 --audios 100 --threads 5

    # 只压 A1 / 只压 A2（对照组）
    python client/mixed_load.py --frames 200 --audios 0
    python client/mixed_load.py --frames 0  --audios 100

    # 多会话（查是否 session 之间互相影响）
    python client/mixed_load.py --sessions 5 --frames 200 --audios 50

    # 假服务端也能跑（只验协议，不验超时）
    python client/mixed_load.py --frames 20 --audios 5 --warmup 0

输出
----
    终端：统计表 + 判定（超时 0 才算 PASS）
    文件：runs/mixed_load/<时间戳>/report.md + report.json

归属：闫（客户端 / 测试工具）｜ 关联：docs/派工单_李_阶段四_20261008.md T1(N23)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_client as mc  # noqa: E402  复用契约常量 / post_frame / post_audio / compress_image

PROJECT_ROOT = mc.PROJECT_ROOT
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "samples" / "acdc"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "samples" / "audio"
OUT_ROOT = PROJECT_ROOT / "runs" / "mixed_load"


# ---------------------------------------------------------------- 数据准备
def load_images(image_dir: Path) -> list[bytes]:
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in mc.IMG_EXTS],
                   key=mc.natural_key)
    if not files:
        raise SystemExit(f"[错误] {image_dir} 下没有图片")
    jpgs = [mc.compress_image(p) for p in files]
    mc.log(f"  · 样例帧 {len(jpgs)} 张（来自 {image_dir.relative_to(PROJECT_ROOT)}）")
    return jpgs


def load_audios(audio_dir: Path) -> list[tuple[str, bytes]]:
    files = sorted([p for p in audio_dir.glob("*.wav")], key=mc.natural_key)
    if not files:
        raise SystemExit(f"[错误] {audio_dir} 下没有 wav")
    out = []
    for p in files:
        data = p.read_bytes()
        if len(data) > mc.AUDIO_MAX_BYTES:
            mc.log(f"  [警告] {p.name} {len(data)/1024:.0f}KB 超契约上限 "
                   f"{mc.AUDIO_MAX_BYTES/1024:.0f}KB，会被服务端拒绝")
        out.append((p.name, data))
    mc.log(f"  · 语音样例 {len(out)} 段（来自 {audio_dir.relative_to(PROJECT_ROOT)}）")
    return out


def build_jobs(n_frame: int, n_audio: int) -> list[str]:
    """把 A2 均匀插进 A1 序列（避免"先全是帧、后全是语音"的偏置）。"""
    total = n_frame + n_audio
    if total == 0:
        raise SystemExit("[错误] --frames 与 --audios 不能同时为 0")
    if n_audio == 0:
        return ["A1"] * n_frame
    if n_frame == 0:
        return ["A2"] * n_audio
    jobs, f, a = [], 0, 0
    for i in range(total):
        want_a = round((i + 1) * n_audio / total)
        if a < want_a:
            jobs.append("A2"); a += 1
        elif f < n_frame:
            jobs.append("A1"); f += 1
        else:
            jobs.append("A2"); a += 1
    return jobs


# ---------------------------------------------------------------- 请求
def one_frame(url: str, sid: str, jpg: bytes, idx: int, timeout: float) -> dict:
    payload, elapsed, err = mc.post_frame(url, sid, jpg, f"mix_{idx:05d}.jpg", timeout)
    return _pack("A1", sid, elapsed, err, payload)


def one_audio(url: str, sid: str, name: str, wav: bytes, idx: int, timeout: float) -> dict:
    payload, elapsed, err = mc.post_audio(url, sid, wav, name, timeout)
    return _pack("A2", sid, elapsed, err, payload)


def _pack(kind: str, sid: str, elapsed: float, err: str | None, payload: dict | None) -> dict:
    server_ms = None
    biz_err = None
    if isinstance(payload, dict):
        try:
            server_ms = int(payload.get("elapsed_ms"))
        except (TypeError, ValueError):
            server_ms = None
        if payload.get("status") == "error":
            biz_err = f"{payload.get('code')}: {payload.get('message')}"
    timeout_hit = bool(err) and "超时" in err
    return {
        "kind": kind, "session": sid, "elapsed_ms": round(elapsed, 1),
        "server_ms": server_ms, "ok": (err is None and biz_err is None),
        "timeout": timeout_hit, "err": err or biz_err,
    }


# ---------------------------------------------------------------- 统计
def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def stats(rows: list[dict], kind: str) -> dict:
    """kind="*" 表示全部（不分类型）。"""
    sel = rows if kind == "*" else [r for r in rows if r["kind"] == kind]
    ok = [r for r in sel if r["ok"]]
    return {
        "n": len(sel),
        "ok": len(ok),
        "timeout": sum(1 for r in sel if r["timeout"]),
        "other_err": sum(1 for r in sel if r["err"] and not r["timeout"]),
        "c_p50": pct([r["elapsed_ms"] for r in ok], 50),
        "c_p95": pct([r["elapsed_ms"] for r in ok], 95),
        "s_p50": pct([r["server_ms"] for r in ok if r["server_ms"] is not None], 50),
        "s_p95": pct([r["server_ms"] for r in ok if r["server_ms"] is not None], 95),
    }


def fmt_row(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"  {label:<6} —（未压测）"
    return (f"  {label:<6} 请求 {s['n']:<5} 成功 {s['ok']:<5} 超时 {s['timeout']:<4} "
            f"其它错 {s['other_err']:<4} | 客户端 P50/P95 {s['c_p50']:.0f}/{s['c_p95']:.0f} ms"
            f" | 服务端 P50/P95 {s['s_p50']:.0f}/{s['s_p95']:.0f} ms")


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="A1/A2 混合负载超时复现（N23）")
    ap.add_argument("--url", default=mc.DEFAULT_URL, help=f"服务器地址，默认 {mc.DEFAULT_URL}")
    ap.add_argument("--frames", type=int, default=400, help="A1 发帧请求数（默认 400）")
    ap.add_argument("--audios", type=int, default=100, help="A2 语音请求数（默认 100）")
    ap.add_argument("--threads", type=int, default=5, help="并发工作线程数（默认 5）")
    ap.add_argument("--sessions", type=int, default=1, help="会话数（默认 1=单会话混合负载）")
    ap.add_argument("--timeout", type=float, default=mc.TIMEOUT,
                    help=f"单请求超时秒数（默认 {mc.TIMEOUT}，与客户端一致）")
    ap.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="样例帧目录")
    ap.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR), help="样例语音目录")
    ap.add_argument("--warmup", type=int, default=5,
                    help="预热帧数（用于测单流基线；0=不预热）")
    ap.add_argument("--out", default="", help="输出目录（默认 runs/mixed_load/<时间戳>）")
    args = ap.parse_args()

    if not args.out:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUT_ROOT / stamp
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mc.log("=" * 78)
    mc.log(f"A1/A2 混合负载复现　服务器 {args.url}")
    mc.log(f"参数：frames={args.frames} audios={args.audios} threads={args.threads} "
           f"sessions={args.sessions} timeout={args.timeout:g}s")
    mc.log("=" * 78)

    jpgs = load_images(Path(args.image_dir))
    wavs = load_audios(Path(args.audio_dir))
    sids = [mc.make_session_id(f"mix{i}") for i in range(max(1, args.sessions))]

    # ---------- ① 设目标（没 target 服务端不跑推理，压不出争抢） ----------
    first_name, first_wav = wavs[0]
    mc.log(f"[1/3] 用 {first_name} 设置分割目标 …")
    payload, elapsed, err = mc.post_audio(args.url, sids[0], first_wav, first_name, 10.0)
    if err or not isinstance(payload, dict):
        mc.log(f"  [错误] A2 请求失败：{err}（服务端起了吗？真引擎要用 --real）")
        return 2
    target = payload.get("target") or ""
    if not target:
        mc.log(f"  [错误] 未取到 target（text={payload.get('text')!r}）；"
               f"没有目标时服务端不跑推理，复现不出超时")
        return 2
    mc.log(f"  [OK] target = {target!r}（{elapsed:.0f} ms）")

    # ---------- ② 单流基线（只发帧，慢速） ----------
    baseline: dict | None = None
    if args.warmup > 0:
        mc.log(f"[2/3] 单流基线：连发 {args.warmup} 帧（串行，记服务端自报耗时）…")
        rows = []
        for i in range(args.warmup):
            rows.append(one_frame(args.url, sids[0], jpgs[i % len(jpgs)], i, 10.0))
        baseline = stats(rows, "A1")
        mc.log(fmt_row("A1基线", baseline))

    # ---------- ③ 混合负载 ----------
    jobs = build_jobs(args.frames, args.audios)
    mc.log(f"[3/3] 混合负载：{len(jobs)} 个请求 / {args.threads} 并发"
           f"（A1 {jobs.count('A1')} + A2 {jobs.count('A2')}）…")
    t0 = time.perf_counter()

    def run_one(item: tuple[int, str]) -> dict:
        i, kind = item
        sid = sids[i % len(sids)]
        if kind == "A1":
            return one_frame(args.url, sid, jpgs[i % len(jpgs)], i, args.timeout)
        name, wav = wavs[i % len(wavs)]
        return one_audio(args.url, sid, name, wav, i, args.timeout)

    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as pool:
        rows = list(pool.map(run_one, list(enumerate(jobs))))

    wall = time.perf_counter() - t0
    s_all, s_a1, s_a2 = stats(rows, "*"), stats(rows, "A1"), stats(rows, "A2")
    timeout_total = sum(1 for r in rows if r["timeout"])
    err_total = timeout_total + sum(1 for r in rows if r["err"] and not r["timeout"])

    # ---------- 报告 ----------
    mc.log("")
    mc.log("=" * 78)
    mc.log("结果")
    mc.log("-" * 78)
    mc.log(fmt_row("全部", s_all))
    mc.log(fmt_row("A1", s_a1))
    mc.log(fmt_row("A2", s_a2))
    mc.log(f"- 墙钟 {wall:.1f}s（实际吞吐 {len(rows)/max(wall, 1e-6):.2f} 请求/s）")
    if baseline:
        drag = (s_a1["s_p95"] - baseline["s_p95"])
        mc.log(f"- 单流基线 A1 服务端 P95 {baseline['s_p95']:.0f} ms → "
               f"混合负载 {s_a1['s_p95']:.0f} ms（差 {drag:+.0f} ms）")
    rate = timeout_total / max(1, len(rows)) * 100
    mc.log("-" * 78)
    verdict = "全部通过 ✅（超时 0）" if timeout_total == 0 else f"存在超时 ❌（{rate:.2f}%）"
    mc.log(f"总判定：{verdict}")
    mc.log("=" * 78)

    head = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "url": args.url,
        "frames": args.frames, "audios": args.audios, "threads": args.threads,
        "sessions": args.sessions, "timeout_s": args.timeout, "target": target,
        "wall_s": round(wall, 1), "timeout": timeout_total, "errors": err_total,
        "result": verdict,
    }
    (out_dir / "report.json").write_text(json.dumps(
        {"head": head, "baseline_A1": baseline, "all": s_all, "A1": s_a1, "A2": s_a2,
         "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# A1/A2 混合负载复现报告 {head['date']}", "",
             f"- 服务器 `{args.url}`　参数 frames={args.frames} audios={args.audios} "
             f"threads={args.threads} sessions={args.sessions} timeout={args.timeout:g}s",
             f"- 分割目标 `{target}`　墙钟 {wall:.1f}s（{len(rows)/max(wall, 1e-6):.2f} 请求/s）",
             f"- **结论：{verdict}**", "", "## 统计", "",
             "| 组 | 请求 | 成功 | 超时 | 其它错 | 客户端 P50/P95 | 服务端 P50/P95 |",
             "|----|-----|-----|-----|-------|---------------|---------------|"]
    for label, s in (("全部", s_all), ("A1", s_a1), ("A2", s_a2)):
        if s["n"] == 0:
            continue
        lines.append(f"| {label} | {s['n']} | {s['ok']} | {s['timeout']} | {s['other_err']} | "
                     f"{s['c_p50']:.0f}/{s['c_p95']:.0f} ms | {s['s_p50']:.0f}/{s['s_p95']:.0f} ms |")
    if baseline:
        lines += ["", f"- 单流基线（A1 串行 {args.warmup} 帧）：服务端 P50/P95 "
                      f"{baseline['s_p50']:.0f}/{baseline['s_p95']:.0f} ms"]
    bad = [r for r in rows if r["err"]]
    if bad:
        lines += ["", "## 失败明细（前 20 条）", "",
                  "| 类型 | session | 客户端耗时 | 服务端耗时 | 错误 |", "|------|---------|-----------|-----------|------|"]
        for r in bad[:20]:
            lines.append(f"| {r['kind']} | {r['session']} | {r['elapsed_ms']:.0f} ms | "
                         f"{r['server_ms'] if r['server_ms'] is not None else '-'} | {r['err']} |")
    lines += ["", "## 判定提示", "",
              "1. **超时集中在 A2** → 偏「A2 超时太紧 / ASR 与推理抢资源」（见派工单 T1 候选 a、c）",
              "2. **超时集中在 A1** → 偏「A1 被 A2 挤占」（候选 b、c）",
              "3. **A1 服务端 P95 比单流基线明显升高** → GPU 争抢成立（服务端侧要排队/隔离）",
              "4. **客户端超时但服务端 elapsed 正常** → 偏排队/连接层（客户端或 uvicorn 侧）"]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    mc.log(f"报告已保存：{out_dir / 'report.md'}")
    return 0 if timeout_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
