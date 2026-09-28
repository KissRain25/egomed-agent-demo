#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端延迟按模态统计（N8-2）

做法：对每个模态——A2 用语音设定目标 → 连发 N 帧（走与真实客户端一致的压缩/上传链路）
      → 统计 A1 的 P50/P95/均值、overlay 命中帧数、A2 耗时。
报告：runs/metrics/latency_<时间戳>/report.md（含逐帧明细）。

为什么要按模态分开统计：实测不同模态差异显著（X光 P95≈0.5s，ACDC MRI≈2.2s）；
只报一个总平均会掩盖差异，答辩时也说不清。

用法（服务端需 --real 启动并预热完成）：
    python client/eval_latency.py --frames 12
归属：闫（测试/评估工具）｜ 关联：docs/总计划.md N8、docs/指标_分割精度_在线vs离线.md
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "client"))

import mock_client as mc

SAMPLES = PROJECT_ROOT / "data" / "samples"
METRICS_DIR = PROJECT_ROOT / "runs" / "metrics"

# (模态, 目标, 设置目标用的语音, 样例帧目录)
CASES = [
    ("ACDC (MRI 心脏)", "myocardium", SAMPLES / "audio" / "cmd_03.wav", SAMPLES / "acdc"),
    ("CAMUS (超声 心脏)", "Left Atrium", SAMPLES / "audio50" / "cmd_04.wav", SAMPLES / "camus"),
    ("Amos (CT 腹部)", "spleen", SAMPLES / "audio" / "cmd_04.wav", SAMPLES / "amos"),
    ("Montgomery (X光 胸片)", "left lung", SAMPLES / "audio" / "cmd_07.wav", SAMPLES / "cxr"),
    ("PolypGen (内窥镜)", "polyp", SAMPLES / "audio" / "cmd_06.wav", SAMPLES / "polyp"),
]


def log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("gbk", "replace").decode("gbk"), flush=True)


def pct(vals, p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def run_case(modality: str, target: str, wav: Path, frames_dir: Path, args):
    import requests

    url = args.url.rstrip("/")
    sid = mc.make_session_id("lat")
    frames = sorted([f for f in frames_dir.glob("*") if f.suffix.lower() in mc.IMG_EXTS],
                    key=mc.natural_key)
    if not frames:
        return {"modality": modality, "error": f"无样例帧：{frames_dir}"}
    if not wav.is_file():
        return {"modality": modality, "error": f"缺少设目标语音：{wav}"}

    # A2 设目标
    payload, a2_ms, err = mc.post_audio(url, sid, wav.read_bytes(), wav.name, args.timeout)
    if err or not payload.get("target"):
        return {"modality": modality,
                "error": f"A2 设目标失败：{err or payload.get('message')}（期望目标 {target}）"}
    got = payload.get("target")
    if got != target:
        log(f"  [注意] 期望目标 {target}，实际 {got}（继续统计）")

    lat, hits = [], 0
    for i in range(args.frames):
        jpg = mc.compress_image(frames[i % len(frames)])
        payload, ms, err = mc.post_frame(url, sid, jpg, f"lat_{i:03d}.jpg", args.timeout)
        lat.append(ms)
        if not err and payload.get("overlay"):
            hits += 1
    return {
        "modality": modality, "target": got, "frames": args.frames, "hits": hits,
        "lat": lat, "a2_ms": a2_ms, "session": sid,
        "p50": pct(lat, 50), "p95": pct(lat, 95), "mean": statistics.fmean(lat),
        "max": max(lat), "min": min(lat),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="端到端延迟按模态统计（N8-2）")
    ap.add_argument("--frames", type=int, default=12, help="每个模态发多少帧（默认 12）")
    ap.add_argument("--url", default=mc.DEFAULT_URL, help="服务端地址")
    ap.add_argument("--timeout", type=float, default=30.0, help="单请求超时（秒）")
    ap.add_argument("--out", default="", help="输出目录（默认 runs/metrics/latency_<时间戳>）")
    args = ap.parse_args()

    try:
        import requests
        hz = requests.get(f"{args.url}/healthz", timeout=3).json()
    except Exception as exc:
        log(f"[!!] 服务端不可用（{exc}）")
        return 2
    if not hz.get("real_engine"):
        log("[!!] 服务端当前是假实现，请用 --real 启动")
        return 2

    log("=" * 78)
    log(f"端到端延迟统计：{len(CASES)} 个模态 × {args.frames} 帧　服务端 {args.url}")
    log("=" * 78)
    results = []
    for modality, target, wav, fdir in CASES:
        log(f"[{modality}] 目标 {target} ← {wav.name}")
        r = run_case(modality, target, wav, fdir, args)
        results.append(r)
        if r.get("error"):
            log(f"  [!!] {r['error']}")
        else:
            log(f"  检出 {r['hits']}/{r['frames']} 帧　A2 {r['a2_ms']:.0f}ms　"
                f"A1 均值 {r['mean']:.0f} / P50 {r['p50']:.0f} / P95 {r['p95']:.0f} ms"
                f"（min {r['min']:.0f} / max {r['max']:.0f}）")
    log("=" * 78)

    out_dir = Path(args.out) if args.out else METRICS_DIR / f"latency_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if not r.get("error")]
    lines = ["# 端到端延迟统计（按模态 · N8-2）", "",
             f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
             f"- 服务端：`{args.url}`（真引擎，已预热）｜ 每模态发帧数：{args.frames}",
             "- 链路：与真实客户端一致（`mock_client.compress_image` 压缩 → A1 multipart 上传）", "",
             "## 汇总", "",
             "| 模态 | 目标 | 检出帧 | A2 语音 | A1 均值 | P50 | **P95** | min | max |",
             "|------|------|-------|--------|--------|-----|--------|-----|-----|"]
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['modality']} | - | - | - | - | - | - | - | ❌ {r['error']} |")
            continue
        lines.append(f"| {r['modality']} | `{r['target']}` | {r['hits']}/{r['frames']} | "
                     f"{r['a2_ms']:.0f}ms | {r['mean']:.0f}ms | {r['p50']:.0f}ms | "
                     f"**{r['p95']:.0f}ms** | {r['min']:.0f} | {r['max']:.0f} |")
    if ok:
        all_lat = [x for r in ok for x in r["lat"]]
        lines += ["", "## 全部模态合并", "",
                  f"- A1 均值 {statistics.fmean(all_lat):.0f} ms　P50 {pct(all_lat, 50):.0f} ms　"
                  f"**P95 {pct(all_lat, 95):.0f} ms**　最大 {max(all_lat):.0f} ms",
                  f"- 契约性能预算参考：首次点名分割 ≤1500ms、后续跟踪帧 ≤200ms（`API_CONTRACT.md` §5.6）",
                  f"- 客户端单请求超时 3000ms：本次 {sum(1 for x in all_lat if x > 3000)} 次超时"]
    lines += ["", "## 逐帧明细", "", "| 模态 | 帧 | 耗时(ms) |", "|------|----|---------|"]
    for r in ok:
        for i, ms in enumerate(r["lat"]):
            lines.append(f"| {r['modality']} | {i} | {ms:.0f} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"报告：{(out_dir / 'report.md').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
