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


def _fetch_timings(url: str, session_id: str) -> dict:
    """取服务端该会话**最近一帧的细粒度分段耗时**（N50）。

    数据来源是 /healthz?session=<id> —— 非契约运维接口（契约 P2.11 白名单），
    服务端在响应前把 _timings 私有字段挂到会话上，取不到就返回空 dict。
    """
    try:
        import requests
        r = requests.get(f"{url.rstrip('/')}/healthz",
                         params={"session": session_id}, timeout=5)
        data = r.json()
        return data.get("timings") or {}
    except Exception:  # noqa: BLE001  取不到不影响主流程
        return {}


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
    details: list[dict] = []
    for i in range(args.frames):
        jpg = mc.compress_image(frames[i % len(frames)])
        payload, ms, err = mc.post_frame(url, sid, jpg, f"lat_{i:03d}.jpg", args.timeout)
        lat.append(ms)
        if not err and payload.get("overlay"):
            hits += 1
        if getattr(args, "detail", False):
            details.append(_fetch_timings(url, sid))
    return {
        "modality": modality, "target": got, "frames": args.frames, "hits": hits,
        "lat": lat, "a2_ms": a2_ms, "session": sid, "details": details,
        "p50": pct(lat, 50), "p95": pct(lat, 95), "mean": statistics.fmean(lat),
        "max": max(lat), "min": min(lat),
    }


# N50：服务端分段耗时的中文说明（段名 → 报告里显示的名字）
SEG_LABEL = {
    "cls": "分类(N48)",
    "write": "写盘",
    "prune": "删旧帧",
    "models": "模型加载",
    "resolve": "取帧",
    "segment": "分割(总)",
    "seg_detect": "└YOLO检测",
    "seg_sam2_init": "└SAM2初始化",
    "seg_reset_state": "└SAM2状态重置",
    "seg_prop": "└SAM2传播",
    "seg_mask_conv": "└掩膜换算",
    "seg_render": "└渲染落盘",
    "seg_add_box": "└SAM2加框解码",
    "seg_read_img": "└引擎再读图",
    "seg_save_mask": "└mask落盘PNG",
    "seg_bbox": "└bbox计算",
    "seg_decoder": "└mask解码器",
    "seg_interp": "└插值回原分辨率",
    "seg_consolidate": "└多目标合并",
    "seg_prompt": "└prompt编码器",
    "seg_empty_cache": "└清显存缓存",
}
SEG_ORDER = ["cls", "write", "prune", "models", "resolve", "segment",
             "seg_detect", "seg_sam2_init", "seg_reset_state", "seg_prop",
             "seg_add_box", "seg_prompt", "seg_decoder", "seg_interp", "seg_consolidate",
             "seg_mask_conv", "seg_read_img", "seg_save_mask",
             "seg_bbox", "seg_render", "seg_empty_cache"]


def _agg_timings(details: list[dict]) -> dict:
    """把多帧的分段耗时聚合成"每段平均毫秒"。"""
    out: dict[str, float] = {}
    for key in SEG_ORDER:
        vals = [d[key] for d in details if d.get(key)]
        if vals:
            out[key] = statistics.fmean(vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="端到端延迟按模态统计（N8-2 / N50）")
    ap.add_argument("--frames", type=int, default=12, help="每个模态发多少帧（默认 12）")
    ap.add_argument("--url", default=mc.DEFAULT_URL, help="服务端地址")
    ap.add_argument("--timeout", type=float, default=30.0, help="单请求超时（秒）")
    ap.add_argument("--out", default="", help="输出目录（默认 runs/metrics/latency_<时间戳>）")
    ap.add_argument("--detail", action="store_true",
                    help="N50：额外抓服务端细粒度分段耗时，输出各段耗时占比表"
                         "（写盘/模型加载/检测/SAM2初始化/传播…），用于定位瓶颈")
    args = ap.parse_args()

    try:
        import requests
        hz = requests.get(f"{args.url}/healthz", timeout=3).json()
    except Exception as exc:
        log(f"[!!] 服务端不可用（{exc}）")
        return 2
    if hz.get("state") == "warming":
        # T2 配套：预热期 /healthz 是 503 + state=warming；此时量延迟会得到失真数字。
        log("[!!] 服务端正在预热（state=warming）：请等就绪后再跑（否则延迟数字失真）")
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
    # ---- N50：分段耗时占比表 ----
    if getattr(args, "detail", False) and ok:
        lines += ["", "## N50 分段耗时占比（服务端侧）", "",
                  "口径：服务端 `wait/infer/encode` 之外的**细粒度**分段，"
                  "由 `/healthz?session=<id>` 取（运维接口，非契约）；"
                  "占比 = 该段平均毫秒 ÷ 客户端往返平均毫秒。",
                  "分段名：`写盘`=落盘滚动窗口帧；`删旧帧`=prune 旧帧（Windows 上 unlink 可能触发安全扫描）；"
                  "`模型加载`=按模态取 SAM2+YOLO（换模态首帧的现场加载，预热到位应为 0）；"
                  "`取帧`=解析窗口帧；`分割(总)`=检测+SAM2+渲染（下面几行是它的细分）；"
                  "`└…`= 引擎内运行时探针（引擎源文件未改）。", "",
                  "| 模态 | 客户端往返均值 | " + " | ".join(SEG_LABEL[k] for k in SEG_ORDER) + " |",
                  "|------|------|" + "------|" * len(SEG_ORDER)]
        for r in ok:
            agg = _agg_timings(r.get("details") or [])
            cells = [f"{agg[k]:.0f} ({agg[k] / r['mean'] * 100:.0f}%)" if agg.get(k) else "—"
                     for k in SEG_ORDER]
            lines.append(f"| {r['modality']} | {r['mean']:.0f}ms | " + " | ".join(cells) + " |")
        all_details = [d for r in ok for d in (r.get("details") or [])]
        if all_details:
            agg = _agg_timings(all_details)
            all_lat = [x for r in ok for x in r["lat"]]
            mean_all = statistics.fmean(all_lat)
            cells = [f"{agg[k]:.0f} ({agg[k] / mean_all * 100:.0f}%)" if agg.get(k) else "—"
                     for k in SEG_ORDER]
            lines.append(f"| **全部合并** | **{mean_all:.0f}ms** | " + " | ".join(cells) + " |")
            top = sorted(agg.items(), key=lambda kv: -kv[1])[:3]
            lines += ["", "**占比最大的三段**："
                      + "、".join(f"{SEG_LABEL.get(k, k)} {v:.0f}ms（{v / mean_all * 100:.0f}%）"
                                  for k, v in top)]
            lines += ["", "> 优化按这个表的占比来定，**不靠猜**；"
                      "任何降精度类手段都必须附 Dice 对照（`client/eval_segmentation.py`）。"]
    lines += ["", "## 逐帧明细", "", "| 模态 | 帧 | 耗时(ms) |", "|------|----|---------|"]
    for r in ok:
        for i, ms in enumerate(r["lat"]):
            lines.append(f"| {r['modality']} | {i} | {ms:.0f} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"报告：{(out_dir / 'report.md').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
