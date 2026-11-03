#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoMed ASR 显存累积诊断（只读实验，不改 server/ 任何代码）
==========================================================

为什么单独测 ASR
----------------
`scripts/diag_gpu_leak.py` 已证明「YOLO + SAM2 逐帧推理路径」连续 200 帧
allocated 恒定为 562.26MB、零累积。但长稳实测整卡从 3984MB 涨到 7514MB，
差值中约 2GB 是 ASR 侧（Whisper medium + CTranslate2 CUDA 运行时）——长稳
里每个 A2 语音请求都会真正跑一次 forward，而 faster-whisper 底层的
CTranslate2 有自己的 CUDA 内存池，行为与 torch 缓存分配器不一致。

本脚本按 server/api.py::_handle_audio_real 的真实调用方式（transcribe(str(wav))）
循环转写，逐次采样显存，判断 ASR 是否也存在累积。

用法
----
    python scripts/diag_asr_leak.py                    # 默认 40 次
    python scripts/diag_asr_leak.py --loops 100 --every 2

输出
----
    runs/diag/asr_leak_<时间戳>/samples.csv
    runs/diag/asr_leak_<时间戳>/report.md

注意：只读调用 asr_medical.transcribe，不修改服务端代码，不放宽任何阈值。
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "demo"))

import asr_medical  # noqa: E402

import psutil  # noqa: E402
import torch  # noqa: E402


def allocated_mb():
    return round(torch.cuda.memory_allocated() / 1048576, 2)


def reserved_mb():
    return round(torch.cuda.memory_reserved() / 1048576, 2)


def rss_mb():
    return round(psutil.Process(os.getpid()).memory_info().rss / 1048576, 1)


def cuda_used_mb():
    try:
        free, total = torch.cuda.mem_get_info()
        return round((total - free) / 1048576, 1)
    except Exception:
        return None


def sample(i, asr_ms):
    return {
        "loop": i,
        "alloc_mb": allocated_mb(),
        "reserved_mb": reserved_mb(),
        "rss_mb": rss_mb(),
        "cuda_used_mb": cuda_used_mb(),
        "asr_ms": round(asr_ms, 1),
    }


def force_reclaim():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return allocated_mb(), reserved_mb(), cuda_used_mb()


def slope_per_unit(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main() -> int:
    ap = argparse.ArgumentParser(description="EgoMed ASR 显存累积诊断（只读）")
    ap.add_argument("--loops", type=int, default=40, help="转写循环次数（默认 40）")
    ap.add_argument("--audio-dir", default=None,
                    help="音频目录（默认 data/samples/audio）")
    ap.add_argument("--every", type=int, default=2, help="每 N 次采样一次")
    ap.add_argument("--warmup", type=int, default=2, help="前 N 次视为预热")
    ap.add_argument("--gc-at", default="", help="在这些次数后强制回收，逗号分隔")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else REPO_ROOT / "runs" / "diag" / f"asr_leak_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    adir = Path(args.audio_dir) if args.audio_dir else REPO_ROOT / "data" / "samples" / "audio"
    wavs = sorted(p for p in adir.glob("*.wav"))
    if not wavs:
        print(f"[FAIL] 没有音频样本：{adir}")
        return 2
    print(f"[asr-diag] 音频 {len(wavs)} 个（{adir}），计划转写 {args.loops} 次")

    base_before_load = sample(0, 0.0)
    print(f"[asr-diag] 加载前 alloc={base_before_load['alloc_mb']}MB "
          f"整卡={base_before_load['cuda_used_mb']}MB RSS={base_before_load['rss_mb']}MB")

    t0 = time.perf_counter()
    warm_ok = asr_medical.warmup()
    print(f"[asr-diag] warmup 用时 {time.perf_counter() - t0:.1f}s（返回 {warm_ok}）")

    rows = [dict(base_before_load, loop=-1, asr_ms=0.0)]
    after_load = sample(0, 0.0)
    rows.append(dict(after_load, loop=0, asr_ms=0.0))
    print(f"[asr-diag] 加载后 alloc={after_load['alloc_mb']}MB "
          f"整卡={after_load['cuda_used_mb']}MB RSS={after_load['rss_mb']}MB")

    gc_at = {int(x) for x in args.gc_at.split(",") if x.strip()} or {
        max(1, args.loops // 3), max(1, 2 * args.loops // 3)}

    reclaim_log = []
    texts = []
    for i in range(1, args.loops + 1):
        wav = wavs[(i - 1) % len(wavs)]
        t1 = time.perf_counter()
        try:
            text = asr_medical.transcribe(str(wav))
        except Exception as exc:
            print(f"[WARN] 第 {i} 次异常：{type(exc).__name__}: {exc}")
            text = ""
        asr_ms = (time.perf_counter() - t1) * 1000
        texts.append(text)

        if i % args.every == 0 or i == args.loops:
            row = sample(i, asr_ms)
            rows.append(row)
            if i % 5 == 0 or i <= 3:
                print(f"[asr-diag] 第 {i:3d} 次 | alloc={row['alloc_mb']:8.2f}MB "
                      f"reserved={row['reserved_mb']:8.2f}MB "
                      f"整卡={row['cuda_used_mb']}MB RSS={row['rss_mb']}MB "
                      f"asr={row['asr_ms']:.0f}ms")

        if i in gc_at:
            a, r, c = force_reclaim()
            reclaim_log.append({"loop": i, "alloc_mb": a, "reserved_mb": r, "cuda_used_mb": c})
            print(f"[asr-diag] === 强制回收 @第 {i} 次：alloc={a}MB reserved={r}MB 整卡={c}MB ===")
            rows.append({"loop": i, "alloc_mb": a, "reserved_mb": r,
                         "rss_mb": rss_mb(), "cuda_used_mb": c,
                         "asr_ms": 0.0, "note": "after_gc"})

    # --- 分析（排除加载前基线与预热）---
    main_rows = [r for r in rows if r.get("note") != "after_gc" and r["loop"] >= args.warmup]
    if len(main_rows) < 2:
        print("[FAIL] 有效样本不足")
        return 2
    xs = [r["loop"] for r in main_rows]
    k_alloc = slope_per_unit(xs, [r["alloc_mb"] for r in main_rows])
    k_res = slope_per_unit(xs, [r["reserved_mb"] for r in main_rows])
    k_cuda = slope_per_unit(xs, [(r.get("cuda_used_mb") or 0) for r in main_rows])
    k_rss = slope_per_unit(xs, [r["rss_mb"] for r in main_rows])

    first, last = main_rows[0], main_rows[-1]
    d_alloc = last["alloc_mb"] - first["alloc_mb"]
    d_res = last["reserved_mb"] - first["reserved_mb"]
    d_cuda = (last.get("cuda_used_mb") or 0) - (first.get("cuda_used_mb") or 0)

    gc_verdict = []
    for rec in reclaim_log:
        before = [r for r in rows if r["loop"] == rec["loop"] and r.get("note") != "after_gc"]
        b = before[-1]["alloc_mb"] if before else None
        gc_verdict.append({"loop": rec["loop"], "alloc_before": b,
                           "alloc_after": rec["alloc_mb"],
                           "freed": None if b is None else round(b - rec["alloc_mb"], 2)})

    # 判定以「整卡」为准：faster-whisper 走 CTranslate2，不经过 torch 分配器，
    # torch.cuda.memory_allocated() 恒为 0，对 ASR 没有判别力。
    gc_freed = [v["freed"] for v in gc_verdict if v["freed"] is not None]
    gc_helps = bool(gc_freed) and max(gc_freed) > 0
    if k_cuda > 1.0 and not gc_helps:
        verdict = "A. ASR 真泄漏（整卡逐次上升，强制回收不回落）"
    elif k_cuda > 1.0 and gc_helps:
        verdict = "B. ASR 缓存可回收（整卡上升，但强制回收能压回）"
    elif abs(k_cuda) <= 1.0:
        verdict = "C. ASR 无累积（整卡平稳，波动属 CTranslate2 内部池）"
    else:
        verdict = "D. 趋势不明确，需更多次转写"

    empty_ratio = sum(1 for t in texts if not t) / max(len(texts), 1)

    csv_path = out_root / "samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["loop", "alloc_mb", "reserved_mb",
                                           "rss_mb", "cuda_used_mb", "asr_ms", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    lines = []
    lines.append("# EgoMed ASR 显存累积诊断报告\n")
    lines.append(f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 脚本：`scripts/diag_asr_leak.py`（只读实验，未修改 `server/`）")
    lines.append(f"- 音频：`{adir}`（{len(wavs)} 个循环使用）")
    lines.append(f"- 转写次数：{args.loops}（空结果占比 {empty_ratio*100:.0f}%）")
    lines.append(f"- 模型：faster-whisper medium / CUDA float16\n")

    lines.append("## 一、结论\n")
    lines.append(f"**{verdict}**\n")
    lines.append("| 指标 | 起始 | 结束 | 变化 | 斜率 |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| torch allocated (MB) | {first['alloc_mb']} | {last['alloc_mb']} | "
                 f"{d_alloc:+.2f} | {k_alloc:+.4f} MB/次 |")
    lines.append(f"| torch reserved (MB) | {first['reserved_mb']} | {last['reserved_mb']} | "
                 f"{d_res:+.2f} | {k_res:+.4f} MB/次 |")
    lines.append(f"| 整卡已用（CUDA 视角，MB） | {first.get('cuda_used_mb')} | "
                 f"{last.get('cuda_used_mb')} | {d_cuda:+.1f} | {k_cuda:+.4f} MB/次 |")
    lines.append(f"| 进程 RSS (MB) | {first['rss_mb']} | {last['rss_mb']} | "
                 f"{last['rss_mb'] - first['rss_mb']:+.1f} | {k_rss:+.4f} MB/次 |")
    lines.append("")
    lines.append(f"> 统计口径：跳过前 {args.warmup} 次预热；"
                 f"模型加载本身占用约 "
                 f"{(after_load.get('cuda_used_mb') or 0) - (base_before_load.get('cuda_used_mb') or 0):.0f} MB"
                 f"（一次性，不算泄漏）。")
    lines.append("")
    lines.append("> **指标说明**：faster-whisper 底层是 CTranslate2，**不经过 torch "
                 "缓存分配器**，因此 `torch.cuda.memory_allocated()` 恒为 0，"
                 "对本实验没有判别力；泄漏判定以 **整卡已用（CUDA 视角）** 为准。")
    lines.append("")

    lines.append("## 二、强制回收对照\n")
    if gc_verdict:
        lines.append("| 第几次 | 回收前 allocated | 回收后 allocated | 释放量 |")
        lines.append("|---:|---:|---:|---:|")
        for v in gc_verdict:
            lines.append(f"| {v['loop']} | {v['alloc_before']} | {v['alloc_after']} | {v['freed']} |")
        lines.append("")
    else:
        lines.append("（未执行）\n")

    lines.append("## 三、逐次采样\n")
    lines.append("| 次 | allocated | reserved | 整卡 | RSS | asr_ms | 备注 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for r in main_rows:
        lines.append(f"| {r['loop']} | {r['alloc_mb']} | {r['reserved_mb']} | "
                     f"{r.get('cuda_used_mb')} | {r['rss_mb']} | {r['asr_ms']} | "
                     f"{r.get('note', '')} |")
    lines.append("")
    lines.append(f"## 四、原始数据\n\n- `{csv_path.relative_to(REPO_ROOT)}`\n")

    report_path = out_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (out_root / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "loops": args.loops, "warmup_skipped": args.warmup,
         "slope_alloc_mb_per_loop": k_alloc, "slope_reserved_mb_per_loop": k_res,
         "slope_cuda_used_mb_per_loop": k_cuda,
         "delta_alloc_mb": d_alloc, "delta_reserved_mb": d_res,
         "delta_cuda_used_mb": d_cuda, "empty_ratio": empty_ratio,
         "gc_checks": gc_verdict}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"[asr-diag] 结论：{verdict}")
    print(f"[asr-diag] allocated {first['alloc_mb']} -> {last['alloc_mb']} MB ({d_alloc:+.2f})")
    print(f"[asr-diag] reserved  {first['reserved_mb']} -> {last['reserved_mb']} MB ({d_res:+.2f})")
    print(f"[asr-diag] 整卡      {first.get('cuda_used_mb')} -> {last.get('cuda_used_mb')} MB ({d_cuda:+.1f})")
    print(f"[asr-diag] 报告：{report_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
