#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoMed 显存累积诊断（只读实验，不改 server/ 任何代码）
=======================================================

背景
----
第二轮长稳（runs/soak/20261103_000949）跑到 25 分钟时 GPU 从 3984MB 涨到
7514MB（8GB 卡的 92%），随后单帧推理从 10~16ms 退化到 218s。
本脚本用来判断这到底是：

  (A) **真泄漏**   —— torch.cuda.memory_allocated 单调上升，gc+empty_cache 不回落
  (B) **分配器碎片** —— allocated 平稳，只有 reserved 上升，empty_cache 能压回
  (C) **非本进程**   —— 本进程 allocated/reserved 都平，但整卡 nvidia-smi 在涨

做法：完全按 server/api.py::_handle_frame_real 的调用链复现
（写 1 帧 → prune 到 WINDOW_SIZE=1 → demo._segment_target），逐帧采样，
并在指定帧插入 gc.collect() + empty_cache() 做「强制回收对照」。

用法
----
    python scripts/diag_gpu_leak.py                     # 默认 200 帧，ACDC
    python scripts/diag_gpu_leak.py --frames 60          # 快速试跑
    python scripts/diag_gpu_leak.py --frames 600 --every 5

输出
----
    runs/diag/gpu_leak_<时间戳>/samples.csv     逐帧原始采样
    runs/diag/gpu_leak_<时间戳>/report.md       结论报告

注意：本脚本只读调用 demo 模块，不修改服务端代码，也不放宽任何验收阈值。
"""

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "scripts" / "demo"
sys.path.insert(0, str(DEMO_DIR))

import egomed_demo as demo  # noqa: E402  （内部会 exec 引擎模块并配置 CUDA）

import psutil  # noqa: E402
import torch  # noqa: E402

WINDOW_SIZE = 1          # 与服务端 server/api.py 的 WINDOW_SIZE 保持一致


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def allocated_mb() -> float:
    return round(torch.cuda.memory_allocated() / 1048576, 2)


def reserved_mb() -> float:
    return round(torch.cuda.memory_reserved() / 1048576, 2)


def max_allocated_mb() -> float:
    return round(torch.cuda.max_memory_allocated() / 1048576, 2)


def rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1048576, 1)


def smi_proc_mb(pid: int):
    """本进程在 nvidia-smi 里的显存占用（进程级，不是整卡）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                return float(parts[1].split()[0])
            except ValueError:
                return None
    return None


def smi_total_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        used, total = [x.strip().split()[0] for x in out.split(",")]
        return float(used), float(total)
    except Exception:
        return None, None


def cuda_used_mb():
    """CUDA 视角的整卡已用显存（total - free），比 nvidia-smi 在 WDDM 下更可靠。"""
    try:
        free, total = torch.cuda.mem_get_info()
        return round((total - free) / 1048576, 1), round(total / 1048576, 1)
    except Exception:
        return None, None


def sample(pid, frame_no, infer_ms):
    used, total = smi_total_mb()
    cuda_used, cuda_total = cuda_used_mb()
    return {
        "frame": frame_no,
        "alloc_mb": allocated_mb(),
        "reserved_mb": reserved_mb(),
        "max_alloc_mb": max_allocated_mb(),
        "rss_mb": rss_mb(),
        "smi_proc_mb": smi_proc_mb(pid),
        "smi_total_used_mb": used,
        "smi_total_mb": total,
        "cuda_used_mb": cuda_used,
        "cuda_total_mb": cuda_total,
        "infer_ms": round(infer_ms, 1),
    }


def force_reclaim():
    """强制回收对照：真泄漏不会回落，碎片会回落。"""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return allocated_mb(), reserved_mb(), smi_proc_mb(os.getpid())


# ---------------------------------------------------------------------------
# 线性拟合（最小二乘，只看趋势斜率，不做严格统计）
# ---------------------------------------------------------------------------
def slope_per_frame(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="EgoMed 显存累积诊断（只读）")
    ap.add_argument("--frames", type=int, default=200, help="循环帧数（默认 200）")
    ap.add_argument("--modality", default="ACDC", help="模态关键词（默认 ACDC）")
    ap.add_argument("--target", default="LV cavity", help="目标名（默认 LV cavity）")
    ap.add_argument("--src", default=None,
                    help="源帧目录（默认 data/ACDC/img/14）")
    ap.add_argument("--every", type=int, default=1,
                    help="每 N 帧做一次完整采样（默认 1，即每帧）")
    ap.add_argument("--warmup", type=int, default=5,
                    help="前 N 帧视为预热、不计入趋势统计（默认 5）。"
                         "首次推理会一次性分配 mask decoder 缓冲，"
                         "若把它当基线会算出一个虚假的正斜率。")
    ap.add_argument("--gc-at", default="",
                    help="在这些帧号后插入强制回收，逗号分隔；默认取 1/3 与 2/3 处")
    ap.add_argument("--out", default=None, help="输出目录")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else REPO_ROOT / "runs" / "diag" / f"gpu_leak_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    frames_dir = out_root / "session" / "frames"
    seg_dir = out_root / "session" / "seg"
    frames_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(args.src) if args.src else REPO_ROOT / "data" / "ACDC" / "img" / "14"
    src_files = demo.eng.collect_image_files(src_dir)
    if not src_files:
        print(f"[FAIL] 源帧目录没有图片：{src_dir}")
        return 2

    modality, name_to_gray = demo.resolve_modality(args.target, args.modality)
    if modality is None:
        print(f"[FAIL] 目标「{args.target}」在模态「{args.modality}」里找不到")
        return 2

    print(f"[diag] 模态={modality} 目标={args.target}")
    print(f"[diag] 源帧={src_dir}（{len(src_files)} 帧，循环复用）")
    print(f"[diag] 计划 {args.frames} 帧，窗口 WINDOW_SIZE={WINDOW_SIZE}")

    # --- 模型加载（一次性，与服务端一致的进程内缓存）---
    t0 = time.perf_counter()
    predictor, yolo_model = demo._get_models(modality)
    id_to_name = {
        k: v for k, v in demo.eng.load_yolo_class_names_from_model(yolo_model).items()
        if v in name_to_gray
    }
    class_id_to_gray = {k: name_to_gray[v] for k, v in id_to_name.items()}
    target_id = next((k for k, v in id_to_name.items() if v == args.target), None)
    if target_id is None:
        print(f"[FAIL] YOLO 权重里没有目标「{args.target}」")
        return 2
    print(f"[diag] 模型加载 {time.perf_counter() - t0:.1f}s，target_id={target_id}")

    pid = os.getpid()
    # 基线：模型加载完成后、进入循环前
    base = sample(pid, 0, 0.0)
    print(f"[diag] 基线 alloc={base['alloc_mb']}MB reserved={base['reserved_mb']}MB "
          f"RSS={base['rss_mb']}MB smi_proc={base['smi_proc_mb']}MB")

    gc_at = set()
    if args.gc_at.strip():
        gc_at = {int(x) for x in args.gc_at.split(",") if x.strip()}
    else:
        gc_at = {max(1, args.frames // 3), max(1, 2 * args.frames // 3)}

    rows = [dict(base, note="baseline")]
    reclaim_log = []
    t_start = time.perf_counter()
    ok_frames = 0
    miss_frames = 0

    for k in range(1, args.frames + 1):
        img_path = src_files[(k - 1) % len(src_files)]
        # --- 与服务端一致：写 1 帧进滚动窗口，只保留最近 1 帧 ---
        dst = frames_dir / f"{k:06d}.jpg"
        dst.write_bytes(Path(img_path).read_bytes())
        olds = sorted(frames_dir.glob("*.jpg"), key=lambda p: p.stem)[:-WINDOW_SIZE]
        for f in olds:
            try:
                f.unlink(missing_ok=True)
            except (Exception, SystemExit):
                pass

        image_files, _ = demo.resolve_frames(frames_dir, seg_dir)
        t1 = time.perf_counter()
        try:
            result = demo._segment_target(
                predictor, yolo_model, image_files, frames_dir, target_id,
                id_to_name, class_id_to_gray, seg_dir, args.target,
                min_occurrence=1,
                min_area_ratio=0.001,
            )
            status = result.get("status", "")
        except Exception as exc:  # 单帧失败不该打断实验
            print(f"[WARN] 第 {k} 帧异常：{type(exc).__name__}: {exc}")
            status = f"EXC:{type(exc).__name__}"
        infer_ms = (time.perf_counter() - t1) * 1000

        if status.startswith("完成"):
            ok_frames += 1
        else:
            miss_frames += 1

        if k % args.every == 0 or k == args.frames:
            row = sample(pid, k, infer_ms)
            row["note"] = "ok" if status.startswith("完成") else status[:40]
            rows.append(row)
            if k % 10 == 0 or k <= 3:
                print(f"[diag] 帧 {k:4d} | alloc={row['alloc_mb']:8.2f}MB "
                      f"reserved={row['reserved_mb']:8.2f}MB "
                      f"RSS={row['rss_mb']:8.1f}MB smi={row['smi_proc_mb']}MB "
                      f"infer={row['infer_ms']:.0f}ms")

        if k in gc_at:
            a, r, s = force_reclaim()
            reclaim_log.append({"frame": k, "alloc_mb": a,
                                "reserved_mb": r, "smi_proc_mb": s})
            print(f"[diag] === 强制回收 @帧 {k}：alloc={a}MB reserved={r}MB smi={s}MB ===")
            rows.append({"frame": k, "alloc_mb": a, "reserved_mb": r,
                         "max_alloc_mb": 0.0, "rss_mb": rss_mb(),
                         "smi_proc_mb": s, "smi_total_used_mb": None,
                         "smi_total_mb": None, "infer_ms": 0.0,
                         "note": "after_gc"})

    elapsed = time.perf_counter() - t_start

    # --- CSV ---
    csv_path = out_root / "samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- 趋势分析（关键：排除预热帧）---
    # 首次推理会一次性分配 mask decoder 中间缓冲（本机实测 +111MB），若把它
    # 算作起点会得到一个虚假的正斜率 —— 这正是长稳报告「首桶基线偏低」的同
    # 一类错误。基线必须取预热之后的稳定点。
    all_rows = [r for r in rows if r["note"] != "after_gc"]
    main_rows = [r for r in all_rows if r["frame"] >= args.warmup]
    if len(main_rows) < 2:
        print(f"[FAIL] 有效帧数不足（frames={args.frames}, warmup={args.warmup}）")
        return 2
    xs = [r["frame"] for r in main_rows]
    k_alloc = slope_per_frame(xs, [r["alloc_mb"] for r in main_rows])
    k_res = slope_per_frame(xs, [r["reserved_mb"] for r in main_rows])
    k_rss = slope_per_frame(xs, [r["rss_mb"] for r in main_rows])
    k_cuda = slope_per_frame(xs, [r["cuda_used_mb"] or 0.0 for r in main_rows])

    first, last = main_rows[0], main_rows[-1]
    d_alloc = last["alloc_mb"] - first["alloc_mb"]
    d_res = last["reserved_mb"] - first["reserved_mb"]
    d_rss = last["rss_mb"] - first["rss_mb"]
    d_cuda = (last["cuda_used_mb"] or 0) - (first["cuda_used_mb"] or 0)

    # 预热开销（帧 0 基线 -> 预热完成后），单独列出、不参与泄漏判定
    warm_cost = None
    if all_rows and main_rows:
        warm_cost = round(main_rows[0]["alloc_mb"] - all_rows[0]["alloc_mb"], 2)

    # gc 对照：看回收点后的 alloc 是否回到回收前水平
    gc_verdict = []
    for rec in reclaim_log:
        before = [r for r in rows
                  if r["frame"] == rec["frame"] and r["note"] != "after_gc"]
        b = before[0]["alloc_mb"] if before else None
        gc_verdict.append({"frame": rec["frame"], "alloc_before": b,
                           "alloc_after": rec["alloc_mb"],
                           "freed": None if b is None else round(b - rec["alloc_mb"], 2)})

    # --- 判定 ---
    if k_alloc > 0.5 and all((v["freed"] or 0) < 0.5 * k_alloc * args.frames
                             for v in gc_verdict):
        verdict = "A. 真泄漏（allocated 逐帧上升，强制回收也不回落）"
    elif abs(k_alloc) <= 0.5 and k_res > 0.5:
        verdict = "B. 分配器碎片/缓存（allocated 平稳，仅 reserved 上升）"
    elif abs(k_alloc) <= 0.5 and abs(k_res) <= 0.5 and abs(k_cuda) <= 1.0:
        verdict = "C. 本进程无累积（allocated / reserved / 整卡 均平稳）"
    else:
        verdict = "D. 趋势不明确，需更多帧或更长观察窗"

    # --- 报告 ---
    lines = []
    lines.append("# EgoMed 显存累积诊断报告\n")
    lines.append(f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 脚本：`scripts/diag_gpu_leak.py`（只读实验，未修改 `server/`）")
    lines.append(f"- 模态 / 目标：{modality} / {args.target}")
    lines.append(f"- 源帧：`{src_dir}`（{len(src_files)} 帧循环复用）")
    lines.append(f"- 循环帧数：{args.frames}（成功 {ok_frames}，未命中 {miss_frames}）")
    lines.append(f"- 窗口：WINDOW_SIZE={WINDOW_SIZE}（与服务端一致）")
    lines.append(f"- 总耗时：{elapsed:.1f}s（平均 {elapsed / max(args.frames,1)*1000:.0f} ms/帧）")
    lines.append(f"- GPU：{torch.cuda.get_device_name(0)}\n")

    lines.append("## 一、结论\n")
    lines.append(f"**{verdict}**\n")
    lines.append("| 指标 | 起始 | 结束 | 变化 | 斜率 |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| torch allocated (MB) | {first['alloc_mb']} | {last['alloc_mb']} | "
                 f"{d_alloc:+.2f} | {k_alloc:+.4f} MB/帧 |")
    lines.append(f"| torch reserved (MB) | {first['reserved_mb']} | {last['reserved_mb']} | "
                 f"{d_res:+.2f} | {k_res:+.4f} MB/帧 |")
    lines.append(f"| 进程 RSS (MB) | {first['rss_mb']} | {last['rss_mb']} | "
                 f"{d_rss:+.1f} | {k_rss:+.4f} MB/帧 |")
    lines.append(f"| 整卡已用（CUDA 视角，MB） | {first['cuda_used_mb']} | {last['cuda_used_mb']} | "
                 f"{d_cuda:+.1f} | {k_cuda:+.4f} MB/帧 |")
    lines.append("")
    lines.append(f"> **统计口径**：跳过前 {args.warmup} 帧预热（首次推理一次性开销 "
                 f"{warm_cost} MB，不参与泄漏判定）；统计区间为帧 "
                 f"{first['frame']} ~ {last['frame']}，共 {len(main_rows)} 个采样点。")
    lines.append("")

    lines.append("## 二、强制回收对照（gc.collect + empty_cache）\n")
    if gc_verdict:
        lines.append("| 帧 | 回收前 allocated | 回收后 allocated | 释放量 |")
        lines.append("|---:|---:|---:|---:|")
        for v in gc_verdict:
            lines.append(f"| {v['frame']} | {v['alloc_before']} | {v['alloc_after']} | {v['freed']} |")
        lines.append("")
        lines.append("> 判读：**释放量接近 0** = 张量仍被引用（真泄漏）；"
                     "**释放量明显** = 属于缓存/碎片，可回收。\n")
    else:
        lines.append("（未执行强制回收）\n")

    lines.append("## 三、逐帧采样（每 %d 帧）\n" % args.every)
    lines.append("| 帧 | allocated | reserved | RSS | smi本进程 | infer | 备注 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for r in main_rows:
        lines.append(f"| {r['frame']} | {r['alloc_mb']} | {r['reserved_mb']} | "
                     f"{r['rss_mb']} | {r['smi_proc_mb']} | {r['infer_ms']} | {r['note']} |")
    lines.append("")

    lines.append("## 四、原始数据\n")
    lines.append(f"- `{csv_path.relative_to(REPO_ROOT)}`")
    lines.append("")

    report_path = out_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"[diag] 结论：{verdict}")
    print(f"[diag] allocated {first['alloc_mb']} -> {last['alloc_mb']} MB "
          f"({d_alloc:+.2f}, {k_alloc:+.4f} MB/帧)")
    print(f"[diag] reserved  {first['reserved_mb']} -> {last['reserved_mb']} MB "
          f"({d_res:+.2f})")
    print(f"[diag] 整卡CUDA  {first['cuda_used_mb']} -> {last['cuda_used_mb']} MB ({d_cuda:+.1f})")
    print(f"[diag] 预热一次性开销 {warm_cost} MB（已排除在判定外）")
    print(f"[diag] 报告：{report_path}")
    print(f"[diag] 采样：{csv_path}")
    print("=" * 70)

    (out_root / "verdict.json").write_text(
        json.dumps({"verdict": verdict, "frames": args.frames,
                    "warmup_skipped": args.warmup,
                    "warmup_once_mb": warm_cost,
                    "slope_alloc_mb_per_frame": k_alloc,
                    "slope_reserved_mb_per_frame": k_res,
                    "slope_cuda_used_mb_per_frame": k_cuda,
                    "delta_alloc_mb": d_alloc, "delta_reserved_mb": d_res,
                    "delta_cuda_used_mb": d_cuda,
                    "gc_checks": gc_verdict}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
