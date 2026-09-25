#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分割精度量化：在线（服务端逐帧） vs 离线（demo 批处理） —— N8/B2

为什么要这个脚本（答辩价值最高的一段）：
    为修 N13（延迟）把服务端窗口从 10 帧改成 1 帧（放弃 SAM2 跨帧传播），
    这**可能让掩膜质量下降**。本脚本用带标注的数据集量化这个代价：
        在线（窗口=1，当前实现，逐帧 HTTP 请求）  vs  离线（整段批处理，demo 原路径）
    指标：Dice / IoU（逐帧）＋ 检出率，并给出两者的差值。

数据要求：`data/<数据集>/img/<病例>/<stem>.jpg` 与 `data/<数据集>/label/<病例>/<stem>.png`
         标签灰阶值取 `egomed_demo.MODALITIES[模态]["name_to_gray"]`（如 ACDC: myocardium=2）。

用法
----
    # 起服务端（真引擎）并等预热
    python server/api.py --real

    # 在线 + 离线对比（ACDC 病例 10 的 8 帧，目标心肌）
    python client/eval_segmentation.py --modality "ACDC (MRI 心脏)" --case 10 \
        --target myocardium --frames 8 --mode both

    # 只跑离线（不需要服务端）
    python client/eval_segmentation.py --modality "ACDC (MRI 心脏)" --case 10 \
        --target myocardium --frames 8 --mode offline

归属：闫（测试/评估工具）｜ 关联：docs/总计划.md N8、docs/联调发现_20260920_真引擎实时链路.md
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "client"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "demo"))

import cv2
import mock_client as mc          # 复用压缩/发送逻辑（在线模式走与客户端一致的链路）

STREAM_DIR = PROJECT_ROOT / "runs" / "stream"
METRICS_DIR = PROJECT_ROOT / "runs" / "metrics"

# 数据集根目录（img/ 与 label/ 所在层）
DATASET_ROOT = {
    "ACDC (MRI 心脏)": PROJECT_ROOT / "data" / "ACDC",
    "Amos (CT 腹部)": PROJECT_ROOT / "data" / "Amos",
}

# 用哪段语音把目标设进会话（服务端只能通过 A2 设目标）
TARGET_WAV = {
    "myocardium": "cmd_03.wav",
    "LV cavity": "cmd_01.wav",
    "RV cavity": "cmd_02.wav",
    "spleen": "cmd_04.wav",
    "duodenum": "cmd_05.wav",
    "polyp": "cmd_06.wav",
    "left lung": "cmd_07.wav",
    "liver": "cmd_08.wav",
}


def log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("gbk", "replace").decode("gbk"), flush=True)


# ---------------------------------------------------------------- 数据准备
def prepare_frames(modality: str, case: str, start: int, n: int, out_dir: Path):
    """把病例帧与标注复制成**数字命名**（SAM2 只认整数文件名），并返回对应标签灰阶值"""
    root = DATASET_ROOT.get(modality)
    if root is None:
        raise SystemExit(f"未配置该模态的数据集根目录：{modality}（可选：{list(DATASET_ROOT)}）")
    frames = sorted((root / "img" / case).glob("*.*"))
    if not frames:
        raise SystemExit(f"找不到图像：{root / 'img' / case}")
    picked = frames[start:start + n]
    if not picked:
        raise SystemExit(f"起始帧超出范围（共 {len(frames)} 帧）")

    fdir = out_dir / "frames"
    ldir = out_dir / "labels"
    fdir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i, img_path in enumerate(picked):
        lab_path = root / "label" / case / (img_path.stem + ".png")
        if not lab_path.exists():
            raise SystemExit(f"缺少标注：{lab_path}")
        shutil.copy(img_path, fdir / f"{i:06d}.jpg")
        shutil.copy(lab_path, ldir / f"{i:06d}.png")
        pairs.append((f"{i:06d}", img_path.stem))
    log(f"准备 {len(pairs)} 帧 → {fdir.relative_to(PROJECT_ROOT)}（数字命名，供 SAM2 读取）")
    return fdir, ldir, pairs


def find_annotated_start(modality: str, case: str, target: str, gray: int,
                         scan_limit: int = 300, min_px: int = 20,
                         conf: float = 0.3) -> int:
    """返回第一个**有该目标标注**（标注非全 0）的帧序号

    教训（N14 / 2026-09-25）：只看前几帧就评估会得到 Dice=0——
    因为数据集的许多帧本来就没有标注（无目标切面），并非算法失效。
    """
    import egomed_demo as demo

    root = DATASET_ROOT[modality]
    labs = sorted((root / "label" / case).glob("*.png"))
    imgs = sorted((root / "img" / case).glob("*.*"))
    if not labs or not imgs:
        log(f"[auto-start] 病例 {case}：数据缺失，退回 start=0")
        return 0
    _pred, yolo = demo._get_models(modality)
    names = demo.eng.load_yolo_class_names_from_model(yolo)
    for i, lp in enumerate(labs[:scan_limit]):
        lab = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        if lab is None:
            continue
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        npx = int((lab == gray).sum())
        if npx < min_px:
            continue                      # 该帧没有此目标标注
        # 标注有目标，还要确认 YOLO 能检出（否则评估必然是 0 检出，属数据/权重不匹配）
        if i < len(imgs):
            # 阈值与引擎一致（eng.DET_CONF_THRES = 0.3），否则会选到管线检不到的帧
            res = yolo.predict(str(imgs[i]), conf=conf, verbose=False)
            if not any(names.get(int(b.cls)) == target for r in res for b in r.boxes):
                continue
        log(f"[auto-start] 病例 {case}：第 {i} 帧起「有标注且可检出」（{lp.name}，"
            f"标注像素 {npx}）")
        return i
    log(f"[auto-start] 病例 {case}：前 {scan_limit} 帧未找到「有标注且可检出」的帧，退回 start=0")
    return 0


def label_gray(modality: str, target: str) -> int:
    import egomed_demo as demo
    mapping = demo.MODALITIES[modality]["name_to_gray"]
    if target not in mapping:
        raise SystemExit(f"模态「{modality}」不支持目标「{target}」；支持：{list(mapping)}")
    return int(mapping[target])


# ---------------------------------------------------------------- 指标
def load_binary(path: Path, gray: int):
    """读掩膜/标注 PNG → 二维二值布尔掩膜；文件不存在返回 None"""
    if not Path(path).exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.ndim == 3:                     # 有些 PNG（带调色板/通道）会读出 (h, w, 1)
        img = img[:, :, 0]
    return img == gray


def align(pred, gt):
    """把预测掩膜对齐到标注尺寸（在线模式的掩膜来自客户端压缩后的帧，长边 1280）"""
    if pred.shape == gt.shape:
        return pred
    p = (pred.astype(np.uint8) * 255)
    p = cv2.resize(p, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return p > 127


def dice_iou(pred, gt):
    pred = align(pred, gt)
    inter = np.logical_and(pred, gt).sum()
    if inter == 0:
        return 0.0, 0.0
    return (2.0 * inter / (pred.sum() + gt.sum()),
            inter / np.logical_or(pred, gt).sum())


def eval_masks(mask_dir: Path, ldir: Path, gray: int, pairs, tag: str):
    """逐帧对比掩膜与标注，返回统计 dict"""
    rows = []
    skipped = 0
    for name, _stem in pairs:
        gt = load_binary(ldir / f"{name}.png", gray)
        if gt is None or not gt.any():
            skipped += 1          # 该帧没有此目标标注 → 不参与评价（数据问题，非算法问题）
            continue
        pred = load_binary(mask_dir / f"{name}.png", gray)
        if pred is None:
            rows.append({"frame": name, "hit": False, "dice": 0.0, "iou": 0.0})
            continue
        d, j = dice_iou(pred, gt)
        rows.append({"frame": name, "hit": True, "dice": d, "iou": j})
    hits = [r for r in rows if r["hit"]]
    res = {
        "tag": tag,
        "skipped": skipped,
        "frames": len(rows),
        "hits": len(hits),
        "hit_rate": (len(hits) / len(rows)) if rows else 0.0,
        "dice_mean": float(np.mean([r["dice"] for r in hits])) if hits else 0.0,
        "iou_mean": float(np.mean([r["iou"] for r in hits])) if hits else 0.0,
        "dice_std": float(np.std([r["dice"] for r in hits])) if hits else 0.0,
        "dice_all_mean": float(np.mean([r["dice"] for r in rows])) if rows else 0.0,
        "rows": rows,
    }
    log(f"  [{tag}] 评价 {res['frames']} 帧（跳过无标注 {res['skipped']} 帧）："
        f"检出 {res['hits']}/{res['frames']}，"
        f"Dice(检出帧) {res['dice_mean']:.3f}±{res['dice_std']:.3f}，"
        f"IoU {res['iou_mean']:.3f}，Dice(含未检出=0) {res['dice_all_mean']:.3f}")
    return res


# ---------------------------------------------------------------- 离线模式
def run_offline(frames_dir: Path, modality: str, target: str, out_dir: Path,
                min_area_ratio: float = 0.001, min_occurrence: int = 2):
    import egomed_demo as demo

    pred, yolo = demo._get_models(modality)
    name_to_gray = demo.MODALITIES[modality]["name_to_gray"]
    id_to_name = {k: v for k, v in demo.eng.load_yolo_class_names_from_model(yolo).items()
                  if v in name_to_gray}
    class_id_to_gray = {k: name_to_gray[v] for k, v in id_to_name.items()}
    target_id = next((k for k, v in id_to_name.items() if v == target), None)
    if target_id is None:
        raise SystemExit(f"YOLO 权重里没有目标 {target}")

    image_files = sorted(frames_dir.glob("*.jpg"))
    off_out = out_dir / "offline"
    off_out.mkdir(parents=True, exist_ok=True)
    log(f"离线模式：整段批处理 {len(image_files)} 帧（min_occurrence=2，demo 原路径）…")
    t0 = time.perf_counter()
    result = demo._segment_target(
        pred, yolo, image_files, frames_dir, target_id,
        id_to_name, class_id_to_gray, off_out, target,
        min_area_ratio=min_area_ratio, min_occurrence=min_occurrence,
    )
    dt = time.perf_counter() - t0
    log(f"  完成：{result['status']}，耗时 {dt:.1f}s（{dt / max(1, len(image_files)):.2f}s/帧）")
    return off_out / "masks", dt


# ---------------------------------------------------------------- 在线模式
def run_online(frames_dir: Path, modality: str, target: str, args):
    """逐帧走 HTTP（与真实客户端一致）：A2 设目标 → 逐帧 A1 → 取服务端落盘的掩膜"""
    import requests

    url = args.url.rstrip("/")
    try:
        hz = requests.get(f"{url}/healthz", timeout=3).json()
    except Exception as exc:
        raise SystemExit(f"服务端不可用（{exc}）：请先 `python server/api.py --real` 并等预热")
    if not hz.get("real_engine"):
        raise SystemExit("服务端当前是**假实现**，请用 --real 启动后再跑在线模式")

    wav_name = TARGET_WAV.get(target)
    if not wav_name:
        raise SystemExit(f"没有可用于设置目标「{target}」的语音样例（映射表缺）")
    wav = PROJECT_ROOT / "data" / "samples" / "audio" / wav_name
    sid = mc.make_session_id("eval")
    payload, ms, err = mc.post_audio(url, sid, wav.read_bytes(), wav_name, args.timeout)
    if err or not payload.get("target"):
        raise SystemExit(f"A2 设目标失败：{err or payload.get('message')}")
    log(f"在线模式：A2 「{payload.get('text')}」→ {payload['target']}（{ms:.0f}ms）；session={sid}")

    mask_dir = STREAM_DIR / sid / "seg" / "masks"
    frames = sorted(frames_dir.glob("*.jpg"))
    latencies, masks_copied = [], 0
    for i, fp in enumerate(frames):
        jpg = mc.compress_image(fp)
        payload, ms, err = mc.post_frame(url, sid, jpg, f"{i:06d}.jpg", args.timeout)
        latencies.append(ms)
        if err:
            log(f"  帧 {i}: 请求失败 {err}")
            continue
        pngs_now = list(mask_dir.glob("*.png")) if mask_dir.exists() else []
        newest = max(pngs_now, key=lambda p: p.stat().st_mtime) if pngs_now else None
        if payload.get("overlay") and newest is not None:
            masks_copied += 1
        log(f"  帧 {i}: overlay={'有' if payload.get('overlay') else '无'} "
            f"target={payload.get('target') or '-'} {ms:.0f}ms")
    log(f"  A1 共 {len(frames)} 帧，检出 {masks_copied} 帧；延迟 均值 {np.mean(latencies):.0f}ms / "
        f"P95 {np.percentile(latencies, 95):.0f}ms")
    return mask_dir, sid, latencies


def collect_online_masks(mask_dir: Path, frames_dir: Path, out_dir: Path):
    """把服务端掩膜按帧序复制出来（服务端文件名是它自己的计数器，这里按 mtime 排序对齐）"""
    dst = out_dir / "online" / "masks"
    dst.mkdir(parents=True, exist_ok=True)
    # 服务端掩膜文件名是它自己的零填充计数器（000001.png …），按名排序即发送顺序
    pngs = sorted(mask_dir.glob("*.png"), key=lambda p: p.stem) if mask_dir.exists() else []
    n = len(sorted(frames_dir.glob("*.jpg")))
    for i, src in enumerate(pngs[:n]):
        shutil.copy(src, dst / f"{i:06d}.png")
    log(f"  收集服务端掩膜 {len(list(dst.glob('*.png')))} 张 → {dst.relative_to(PROJECT_ROOT)}")
    return dst


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="分割精度量化：在线 vs 离线（N8/B2）")
    ap.add_argument("--modality", default="ACDC (MRI 心脏)", help="模态名（见 egomed_demo.MODALITIES）")
    ap.add_argument("--case", default="10", help="病例编号（文件夹名）")
    ap.add_argument("--target", default="myocardium", help="目标类别名（YOLO 类别）")
    ap.add_argument("--frames", type=int, default=8, help="评估帧数（默认 8）")
    ap.add_argument("--start", type=int, default=0, help="起始帧序号（默认 0）")
    ap.add_argument("--auto-start", action="store_true",
                    help="自动从第一个可检出帧开始（避免评估落在无目标切面上）")
    ap.add_argument("--min-area-ratio", type=float, default=0.001,
                    help="离线模式的误检面积过滤阈值（引擎默认 0.001＝帧面积 0.1%%；"
                         "小器官需调低，如 0 关闭过滤）")
    ap.add_argument("--min-occurrence", type=int, default=2,
                    help="离线模式的时序过滤（引擎默认 2 帧）")
    ap.add_argument("--mode", choices=["both", "online", "offline"], default="both")
    ap.add_argument("--url", default=mc.DEFAULT_URL, help="服务端地址")
    ap.add_argument("--timeout", type=float, default=30.0, help="单请求超时（秒）")
    ap.add_argument("--out", default="", help="输出目录（默认 runs/metrics/<时间戳>）")
    args = ap.parse_args()

    if args.mode in ("both", "online") and args.target not in TARGET_WAV:
        raise SystemExit(f"在线模式需要一段能设置「{args.target}」的语音样例；"
                         f"请扩展 TARGET_WAV 映射或改用 --mode offline")

    out_dir = Path(args.out) if args.out else METRICS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    gray = label_gray(args.modality, args.target)
    if args.auto_start:
        args.start = find_annotated_start(args.modality, args.case, args.target, gray)
    log("=" * 78)
    log(f"分割精度评估  {args.modality} / 病例 {args.case} / 目标 {args.target} / "
        f"{args.frames} 帧（start={args.start}）  模式：{args.mode}")
    log("=" * 78)

    frames_dir, ldir, pairs = prepare_frames(args.modality, args.case, args.start, args.frames, out_dir)
    log(f"目标标签灰阶值：{args.target} = {gray}")

    results = {}
    if args.mode in ("both", "offline"):
        mask_dir, dt = run_offline(frames_dir, args.modality, args.target, out_dir,
                                   min_area_ratio=args.min_area_ratio,
                                   min_occurrence=args.min_occurrence)
        results["offline"] = eval_masks(mask_dir, ldir, gray, pairs, "离线(窗口=10,原 demo)")
        results["offline"]["seconds"] = dt
    if args.mode in ("both", "online"):
        mask_dir, sid, lat = run_online(frames_dir, args.modality, args.target, args)
        out_masks = collect_online_masks(mask_dir, frames_dir, out_dir)
        results["online"] = eval_masks(out_masks, ldir, gray, pairs, "在线(窗口=1,当前实现)")
        results["online"]["latency_mean_ms"] = float(np.mean(lat))
        results["online"]["latency_p95_ms"] = float(np.percentile(lat, 95))
        results["online"]["session_id"] = sid

    # ---- 报告
    lines = ["# 分割精度评估（在线 vs 离线）", "",
             f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 数据：`{DATASET_ROOT[args.modality].relative_to(PROJECT_ROOT)}` 病例 `{args.case}`，"
             f"帧 {args.start}~{args.start + len(pairs) - 1}（共 {len(pairs)} 帧）",
             f"- 目标：`{args.target}`（标签灰阶 {gray}）", "",
             "## 结果", "",
             "| 模式 | 检出帧 | Dice(检出帧) | IoU | Dice(含未检出=0) | 速度 |",
             "|------|-------|-------------|-----|-----------------|------|"]
    for key, label in (("offline", "离线（窗口=10，demo 原路径）"),
                       ("online", "在线（窗口=1，当前实现）")):
        r = results.get(key)
        if not r:
            continue
        speed = (f"{r['seconds'] / max(1, r['frames']):.2f}s/帧" if key == "offline"
                 else f"均值 {r.get('latency_mean_ms', 0):.0f}ms / P95 {r.get('latency_p95_ms', 0):.0f}ms")
        lines.append(f"| {label} | {r['hits']}/{r['frames']} | {r['dice_mean']:.3f}±{r['dice_std']:.3f} | "
                     f"{r['iou_mean']:.3f} | {r['dice_all_mean']:.3f} | {speed} |")
    if "offline" in results and "online" in results:
        do, dn = results["offline"], results["online"]
        lines += ["", "## 在线化的代价（在线 − 离线）", "",
                  f"- Dice 变化：**{dn['dice_mean'] - do['dice_mean']:+.3f}**"
                  f"（{do['dice_mean']:.3f} → {dn['dice_mean']:.3f}）",
                  f"- 检出率变化：{do['hit_rate']:.0%} → {dn['hit_rate']:.0%}"]
    lines += ["", "## 逐帧明细", "", "| 帧 | 检出 | Dice | IoU |", "|----|------|------|-----|"]
    for key in ("offline", "online"):
        r = results.get(key)
        if not r:
            continue
        lines.append(f"| **{key}** | | | |")
        for row in r["rows"]:
            lines.append(f"| {row['frame']} | {'是' if row['hit'] else '否'} | "
                         f"{row['dice']:.3f} | {row['iou']:.3f} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "result.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in results.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log("=" * 78)
    log(f"报告：{(out_dir / 'report.md').relative_to(PROJECT_ROOT)}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
