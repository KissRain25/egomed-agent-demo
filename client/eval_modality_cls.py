#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""帧级模态分类评测（N48）—— 五模态判对率 / 混淆矩阵 / 置信度分布 / 建议阈值

为什么要有这个脚本
------------------
契约 §5.3 要求"每帧分类 + 连续 3 帧一致才切"，但服务端的 `modality` 一直是从
**目标反推**出来的（`resolve_modality`），帧级分类一直没做（附录 A.10 P1.5）。
接上分类器之后，"判得准不准"就成了**必须用数字回答**的问题：
置信度阈值定多少才既不误报、又不频繁"判不出"？只能实测。

抽样口径（重要）
----------------
从**真实测试集病例**抽帧：`data/<模态>/img/<病例>/*.jpg`
（ACDC 72 / Amos 273 / CAMUS 80 / Montgomery 150 / PolypGen 97 病例）。
**跨病例等间隔抽样**，不取"前 N 帧"—— 同一病例的相邻帧高度相关，
取前 N 帧会把准确率虚高（第一版脚本就踩过：只跑 `data/samples` 样例帧 → 100%，
那是训练分布内的样本，不作数）。

输出：runs/metrics/modality_cls_<时间戳>/report.md + report.json

用法：
    python client/eval_modality_cls.py --frames 20        # 验收口径（五模态各 20 帧）
    python client/eval_modality_cls.py --frames 100       # 更稳的样本量
    python client/eval_modality_cls.py --thresholds 0,0.5,0.8,0.9,0.95

归属：闫（测试/评估工具）｜ 关联：docs/总计划.md N48、docs/API_CONTRACT.md §5.3 / A.10 P1.5
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "client"))
sys.path.insert(0, str(PROJECT_ROOT / "server"))

import modality_classifier as mc_cls  # noqa: E402

METRICS_DIR = PROJECT_ROOT / "runs" / "metrics"

# data 下的目录名 → 服务端模态键（与 demo.MODALITIES 的键一致）
MODALITY_DIRS: dict[str, str] = {
    "ACDC": "ACDC (MRI 心脏)",
    "CAMUS": "CAMUS (超声 心脏)",
    "Amos": "Amos (CT 腹部)",
    "Montgomery-County-CXR-Set": "Montgomery (X光 胸片)",
    "PolypGen2021_MultiCenterData_v3": "PolypGen (内窥镜)",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("gbk", "replace").decode("gbk"), flush=True)


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def pick_frames(modality_dir: Path, n: int, by_case: bool = False) -> list[Path]:
    """跨病例等间隔抽 n 帧（确定性、可复现，不受文件系统顺序影响）。

    `by_case=True` 时改成**每病例取 1 帧**（取中位帧）——
    覆盖的**病例数**最大化，用来测"有多少病例整体不可判"（蒙太奇实测 150 病例中
    28 个是"手机拍屏幕"的实拍照片，分类器对这批**系统性误判**，单看帧级抽样会低估）。
    """
    cases = sorted([d for d in modality_dir.iterdir() if d.is_dir()]) \
        if modality_dir.is_dir() else []
    all_frames: list[Path] = []
    for case in cases:
        frames = sorted(p for p in case.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not frames:
            continue
        if by_case:
            all_frames.append(frames[len(frames) // 2])
        else:
            all_frames.extend(frames)
    if not all_frames:
        return []
    if len(all_frames) <= n:
        return all_frames
    step = len(all_frames) / n          # 等间隔 → 天然跨病例
    return [all_frames[int(i * step)] for i in range(n)]


def evaluate(frames_per_modality: int, by_case: bool = False) -> dict:
    rows: list[dict] = []
    confusion: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int))
    confs: list[dict] = []
    t0 = time.time()

    for folder, expect in MODALITY_DIRS.items():
        frames = pick_frames(PROJECT_ROOT / "data" / folder / "img",
                             frames_per_modality, by_case=by_case)
        if not frames:
            log(f"  [!!] {folder}: 没找到帧（data/{folder}/img）")
            continue
        hit = 0
        mod_confs: list[float] = []
        for p in frames:
            g = mc_cls.classify(p.read_bytes())
            guess = g.name or "(判不出)"
            hit += (g.name == expect)
            mod_confs.append(g.confidence)
            confusion[expect][guess] += 1
            confs.append({"expect": expect, "guess": g.name or "",
                           "raw": g.raw_label, "conf": g.confidence,
                           "file": str(p.relative_to(PROJECT_ROOT))})
        rate = hit / len(frames)
        rows.append({"modality": expect, "n": len(frames), "hit": hit,
                     "rate": rate, "conf_min": min(mod_confs),
                     "conf_p50": pct(mod_confs, 50), "conf_p95": pct(mod_confs, 95)})
        log(f"  {expect:<22} n={len(frames):<4} 判对 {hit}/{len(frames)} = {rate * 100:5.1f}%  "
            f"conf min={min(mod_confs):.3f} P50={pct(mod_confs, 50):.3f} P95={pct(mod_confs, 95):.3f}")

    total = sum(r["n"] for r in rows)
    hit_all = sum(r["hit"] for r in rows)
    return {"rows": rows, "confusion": {k: dict(v) for k, v in confusion.items()},
            "details": confs, "total": total, "hit": hit_all,
            "rate": (hit_all / total) if total else 0.0,
            "elapsed_s": time.time() - t0}


def threshold_scan(details: list[dict], thresholds: list[float]) -> list[dict]:
    """阈值扫描：每个阈值下的判对率、**误判率**（自信地判错）、"判不出"占比。

    阈值的作用：分类器**只有 5 类、没有"不是影像"这一类**，非影像输入会被强行归类，
    所以低于阈值的猜测一律当作"判不出"（走契约 A1 边界 1）。阈值定多少必须看实测。

    **三个指标必须分开看**（这是本脚本最重要的地方，2026-11-04 实测得出）：
      · 判对率  —— 猜对了多少
      · 误判率  —— **自信地猜错**多少（≥阈值却错）→ 这一栏必须是 0，否则会误导医生
      · 判不出  —— 猜不出多少（低于阈值）→ 无害，回落到"请先告诉我分割哪个器官"
    蒙太奇实测：X光数据集里有 18.7% 的病例是"手机拍屏幕"的实拍照片，
    分类器把它们**稳定误判成 MRI**（置信度 0.46~0.84）；阈值 0.85 能把这批误判
    全部转成"判不出"（误判率 0），代价是这批帧不显示模态提示。
    """
    out = []
    for th in thresholds:
        hit = 0
        unknown = 0
        wrong = 0                      # 自信地判错（≥阈值却错）—— 危害最大的那一类
        per_mod: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for d in details:
            if d["conf"] < th:
                unknown += 1
                ok = False
            else:
                ok = d["guess"] == d["expect"]
                if not ok:
                    wrong += 1
            hit += ok
            per_mod[d["expect"]][0] += ok
            per_mod[d["expect"]][1] += 1
        n = len(details)
        out.append({
            "threshold": th,
            "hit": hit,
            "rate": hit / n if n else 0.0,
            "wrong": wrong,
            "wrong_ratio": wrong / n if n else 0.0,
            "unknown": unknown,
            "unknown_ratio": unknown / n if n else 0.0,
            "per_modality": {m: {"hit": v[0], "n": v[1], "rate": v[0] / v[1]}
                             for m, v in per_mod.items()},
        })
    return out


def suggest_threshold(details: list[dict]) -> dict:
    """给阈值建议（不是拍脑袋，是从实测里推）：
    - `safe`：所有**判对的**样本里最低的置信度 → 低于它就一律判不出（零误判线）
    - `balanced`：使各模态判对率都 ≥90% 的最大阈值（严格而不牺牲准确率）
    - `zero_wrong`：使**误判率 = 0** 的最小阈值 —— **这是给服务端实际用的那个**
      （自信地判错会误导医生，代价远大于"判不出"）
    """
    correct = [d["conf"] for d in details if d["guess"] == d["expect"]]
    safe = min(correct) if correct else 0.0
    best = 0.0
    zero_wrong = 0.0
    for th in [round(x * 0.01, 2) for x in range(0, 101)]:
        hit = sum(1 for d in details
                  if d["guess"] == d["expect"] and d["conf"] >= th)
        wrong = sum(1 for d in details
                    if d["guess"] != d["expect"] and d["conf"] >= th)
        per_mod: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for d in details:
            ok = d["guess"] == d["expect"] and d["conf"] >= th
            per_mod[d["expect"]][0] += ok
            per_mod[d["expect"]][1] += 1
        if wrong == 0:
            zero_wrong = th
        if per_mod and all(v[0] / v[1] >= 0.90 for v in per_mod.values()):
            best = th
    return {"safe_all_correct_min_conf": safe, "balanced_max_threshold": best,
            "zero_wrong_min_threshold": zero_wrong}


def write_report(res: dict, scan: list[dict], sug: dict, args) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = METRICS_DIR / f"modality_cls_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    cols = sorted({c for r in res["confusion"].values() for c in r})
    short = {m: m.split(" ")[0] for m in cols}
    lines: list[str] = []
    lines.append("# 帧级模态分类评测（N48）")
    lines.append("")
    lines.append(f"- 时间：{ts}")
    lines.append(f"- 数据：真实测试集病例 `data/<模态>/img/<病例>`，"
                 f"{'每病例取 1 帧（中位帧）' if getattr(args, 'by_case', False) else '跨病例等间隔抽样'}"
                 f"，每模态 {args.frames} 帧")
    lines.append(f"- 推理设备：{mc_cls.DEVICE}　权重：`{mc_cls.WEIGHTS_PATH.name}`")
    lines.append(f"- 分类器可用：{mc_cls.status()['available']}")
    lines.append("")
    lines.append("## 1. 判对率（Top-1）")
    lines.append("")
    lines.append("| 模态 | 帧数 | 判对 | 判对率 | conf 最小 | conf P50 | conf P95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in res["rows"]:
        lines.append(f"| {r['modality']} | {r['n']} | {r['hit']} | "
                     f"{r['rate'] * 100:.1f}% | {r['conf_min']:.3f} | "
                     f"{r['conf_p50']:.3f} | {r['conf_p95']:.3f} |")
    lines.append("")
    lines.append(f"**合计：{res['hit']}/{res['total']} = {res['rate'] * 100:.1f}%**"
                 f"（验收标准：五模态各 20 帧判对 ≥90%）")
    lines.append("")
    lines.append("## 2. 混淆矩阵（行=真实，列=预测）")
    lines.append("")
    lines.append("| 真实\\预测 | " + " | ".join(short.get(c, c) for c in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for real in MODALITY_DIRS.values():
        row = res["confusion"].get(real, {})
        lines.append(f"| {short.get(real, real)} | " +
                     " | ".join(str(row.get(c, 0)) for c in cols) + " |")
    lines.append("")
    lines.append("## 3. 置信度阈值扫描")
    lines.append("")
    lines.append("> **三个指标必须分开看**：**误判率**（自信地猜错 → 会误导医生，**必须为 0**）"
                 "与**判不出率**（猜不出 → 无害，回落到\"请先告诉我分割哪个器官\"）是两回事。")
    lines.append("")
    lines.append("| 阈值 | 判对 | 判对率 | **误判（自信错）** | 误判率 | 判不出 | 判不出占比 | 各模态最低判对率 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for s in scan:
        worst = min((v["rate"] for v in s["per_modality"].values()), default=0.0)
        lines.append(f"| {s['threshold']:.2f} | {s['hit']}/{res['total']} | "
                     f"{s['rate'] * 100:.1f}% | {s['wrong']} | "
                     f"{s['wrong_ratio'] * 100:.1f}% | {s['unknown']} | "
                     f"{s['unknown_ratio'] * 100:.1f}% | {worst * 100:.1f}% |")
    lines.append("")
    lines.append("## 4. 阈值建议（由实测推出，非拍脑袋）")
    lines.append("")
    lines.append(f"- **零误判线** `safe` = {sug['safe_all_correct_min_conf']:.3f}"
                 f"（所有判对样本里的最低置信度；低于它一律判不出）")
    lines.append(f"- **平衡点** `balanced` = {sug['balanced_max_threshold']:.2f}"
                 f"（使各模态判对率都 ≥90% 的最大阈值）")
    lines.append("")
    lines.append("> 分类器只有 5 类、**没有\"不是医学影像\"这一类**，非影像输入会被强行归类；")
    lines.append("> 因此阈值不是可选项。低于阈值的猜测一律当作\"判不出\"，走契约 A1 边界 1。")
    lines.append("")
    lines.append("## 5. 逐帧明细")
    lines.append("")
    lines.append("| 真实 | 预测 | 原始类别 | 置信度 | 文件 |")
    lines.append("|---|---|---|---:|---|")
    for d in res["details"]:
        lines.append(f"| {d['expect']} | {d['guess'] or '(判不出)'} | {d['raw']} | "
                     f"{d['conf']:.3f} | `{d['file']}` |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(
        {"result": {k: v for k, v in res.items() if k != "details"},
         "details": res["details"], "scan": scan, "suggest": sug,
         "args": vars(args)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="帧级模态分类评测（N48）")
    ap.add_argument("--frames", type=int, default=20,
                    help="每模态抽多少帧（默认 20 = 验收口径）")
    ap.add_argument("--thresholds", default="0,0.3,0.5,0.7,0.8,0.85,0.9,0.95,0.99",
                    help="阈值扫描的候选值（逗号分隔）")
    ap.add_argument("--by-case", action="store_true",
                    help="每病例取 1 帧（中位帧）而不是跨病例抽帧："
                         "最大化覆盖病例数，用于暴露\"整类病例整体不可判\"的问题")
    ap.add_argument("--min-rate", type=float, default=0.90,
                    help="验收线：各模态判对率下限（默认 0.90）")
    ap.add_argument("--min-conf", type=float, default=0.85,
                    help="服务端实际使用的置信度阈值（默认 0.85），"
                         "验收按这个阈值下的误判率=0 判定")
    args = ap.parse_args()

    st = mc_cls.status()
    if not st["available"]:
        log(f"[!!] 分类器不可用：{st['error']}")
        return 2
    log(f"分类器：{st['weights'].split(chr(92))[-1]}  device={st['device']}  "
        f"labels={st['labels']}")
    log(f"开始评测（每模态 {args.frames} 帧，跨病例等间隔抽样）…")
    res = evaluate(args.frames, by_case=args.by_case)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    scan = threshold_scan(res["details"], thresholds)
    sug = suggest_threshold(res["details"])
    out = write_report(res, scan, sug, args)

    log("-" * 68)
    log(f"合计判对 {res['hit']}/{res['total']} = {res['rate'] * 100:.1f}%"
        f"（耗时 {res['elapsed_s']:.1f}s）")
    worst = min((r["rate"] for r in res["rows"]), default=0.0)
    log(f"各模态最低判对率 {worst * 100:.1f}%（验收线 {args.min_rate * 100:.0f}%）")
    log(f"阈值建议：零误判线 {sug['safe_all_correct_min_conf']:.3f} / "
        f"平衡点 {sug['balanced_max_threshold']:.2f} / "
        f"误判率为 0 的最小阈值 {sug['zero_wrong_min_threshold']:.2f}")
    log(f"报告：{out / 'report.md'}")

    # 验收判定（用**服务端实际阈值** --min-conf，不是扫描表里的最小值）：
    #   ① 误判率（自信地猜错）必须为 0 —— 会误导医生，代价远大于"判不出"；
    #   ② 各模态判对率 ≥ --min-rate（默认 0.90，即总计划 N48 的验收线）。
    wrong_at = sum(1 for d in res["details"]
                   if d["conf"] >= args.min_conf and d["guess"] != d["expect"])
    hit_at = sum(1 for d in res["details"]
                 if d["conf"] >= args.min_conf and d["guess"] == d["expect"])
    n_at = hit_at + wrong_at
    log("-" * 68)
    log(f"按服务端实际阈值 {args.min_conf:.2f} 统计：")
    log(f"  判对 {hit_at}/{n_at} = {hit_at / n_at * 100:.1f}%　"
        f"**误判 {wrong_at}（{wrong_at / n_at * 100:.1f}%）**　"
        f"判不出 {res['total'] - n_at}（{(res['total'] - n_at) / res['total'] * 100:.1f}%）")
    if wrong_at:
        log("[!!] 存在'自信地判错'的样本 —— 必须提高阈值或换权重")
        return 1
    if worst < args.min_rate:
        log(f"[!!] 各模态最低判对率 {worst * 100:.1f}% < 验收线 {args.min_rate * 100:.0f}%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
