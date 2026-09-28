#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASR 字准率与 NLU 解析率评估（N8-3：把 A.11 的 8 句扩到 50 句）

做法（不依赖服务端，直接调模块，链路与 A2 内部一致）：
    参考文本 = `client/test_audio_50.txt` 的每一行（TTS 时用的原文）
    待测音频 = `data/samples/audio50/cmd_01.wav … cmd_50.wav`（由 `client/gen_test_audio.ps1` 生成）
    指标：① 字准率 = 1 − CER（CER = 编辑距离 / 参考长度，按字符）
          ② NLU 解析率 = parse_target(text).ok 的比例；另统计"多目标（需澄清）"与失败句
    报告：runs/metrics/asr_<时间戳>/report.md（含逐句明细）

用法：
    python client/eval_asr.py                    # 全部 50 句
    python client/eval_asr.py --limit 10         # 只跑前 10 句（快速自检）
归属：闫（测试/评估工具）｜ 关联：docs/API_CONTRACT.md A.11、docs/总计划.md N8
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "demo"))

REF_FILE = PROJECT_ROOT / "client" / "test_audio_50.txt"
AUDIO_DIR = PROJECT_ROOT / "data" / "samples" / "audio50"
METRICS_DIR = PROJECT_ROOT / "runs" / "metrics"


def log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("gbk", "replace").decode("gbk"), flush=True)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(text: str) -> str:
    """对齐比较用的归一：去掉空白与常见标点（TTS 不会加标点，但模型可能加）"""
    drop = set(" \t，。、？！,.?!;；:：\"'“”‘’()（）")
    return "".join(ch for ch in text if ch not in drop).lower()


def main() -> int:
    ap = argparse.ArgumentParser(description="ASR 字准率 + NLU 解析率评估（N8-3）")
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 句（0=全部）")
    ap.add_argument("--ref", default=str(REF_FILE), help="参考文本文件（每行一句）")
    ap.add_argument("--audio-dir", default=str(AUDIO_DIR), help="音频目录（cmd_01.wav …）")
    ap.add_argument("--out", default="", help="输出目录（默认 runs/metrics/asr_<时间戳>）")
    ap.add_argument("--nonsense", type=int, default=3,
                    help="末尾 N 句是'无关语句'（预期无法解析），单独统计拦截率，不计入指令解析率")
    args = ap.parse_args()

    refs = [ln.strip() for ln in Path(args.ref).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        refs = refs[:args.limit]
    adir = Path(args.audio_dir)
    if not adir.is_dir():
        log(f"[!!] 音频目录不存在：{adir}\n     请先运行：powershell -ExecutionPolicy Bypass "
            f"-File client/gen_test_audio.ps1 -CmdFile client/test_audio_50.txt -OutDir {adir}")
        return 2

    import asr_medical
    from nl_parser import parse_target

    log("=" * 78)
    log(f"ASR 评估：{len(refs)} 句　音频目录 {adir.relative_to(PROJECT_ROOT)}")
    log("=" * 78)
    log("加载/预热模型 …")
    t0 = time.perf_counter()
    asr_medical.warmup()
    log(f"预热完成 {time.perf_counter() - t0:.0f}s")

    rows = []
    for i, ref in enumerate(refs, 1):
        wav = adir / f"cmd_{i:02d}.wav"
        if not wav.is_file():
            rows.append({"i": i, "ref": ref, "text": "", "cer": 1.0, "ok": False,
                         "multi": False, "err": "缺文件", "ms": 0.0})
            continue
        t1 = time.perf_counter()
        text = asr_medical.transcribe(str(wav))
        ms = (time.perf_counter() - t1) * 1000
        r = parse_target(text)
        nr, nt = norm(ref), norm(text)
        cer = levenshtein(nr, nt) / max(1, len(nr))
        rows.append({"i": i, "ref": ref, "text": text, "cer": cer,
                     "ok": bool(text) and r.ok, "multi": bool(text) and r.ok and len(r.targets) > 1,
                     "targets": list(getattr(r, "targets", []) or []),
                     "err": "" if (text and r.ok) else (r.message or "转写为空"), "ms": ms})

    n = len(rows)
    n_cmd = max(0, n - args.nonsense)            # 有效指令句数
    n_non = n - n_cmd                            # 无关语句数
    cmd_rows = rows[:n_cmd]
    non_rows = rows[n_cmd:]
    hits = sum(1 for x in cmd_rows if x["ok"])
    blocked = sum(1 for x in non_rows if not x["ok"])   # 无关语句被正确拒绝的条数
    exact = sum(1 for x in cmd_rows if x["cer"] == 0)
    cer_mean = sum(x["cer"] for x in cmd_rows) / max(1, n_cmd)
    lat = sorted(x["ms"] for x in rows if x["ms"] > 0)
    p50 = lat[len(lat) // 2] if lat else 0.0
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0

    log("-" * 78)
    log(f"指令句 {n_cmd} 条：逐字完全正确 {exact}/{n_cmd}（{exact / max(1, n_cmd):.0%}），"
        f"平均字准率 {1 - cer_mean:.1%}（CER {cer_mean:.3f}）")
    log(f"指令句 NLU 解析成功：{hits}/{n_cmd}（{hits / max(1, n_cmd):.0%}），"
        f"其中多目标需澄清 {sum(1 for x in cmd_rows if x['multi'])} 句")
    log(f"无关语句拦截：{blocked}/{n_non}（预期无法解析）")
    log(f"转写耗时 P50/P95：{p50:.0f}/{p95:.0f} ms")
    bad = [x for x in cmd_rows if not x["ok"]]
    if bad:
        log(f"指令句解析失败 {len(bad)} 条：")
        for x in bad[:12]:
            log(f"  #{x['i']:02d} 参考「{x['ref']}」→ 转写「{x['text']}」（{x['err']}）")
    log("=" * 78)

    out_dir = Path(args.out) if args.out else METRICS_DIR / f"asr_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# ASR 字准率与 NLU 解析率评估（N8-3）", "",
             f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
             f"- 句数：{n}（参考文本 `client/{Path(args.ref).name}`，音频 `{adir.relative_to(PROJECT_ROOT)}`）",
             f"- 模型：`scripts/demo/asr_medical.py`（medium + zh + 领域 prompt + hotwords + VAD + 幻觉防护）",
             "",
             "## 汇总", "",
             "| 指标 | 数值 |", "|------|------|",
             f"| 指令句数 | {n_cmd}（另 {n_non} 条无关语句） |",
             f"| 逐字完全正确 | {exact}/{n_cmd}（{exact / max(1, n_cmd):.0%}） |",
             f"| 平均字准率（1−CER） | **{1 - cer_mean:.1%}** |",
             f"| NLU 解析成功 | **{hits}/{n_cmd}（{hits / max(1, n_cmd):.0%}）** |",
             f"| 多目标（需澄清） | {sum(1 for x in cmd_rows if x['multi'])} |",
             f"| 无关语句拦截 | {blocked}/{n_non}（预期解析失败） |",
             f"| 转写耗时 P50 / P95 | {p50:.0f} / {p95:.0f} ms |",
             "", "## 逐句明细", "",
             "| # | 参考文本 | 转写 | 字准 | 解析 | 目标 | 耗时 |",
             "|---|---------|------|------|------|------|------|"]
    for x in rows:
        lines.append(f"| {x['i']} | {x['ref']} | {x['text'] or '（空）'} | "
                     f"{1 - x['cer']:.0%} | {'OK' if x['ok'] else 'FAIL'} | "
                     f"{','.join(x.get('targets') or []) or '-'} | {x['ms']:.0f}ms |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"报告：{(out_dir / 'report.md').relative_to(PROJECT_ROOT)}")
    return 0 if hits == n else 1


if __name__ == "__main__":
    sys.exit(main())
