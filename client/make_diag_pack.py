#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""N28 诊断包 —— 一键把"排障需要的东西"打包成一个 zip

现场出问题时最耗时的不是修，而是来回问：
  "你哪个版本？""配置怎么设的？""服务端日志最后几行是什么？""指标报告在哪？"

本工具把下面这些**一次性收齐**（并附一份人读的 `summary.md`）：

  ① 环境     ：解释器 / Python / 关键包版本 / GPU / git commit / OS
  ② 配置快照 ：`configs/live.yaml`（现场到底怎么设的）
  ③ 服务端日志：`runs/server_log.txt` 末尾 N 行（含 A1/A2 每请求一行）
  ④ 最近报告 ：验收（acceptance）/ 长稳（soak）/ 混合负载（mixed_load）/ 控制台（console）
  ⑤ 最近帧   ：客户端落盘的抽样帧（默认最多 6 张，便于直接读图判断）
  ⑥ 会话日志 ：`runs/live/*/report.md` 最近几份

**不打包**模型权重与数据集（太大且与排障无关）。

用法
----
    python client/make_diag_pack.py                 # 全量，产出 zip
    python client/make_diag_pack.py --days 1        # 只看最近 1 天的产物
    python client/make_diag_pack.py --no-zip        # 只建目录（便于翻看）
    python client/make_diag_pack.py --tail 200      # 服务端日志取 200 行

输出
----
    runs/diag/<时间戳>/            ← 打包内容（summary.md 在根）
    runs/diag/diag_<时间戳>.zip    ← 可直接发出去的那个文件

归属：闫（客户端 / 运维工具）｜ 关联：docs/总计划.md 阶段四 N28
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_client as mc  # noqa: E402

PROJECT_ROOT = mc.PROJECT_ROOT
RUNS = PROJECT_ROOT / "runs"
OUT_ROOT = RUNS / "diag"
MAX_FILE_MB = 10.0          # 单文件超过这个大小就不进包（避免体积失控）
KEY_PKGS = ("torch", "torchvision", "ultralytics", "sam2", "fastapi", "uvicorn",
            "faster_whisper", "opencv-python", "numpy", "requests", "Pillow", "sounddevice")


def sh(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="ignore", timeout=30)
        return (r.stdout or "").strip() or (r.stderr or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"(执行失败：{exc})"


def collect_env() -> str:
    lines = [
        f"时间        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"主机        : {platform.node()}  ({platform.system()} {platform.release()})",
        f"Python      : {sys.version.splitlines()[0]}",
        f"解释器      : {sys.executable}",
        f"项目根      : {PROJECT_ROOT}",
        "",
        "—— git ——",
        sh(["git", "-C", str(PROJECT_ROOT), "log", "-1", "--oneline"]),
        "(未提交改动：)",
        sh(["git", "-C", str(PROJECT_ROOT), "status", "--short"]),
        "",
        "—— GPU ——",
        sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv"]),
        "",
        "—— 关键包版本 ——",
    ]
    for pkg in KEY_PKGS:
        try:
            mod = __import__(pkg.replace("-", "_"))
            lines.append(f"  {pkg:<18} {getattr(mod, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            lines.append(f"  {pkg:<18} (未安装)")
    lines += ["", "—— torch 设备 ——"]
    try:
        import torch  # noqa: PLC0415
        lines.append(f"  cuda 可用 : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"  设备      : {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  (torch 导入失败：{exc})")
    return "\n".join(lines)


def newest_dirs(root: Path, n: int, days: int | None) -> list[Path]:
    if not root.exists():
        return []
    cutoff = datetime.now() - timedelta(days=days) if days else None
    ds = [d for d in root.iterdir() if d.is_dir()]
    if cutoff:
        ds = [d for d in ds if datetime.fromtimestamp(d.stat().st_mtime) >= cutoff]
    ds.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return ds[:n]


def read_verdict(p: Path) -> str:
    """从 report.md 里抽"结论/总判定"一行，便于写进 summary。"""
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if ("结论" in s or "总判定" in s) and "|" not in s:
                return s.lstrip("#- ")[:120]
    except Exception:  # noqa: BLE001
        pass
    return "(未见结论行)"


def copy_file(src: Path, dst: Path, notes: list[str]) -> bool:
    if not src.exists():
        return False
    mb = src.stat().st_size / 1048576
    if mb > MAX_FILE_MB:
        notes.append(f"跳过（{mb:.1f}MB 超上限）：{src.relative_to(PROJECT_ROOT)}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="EgoMed-Agent 诊断包（N28）")
    ap.add_argument("--out", default="", help="输出目录（默认 runs/diag/<时间戳>）")
    ap.add_argument("--days", type=int, default=2, help="只看最近 N 天的产物（默认 2；0=不限）")
    ap.add_argument("--tail", type=int, default=120, help="服务端日志取末尾多少行（默认 120）")
    ap.add_argument("--frames", type=int, default=6, help="附带最近几张落盘帧（默认 6）")
    ap.add_argument("--runs", type=int, default=3, help="每类报告取最近几份（默认 3）")
    ap.add_argument("--no-zip", action="store_true", help="只建目录，不打包 zip")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage = Path(args.out) if args.out else (OUT_ROOT / stamp)
    if not stage.is_absolute():
        stage = PROJECT_ROOT / stage
    stage.mkdir(parents=True, exist_ok=True)
    days = args.days or None
    notes: list[str] = []
    verdicts: list[str] = []

    mc.log("=" * 74)
    mc.log(f"N28 诊断包　→ {stage}")
    mc.log("=" * 74)

    # ① 环境
    (stage / "env.txt").write_text(collect_env(), encoding="utf-8")
    mc.log("  [1/6] 环境信息 env.txt")

    # ② 配置快照
    for cfg in (PROJECT_ROOT / "configs" / "live.yaml",):
        if copy_file(cfg, stage / "config" / cfg.name, notes):
            mc.log(f"  [2/6] 配置快照 {cfg.name}")

    # ③ 服务端日志尾部
    slog = RUNS / "server_log.txt"
    if slog.exists():
        tail = slog.read_text(encoding="utf-8", errors="ignore").splitlines()[-args.tail:]
        (stage / "server_log_tail.txt").write_text("\n".join(tail), encoding="utf-8")
        mc.log(f"  [3/6] 服务端日志尾部 {len(tail)} 行（共 {slog.stat().st_size/1048576:.1f}MB）")

    # ④ 各类最近报告
    kinds = [
        ("acceptance", RUNS / "acceptance", "validation"),
        ("soak", RUNS / "soak", "stability"),
        ("mixed_load", RUNS / "mixed_load", "concurrency"),
        ("console", RUNS / "console", "console"),
        ("live", RUNS / "live", "client_run"),
    ]
    n_rep = 0
    for label, root, sub in kinds:
        for d in newest_dirs(root, args.runs, days):
            if d.name.startswith("_"):        # 自测目录不进包
                continue
            for fname in ("report.md", "report.json", "metrics.csv", "recording.json"):
                src = d / fname
                if copy_file(src, stage / "reports" / sub / d.name / fname, notes):
                    n_rep += 1
            rp = d / "report.md"
            if rp.exists():
                verdicts.append(f"- {label}/{d.name}: {read_verdict(rp)}")
    mc.log(f"  [4/6] 报告 {n_rep} 个文件（{len(kinds)} 类）")

    # ⑤ 最近落盘帧（客户端 samples/ 与控制台 recording/frames/ 都要收）
    n_fr = 0
    frame_sources: list[tuple[Path, str]] = []
    for d in newest_dirs(RUNS / "live", args.runs, days):
        frame_sources.append((d / "samples", d.name))
    for d in newest_dirs(RUNS / "console", args.runs, days):
        frame_sources.append((d / "recording" / "frames", f"console_{d.name}"))
    for sdir, tag in frame_sources:
        if not sdir.exists():
            continue
        picked = sorted(sdir.glob("*.jpg"))
        if not picked:
            continue
        # 均匀抽 2 张（首尾各一，便于对比渲染）
        picks = picked[:1] + picked[-1:] if len(picked) > 1 else picked
        for jpg in picks:
            if copy_file(jpg, stage / "frames" / tag / jpg.name, notes):
                n_fr += 1
    mc.log(f"  [5/6] 附带帧 {n_fr} 张（来自 {len(frame_sources)} 个来源）")

    # ⑥ summary
    env_head = (stage / "env.txt").read_text(encoding="utf-8").splitlines()[:8]
    summary = [
        "# EgoMed-Agent 诊断包", "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　"
        f"范围：最近 {args.days or '全部'} 天　工具：`client/make_diag_pack.py`", "",
        "## 这个包里有什么", "",
        "| 文件 | 内容 |", "|------|------|",
        "| `env.txt` | 解释器 / Python / 关键包 / GPU / git commit / OS |",
        "| `config/live.yaml` | 现场客户端配置快照 |",
        "| `server_log_tail.txt` | 服务端日志末尾（含 A1/A2 每请求一行） |",
        "| `reports/` | 验收(validation) / 长稳(stability) / 混合负载(concurrency) / 控制台 / 客户端运行 |",
        "| `frames/` | 最近落盘的抽样帧（可直接读图判断渲染对不对） |", "",
        "## 环境速览", "", "```", *env_head, "```", "",
        "## 最近报告结论", "", *(verdicts or ["(未找到 report.md)"]), "",
    ]
    if notes:
        summary += ["## 打包时跳过的内容", "", *[f"- {n}" for n in notes], ""]
    summary += [
        "## 怎么用", "",
        "1. 先看 `env.txt` 确认版本与设备一致；",
        "2. 看 `server_log_tail.txt` 找 `status=error` / 500 / 超时；",
        "3. 看 `reports/` 里相应报告的结论行与指标表；",
        "4. 画面/渲染问题直接看 `frames/` 里的图。", "",
    ]
    (stage / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    mc.log("  [6/6] 摘要 summary.md")

    # 打包
    out = stage
    if not args.no_zip:
        zip_path = stage.parent / f"diag_{stage.name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(stage))
        out = zip_path
        mc.log(f"  [zip] {zip_path.relative_to(PROJECT_ROOT)}"
               f"（{zip_path.stat().st_size/1048576:.2f} MB）")
    mc.log("=" * 74)
    mc.log(f"诊断包就绪：{out}")
    mc.log("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
