"""复现补跑 · 统一进度面板（一条命令看全部）。

用法：python scripts/show_progress.py
显示：后台进程存活、当前下载分片与速率/ETA、各数据集数据就绪度、评测队列状态、已有结果、完成标记。
"""
from __future__ import annotations

import csv
import datetime
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BASE = (RUNS / "eval_yolo26_original_sam2_video_schedule_reset_retrack_correction"
        / "egomed5_yolo26m_original_sam2_online_schedule_reset_retrack_iou06")
DATASETS = ["ACDC", "Montgomery-County-CXR-Set", "CAMUS",
            "PolypGen2021_MultiCenterData_v3", "Amos"]


def procs() -> dict[str, int]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
         "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
        capture_output=True, text=True, timeout=60).stdout
    found: dict[str, int] = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        pid, cmd = line.split("|", 1)
        for key, tag in [("watch_downloads", "看门狗"), ("run_eval_queue", "排队器"),
                         ("egomed-agent-iou06", "评测"), ("download_dataset_labels", "下载器")]:
            if key in cmd:
                found[tag] = int(pid)
    return found


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    now = datetime.datetime.now()
    print("=" * 96)
    print("EgoMed-Agent 复现补跑 · 进度面板   %s" % now.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 96)

    alive = procs()
    print("\n■ 后台进程")
    for tag in ["看门狗", "排队器", "下载器", "评测"]:
        print("   %-6s %s" % (tag, ("运行中 PID %d" % alive[tag]) if tag in alive else "— 未运行"))

    print("\n■ 当前下载（最新分片进度）")
    out = RUNS / "_watchdog_download.out"
    if out.exists():
        lines = [l for l in out.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        for l in lines[-2:]:
            print("   ", l.strip()[:120])
        starts = [l for l in lines if "开始下载" in l]
        if starts:
            print("   （最近启动）", starts[-1].strip()[:110])
    err = RUNS / "_watchdog_download.err"
    if err.exists() and err.stat().st_size:
        tail = [l for l in err.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        if tail:
            print("   ⚠ 下载 stderr 末尾:", tail[-1][:110])

    print("\n■ 数据就绪度（测试病例：标注 + 完整帧）")
    sched = list(csv.DictReader(
        (ROOT / "data/text_prompt_eval/egomed5_test_prompt_schedule.csv").open(encoding="utf-8")))
    want: dict[tuple[str, str], int] = {}
    for r in sched:
        if r.get("prompt_type") == "exact":
            want.setdefault((r["dataset"], str(r["case_id"])), int(float(r["num_case_frames"])))
    for ds in DATASETS:
        cases = [(ds, c) for (d, c) in want if d == ds]
        ready = 0
        for (_, case) in cases:
            lab = ROOT / "data" / ds / "label" / case
            img = ROOT / "data" / ds / "img" / case
            if lab.is_dir() and any(lab.glob("*.png")) and \
                    len(list(img.glob("*.jpg"))) >= want[(ds, case)]:
                ready += 1
        flag = "✅ 齐" if ready == len(cases) else "🔸 %d/%d" % (ready, len(cases))
        print("   %-38s %s" % (ds, flag))

    print("\n■ 评测结果（class_summary.csv）")
    for ds in DATASETS:
        p = BASE / ds / "class_summary.csv"
        if not p.exists():
            print("   %-38s 尚无" % ds)
            continue
        age = (now - datetime.datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() / 3600
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        brief = " / ".join("%s %.4f" % (r["target_class"][:16], float(r["mean_dice"])) for r in rows)
        print("   %-38s %.1f 小时前  %s" % (ds, age, brief))

    print("\n■ 评测队列日志（末尾 3 行）")
    q = RUNS / "_eval_queue.log"
    if q.exists():
        for l in q.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-3:]:
            print("   ", l[:120])

    print("\n■ 完成标记")
    done = RUNS / "_eval_queue_DONE"
    print("   runs/_eval_queue_DONE:", "✅ 已生成（四集全部跑完）" if done.exists() else "尚未（仍在进行）")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
