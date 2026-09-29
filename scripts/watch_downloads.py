"""下载看门狗：只要还有分片没下完，就保证下载器一直有进程在跑（无人值守）。

背景（2026-09-30）：夜间 PolypGen 7 片 + Amos 8 片因网络错误（ProxyError / SSLError /
IncompleteRead）全部失败，下载器跑完一轮就退出，数据缺口无人补。本脚本负责：
    循环 → 数未完成分片 → 若下载器不在跑则拉起（带多轮重试）→ 直到 0 片未完成。

用法：python scripts/watch_downloads.py [--interval 300] [--datasets PolypGen... Amos]
日志：runs/_download_watchdog.log
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import requests

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE = REPO_ROOT / "data" / "_labels_download_state.json"
DOWNLOADER = REPO_ROOT / "scripts" / "download_dataset_labels.py"
LOG = REPO_ROOT / "runs" / "_download_watchdog.log"
ENDPOINT = "https://hf-mirror.com"
REPO = "daizywang/EgoMed-IEMIS"

SPECS = {
    "Montgomery-County-CXR-Set": "Montgomery_CXR",
    "CAMUS": "CAMUS_US",
    "PolypGen2021_MultiCenterData_v3": "PolypGen_Endo",
    "Amos": "AMOS_CT",
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def pending_parts(datasets: list[str]) -> tuple[int, dict[str, int]]:
    """返回 (未完成分片总数, {数据集: 未完成数})。网络异常时抛错由调用方处理。"""
    state = load_state()
    total = 0
    detail: dict[str, int] = {}
    for ds in datasets:
        remote = SPECS[ds]
        r = requests.get(f"{ENDPOINT}/datasets/{REPO}/resolve/main/metadata/{remote}_parts.tsv", timeout=60)
        r.raise_for_status()
        n = 0
        for line in r.text.splitlines()[1:]:
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            part_no = int(cols[0])
            if not state.get(ds, {}).get(f"part_{part_no:03d}", {}).get("done", False):
                n += 1
        detail[ds] = n
        total += n
    return total, detail


def downloader_running() -> bool:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:  # noqa: BLE001
        return False
    return any("download_dataset_labels" in ln for ln in out.splitlines())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="下载看门狗")
    ap.add_argument("--interval", type=int, default=300, help="检查间隔秒数（默认 300）")
    ap.add_argument("--datasets", nargs="*", default=list(SPECS), help="关注的数据集")
    args = ap.parse_args()

    log(f"=== 看门狗启动，关注：{args.datasets} ===")
    idle_loops = 0
    while True:
        try:
            total, detail = pending_parts(args.datasets)
        except Exception as exc:  # noqa: BLE001
            log(f"查询待下载分片失败（{type(exc).__name__}），{args.interval}s 后重试")
            time.sleep(args.interval)
            continue

        if total == 0:
            log("所有分片已完成，看门狗退出")
            return 0

        if downloader_running():
            log(f"下载器运行中；待下载 {total} 片 {detail}")
            time.sleep(args.interval)
            continue

        log(f"未发现下载器进程，拉起新任务；待下载 {total} 片 {detail}")
        with (REPO_ROOT / "runs" / "_watchdog_download.out").open("a", encoding="utf-8") as fo, \
                (REPO_ROOT / "runs" / "_watchdog_download.err").open("a", encoding="utf-8") as fe:
            subprocess.run([sys.executable, str(DOWNLOADER), "--with-images", "--retry-rounds", "2"],
                           cwd=str(REPO_ROOT), stdout=fo, stderr=fe)
        idle_loops += 1
        if idle_loops > 40:
            log("连续拉起次数过多，退出（需人工检查网络）")
            return 1
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
