"""复现补跑编排器：数据到齐后自动逐集拉起评测（无人值守）。

背景（2026-09-29，N19 复现补跑）：
    其余 4 个数据集的 GT 标注与完整帧需要从 HF 分片流式补齐（`download_dataset_labels.py`
    正在后台跑）。评测脚本一次只跑一个数据集（`EGOMED_EVAL_DATASETS` 指定），
    且 GPU/内存有限，必须**串行**。本脚本负责：等数据 → 拉起评测 → 等评测结束 → 下一集。

判定规则：
    - **数据就绪**：该数据集所有"无歧义测试病例"（评测计划表 prompt_type=exact）
      都有 `label/<case>/*.png`，且 `img/<case>` 帧数 ≥ 计划表 `num_case_frames`；
    - **已完成**：`<RUN_ROOT>/<DS>/class_summary.csv` 比本脚本启动时间新（即本轮跑出来的）；
    - 若评测异常退出（未产出新结果）→ 重试，最多 2 次。

用法：
    python scripts/run_eval_queue.py                 # 默认队列：Montgomery → CAMUS → PolypGen → Amos
    python scripts/run_eval_queue.py --datasets CAMUS Amos
日志：`runs/_eval_queue.log`；状态：`runs/_eval_queue_state.json`。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SCHEDULE = DATA_ROOT / "text_prompt_eval" / "egomed5_test_prompt_schedule.csv"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "egomed_agent" / "egomed-agent-iou06.py"
RUN_ROOT = (REPO_ROOT / "runs" / "eval_yolo26_original_sam2_video_schedule_reset_retrack_correction"
            / "egomed5_yolo26m_original_sam2_online_schedule_reset_retrack_iou06")
LOG = REPO_ROOT / "runs" / "_eval_queue.log"
STATE = REPO_ROOT / "runs" / "_eval_queue_state.json"

DEFAULT_QUEUE = [
    "Montgomery-County-CXR-Set",
    "CAMUS",
    "PolypGen2021_MultiCenterData_v3",
    "Amos",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def test_cases(dataset: str) -> dict[str, int]:
    """{case_id: num_case_frames}（仅无歧义测试用例）。"""
    out: dict[str, int] = {}
    with SCHEDULE.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("dataset") == dataset and row.get("prompt_type") == "exact":
                out.setdefault(str(row["case_id"]), int(float(row["num_case_frames"])))
    return out


def readiness(dataset: str) -> tuple[int, int]:
    """返回 (就绪病例数, 总病例数)。"""
    want = test_cases(dataset)
    ready = 0
    for case, n_frames in want.items():
        lab = DATA_ROOT / dataset / "label" / case
        img = DATA_ROOT / dataset / "img" / case
        has_label = lab.is_dir() and any(lab.glob("*.png"))
        n_img = len(list(img.glob("*.jpg"))) if img.is_dir() else 0
        if has_label and n_img >= n_frames:
            ready += 1
    return ready, len(want)


def results_fresh(dataset: str, since: float) -> bool:
    p = RUN_ROOT / dataset / "class_summary.csv"
    return p.exists() and p.stat().st_mtime > since


def eval_running() -> bool:
    """是否已有评测进程在跑（避免并发抢 GPU/内存）。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:  # noqa: BLE001
        return False
    return any("egomed-agent-iou06" in ln for ln in out.splitlines())


def run_one(dataset: str, attempt: int) -> bool:
    started = time.time()
    log(f"启动评测：{dataset}（第 {attempt} 次尝试）")
    env = dict(os.environ)
    env["EGOMED_EVAL_DATASETS"] = dataset
    out_path = REPO_ROOT / "runs" / f"_eval_queue_{dataset.split('-')[0]}.out"
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {dataset} attempt {attempt} =====\n")
    with out_path.open("a", encoding="utf-8") as fh_out, \
            (REPO_ROOT / "runs" / f"_eval_queue_{dataset.split('-')[0]}.err").open("a", encoding="utf-8") as fh_err:
        proc = subprocess.Popen([sys.executable, str(EVAL_SCRIPT)], env=env,
                                stdout=fh_out, stderr=fh_err, cwd=str(REPO_ROOT))
        proc.wait()
    ok = results_fresh(dataset, started)
    log(f"评测结束：{dataset} 退出码 {proc.returncode}，结果{'已写出' if ok else '未写出'}，"
        f"耗时 {(time.time() - started) / 60:.1f} 分钟")
    return ok


LOCK = REPO_ROOT / "runs" / "_eval_queue.lock"


def another_instance_running() -> bool:
    """单实例保护：锁文件里记 PID，若该进程仍活着则说明已有排队器在跑。"""
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | "
                              "Select-Object -ExpandProperty Id"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return False
    return str(pid) in out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="补跑评测排队器")
    ap.add_argument("--datasets", nargs="*", default=None, help="队列（默认 Montgomery→CAMUS→PolypGen→Amos）")
    ap.add_argument("--poll", type=int, default=120, help="等数据/等进程的轮询秒数（默认 120）")
    args = ap.parse_args()

    if another_instance_running():
        print("已有排队器在运行（见 runs/_eval_queue.lock），本次退出", flush=True)
        return 0
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")

    queue = args.datasets or DEFAULT_QUEUE
    start_time = time.time()
    state = load_state()
    log(f"=== 排队器启动，队列：{queue}（PID {os.getpid()}）===")

    pending = [d for d in queue if not state.get(d, {}).get("done")]
    done_all = False
    while pending:
        # 先等已有评测进程结束（例如手工/上一轮遗留的当前任务）
        if eval_running():
            log("检测到已有评测进程在跑，等待其结束…")
            time.sleep(args.poll)
            # 若该进程正是本轮要跑的数据集，结束后会被判为"已完成"
            for ds in list(pending):
                if results_fresh(ds, start_time):
                    state[ds] = {"done": True, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                    save_state(state)
                    pending.remove(ds)
                    log(f"{ds} 结果已产出 → 标记完成")
            continue

        ds = pending[0]
        ready, total = readiness(ds)
        if results_fresh(ds, start_time):
            state[ds] = {"done": True, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_state(state)
            pending.remove(ds)
            log(f"{ds} 结果已存在 → 跳过")
            continue

        if ready < total:
            log(f"{ds} 数据未齐（{ready}/{total} 病例：标注+完整帧），{args.poll}s 后再查")
            time.sleep(args.poll)
            continue

        for attempt in (1, 2, 3):
            if run_one(ds, attempt):
                state[ds] = {"done": True, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                save_state(state)
                pending.remove(ds)
                break
            log(f"{ds} 第 {attempt} 次未产出结果，稍后重试")
            time.sleep(60)
        else:
            log(f"{ds} 连续失败，跳过（需人工检查日志）")
            state[ds] = {"done": False, "error": "3 attempts failed"}
            save_state(state)
            pending.remove(ds)

    LOCK.unlink(missing_ok=True)
    (REPO_ROOT / "runs" / "_eval_queue_DONE").write_text(
        f"all done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    log("=== 队列全部完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
