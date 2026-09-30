"""从 Hugging Face 数据集分片（tar）中**流式抽取 GT 标注**，跳过图像正文。

背景（2026-09-29 复现补跑）：
    `daizywang/EgoMed-IEMIS` 数据集以 ~4 GB 未压缩 tar 分片发布
    （`data/<REMOTE>/parts/<NAME>_part_%03d.tar`），成员为
    `<本地数据集名>/img/<病例>/*.jpg` 与 `<本地数据集名>/label/<病例>/*.png`。
    本机只有 ACDC 全量标注 + Amos 2 个病例，其余数据集缺 GT，无法算 Dice。

策略：
    整片下载不现实（合计 ~94 GB / 2~3 MB/s ≈ 10 小时），而标注 PNG 仅 ~10 KB/张。
    因此边下载边用 tarfile 流式解析，**只落盘 `label/` 成员**，图像字节直接丢弃；
    不保存 tar 本体，磁盘占用仅标注体积（每数据集几十 MB）。

用法：
    python scripts/download_dataset_labels.py --list
    python scripts/download_dataset_labels.py --datasets Montgomery-County-CXR-Set
    python scripts/download_dataset_labels.py                      # 默认全部缺标注的数据集
    python scripts/download_dataset_labels.py --from-local-part D:\\path\\to\\part_004.tar
断点续传：`data/_labels_download_state.json` 记录已完成分片，重跑自动跳过。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = "daizywang/EgoMed-IEMIS"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
REPO_ROOT = Path(os.environ.get("EGOMED_ROOT") or Path(__file__).resolve().parents[1])
DATA_ROOT = REPO_ROOT / "data"
STATE_FILE = DATA_ROOT / "_labels_download_state.json"

# 本地数据集名 -> (远端数据目录名, 分片文件名模板)
DATASETS = {
    "Montgomery-County-CXR-Set": ("Montgomery_CXR", "Montgomery_CXR_part_{n:03d}.tar"),
    "CAMUS": ("CAMUS_US", "CAMUS_part_{n:03d}.tar"),
    "PolypGen2021_MultiCenterData_v3": ("PolypGen_Endo", "PolypGen_Endo_part_{n:03d}.tar"),
    "Amos": ("AMOS_CT", "Amos_part_{n:03d}.tar"),
}


def resolve_url(path: str) -> str:
    return f"{ENDPOINT}/datasets/{REPO}/resolve/main/{path}"


class _RangeReader(io.RawIOBase):
    """按 HTTP Range 分块读取远端文件，**块级自动重试 + 并发预取**。

    动机（2026-09-30）：
      1) 整片 4 GB 单连接流式下载一旦中途断（ProxyError / SSLError / IncompleteRead），
         整片就得从头再来；网络抖动时几乎无法完成 → 改为 8 MB 分块，失败只重试当前块。
      2) 单流实测仅 0.85 MB/s，而 4 并发聚合 1.41 MB/s（链路瓶颈但并发仍有收益）→
         用线程池**乱序预取、顺序交付**（对 tarfile 表现为连续字节流）。
    """

    def __init__(self, url: str, total: int, chunk: int = 8 << 20, workers: int = 4,
                 retries: int = 8, timeout: int = 120):
        self.url = url
        self.total = total
        self.chunk = chunk
        self.workers = max(1, workers)
        self.retries = retries
        self.timeout = timeout
        self.buf = b""
        self.buf_off = 0
        self.n_read = 0         # 对本地累计输出字节
        self.n_bytes = 0        # 已抓取字节
        self.n_retry = 0
        self._next_submit = 0
        self._queue: list = []
        self._local = threading.local()
        self._exec = ThreadPoolExecutor(max_workers=self.workers)
        self._fill_window()

    # -- 内部：连接复用（每线程一个 Session） --------------------------------
    def _session(self) -> requests.Session:
        sess = getattr(self._local, "sess", None)
        if sess is None:
            sess = requests.Session()
            self._local.sess = sess
        return sess

    def _fetch(self, start: int, end: int) -> bytes:
        last = "?"
        for attempt in range(self.retries):
            try:
                r = self._session().get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=self.timeout)
                if r.status_code in (200, 206) and r.content:
                    return r.content
                last = f"HTTP {r.status_code}"
            except Exception as exc:  # noqa: BLE001
                last = type(exc).__name__
                self.n_retry += 1
            time.sleep(min(20, 2 * (attempt + 1)))
        raise OSError(f"Range 读取失败（{start}-{end}/{self.total}）：{last}")

    def _fill_window(self) -> None:
        while len(self._queue) < self.workers and self._next_submit < self.total:
            start = self._next_submit
            end = min(start + self.chunk, self.total) - 1
            self._queue.append(self._exec.submit(self._fetch, start, end))
            self._next_submit = end + 1

    def _fill(self) -> bool:
        if self.buf_off < len(self.buf):
            return True
        if not self._queue:
            return False
        fut = self._queue.pop(0)          # 先进先出 → 投递顺序 == 文件顺序
        self.buf = fut.result()
        self.buf_off = 0
        self.n_bytes += len(self.buf)
        self._fill_window()
        return True

    def readable(self) -> bool:  # noqa: D102
        return True

    def close(self) -> None:  # noqa: D102
        try:
            self._exec.shutdown(wait=False)
        finally:
            super().close()

    def readinto(self, b):  # type: ignore[override]
        if not self._fill():
            return 0
        n = min(len(b), len(self.buf) - self.buf_off)
        b[:n] = self.buf[self.buf_off:self.buf_off + n]
        self.buf_off += n
        self.n_read += n
        return n

    def read(self, size=-1):  # type: ignore[override]
        out = bytearray()
        if size is None or size < 0:
            while self._fill():
                out += self.buf[self.buf_off:]
                self.buf_off = len(self.buf)
        else:
            while len(out) < size and self._fill():
                n = min(size - len(out), len(self.buf) - self.buf_off)
                out += self.buf[self.buf_off:self.buf_off + n]
                self.buf_off += n
        self.n_read += len(out)
        return bytes(out)


SCHEDULE = DATA_ROOT / "text_prompt_eval" / "egomed5_test_prompt_schedule.csv"


def load_needed_cases(with_frames: bool = False):
    """从评测计划表读**测试病例**（唯一权威来源，仓库随发布提供）。

    with_frames=False -> {数据集: {病例号}}（用于决定补哪些病例的帧）
    with_frames=True  -> {数据集: (计划帧数合计, 本地帧数合计)}（用于报告覆盖率）
    """
    import csv

    rows = list(csv.DictReader(SCHEDULE.open(encoding="utf-8")))
    if not with_frames:
        out: dict[str, set[str]] = {}
        for r in rows:
            if r.get("prompt_type") == "exact":
                out.setdefault(r["dataset"], set()).add(str(r["case_id"]))
        return out

    want: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for r in rows:
        if r.get("prompt_type") != "exact":
            continue
        ds, case = r["dataset"], str(r["case_id"])
        if (ds, case) in seen:
            continue
        seen.add((ds, case))
        want[ds] = want.get(ds, 0) + int(float(r["num_case_frames"]))
    have: dict[str, int] = {}
    for ds, case in seen:
        img = DATA_ROOT / ds / "img" / case
        have[ds] = have.get(ds, 0) + (len(list(img.glob("*.jpg"))) if img.exists() else 0)
    return {ds: (want.get(ds, 0), have.get(ds, 0)) for ds in want}


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_part_manifest(remote: str) -> list[tuple[int, int, int]]:
    """读取 metadata/<remote>_parts.tsv -> [(part_no, bytes, cases)]。"""
    r = requests.get(resolve_url(f"metadata/{remote}_parts.tsv"), timeout=60)
    r.raise_for_status()
    rows: list[tuple[int, int, int]] = []
    for line in r.text.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 3:
            rows.append((int(cols[0]), int(cols[1]), int(cols[2])))
    return rows


def _fmt(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}min"
    return f"{sec / 3600:.1f}h"


def extract_labels_from_stream(stream: io.BufferedIOBase, local_ds: str,
                               needed_cases: set[str] | None = None,
                               save_images: bool = False) -> tuple[int, int, int, int]:
    """流式解析 tar：写全部 `label/` 成员；可选补写 `img/` 成员。

    背景：本机 4 个数据集的 `img/` 是**截断的下载**（每病例只有前 ~1/3 帧），
    而评测计划表的 `num_case_frames` 是完整长度；tar 每个字节反正都要读，
    因此顺手把**测试病例**的完整帧补下来（已存在的帧跳过，不重复写盘）。

    返回 (标注文件数, 标注字节, 图像文件数, 图像字节)。
    """
    lab_prefix = local_ds + "/label/"
    img_prefix = local_ds + "/img/"
    needed = needed_cases or set()
    n_lab = n_img = 0
    b_lab = b_img = 0
    tar = tarfile.open(fileobj=stream, mode="r|")   # 流式，不 seek
    for member in tar:
        if not member.isfile():
            continue
        name = member.name
        want_label = name.startswith(lab_prefix)
        want_image = False
        if save_images and name.startswith(img_prefix):
            case = name[len(img_prefix):].split("/", 1)[0]
            want_image = case in needed
        if not (want_label or want_image):
            continue
        rel = name[len(local_ds) + 1:]          # label|img/<case>/<file>
        dst = DATA_ROOT / local_ds / rel
        if want_image and dst.exists():         # 本地已有该帧 → 不重复写
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = tar.extractfile(member)
        if src is None:
            continue
        data = src.read()
        dst.write_bytes(data)
        if want_label:
            n_lab += 1
            b_lab += len(data)
        else:
            n_img += 1
            b_img += len(data)
    return n_lab, b_lab, n_img, b_img


def run_part(local_ds: str, remote: str, tmpl: str, part_no: int, part_bytes: int,
             needed_cases: set[str] | None = None, save_images: bool = False,
             workers: int = 4) -> dict:
    """下载单个分片并抽取标注（可选补全测试病例的帧），不落 tar 本体。"""
    path = f"data/{remote}/parts/{tmpl.format(n=part_no)}"
    url = resolve_url(path)
    t0 = time.time()
    print(f"  [part {part_no:03d}] 开始下载 {part_bytes / 1e9:.2f} GB"
          f"（分块流式抽取：{workers} 并发预取 + 块级自动重试）", flush=True)
    reader = _RangeReader(url, total=part_bytes, workers=workers)

    class _Progress(io.RawIOBase):
        """包装 _RangeReader，每 30 秒打印进度与 ETA（tarfile 只按需读取）。"""

        def __init__(self, inner: _RangeReader):
            self.inner = inner
            self.t0 = time.time()
            self.last = 0.0

        def readable(self) -> bool:
            return True

        def readinto(self, b):  # type: ignore[override]
            n = self.inner.readinto(b)
            self._tick()
            return n

        def read(self, size=-1):  # type: ignore[override]
            data = self.inner.read(size)
            self._tick()
            return data

        def _tick(self) -> None:
            n = self.inner.n_read
            now = time.time()
            if n <= 0 or now - self.last < 30:
                return
            self.last = now
            el = now - self.t0
            speed = n / el / 1e6
            eta = (self.inner.total - n) / (n / el) if n else 0
            extra = f"（块重试 {self.inner.n_retry} 次）" if self.inner.n_retry else ""
            print(f"    [part {part_no:03d}] {n / 1e9:.2f}/{self.inner.total / 1e9:.2f} GB "
                  f"({n * 100 / max(1, self.inner.total):.0f}%) {speed:.2f} MB/s 剩余 ~{_fmt(eta)}{extra}",
                  flush=True)

    buffered = io.BufferedReader(_Progress(reader), buffer_size=1024 * 1024)
    n_lab, b_lab, n_img, b_img = extract_labels_from_stream(
        buffered, local_ds, needed_cases=needed_cases, save_images=save_images)
    got = reader.n_read
    dt = time.time() - t0
    print(f"  [part {part_no:03d}] 完成：读过 {got / 1e9:.2f} GB，落盘标注 {n_lab} 个 / {b_lab / 1e6:.1f} MB"
          + (f"，补帧 {n_img} 张 / {b_img / 1e6:.1f} MB" if save_images else "")
          + f"，耗时 {_fmt(dt)}（{got / max(dt, 1e-6) / 1e6:.2f} MB/s，块重试 {reader.n_retry} 次）",
          flush=True)
    return {"done": True, "files": n_lab, "label_bytes": b_lab,
            "images": n_img, "image_bytes": b_img,
            "read_bytes": got, "seconds": round(dt, 1), "chunk_retries": reader.n_retry,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def run_from_local_tar(local_path: Path, local_ds: str) -> dict:
    t0 = time.time()
    with tarfile.open(local_path, mode="r:") as tar:
        written = 0
        out_bytes = 0
        prefix = local_ds + "/label/"
        for member in tar:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            rel = member.name[len(local_ds) + 1:]
            dst = DATA_ROOT / local_ds / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            data = src.read()
            dst.write_bytes(data)
            written += 1
            out_bytes += len(data)
    print(f"  本地分片 {local_path.name}: 落盘标注 {written} 个 / {out_bytes / 1e6:.1f} MB，耗时 {_fmt(time.time() - t0)}",
          flush=True)
    return {"done": True, "files": written, "label_bytes": out_bytes, "source": "local"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="从 HF 数据集分片流式抽取 GT 标注")
    ap.add_argument("--datasets", nargs="*", default=None, help="本地数据集名（默认全部）")
    ap.add_argument("--list", action="store_true", help="只列出分片与状态")
    ap.add_argument("--from-local-part", default=None, help="从已下载的本地 tar 抽取（需配合 --datasets 指定单个数据集）")
    ap.add_argument("--with-images", action="store_true",
                    help="同时补全**测试病例**的完整帧（本地 img 被截断时用；已存在的帧跳过）")
    ap.add_argument("--reset", action="store_true", help="清空断点状态")
    ap.add_argument("--reset-dataset", nargs="*", default=None,
                    help="清掉指定数据集的分片状态（用于换抽取模式后重跑）")
    ap.add_argument("--retry-rounds", type=int, default=3,
                    help="失败分片的重试轮数（默认 3；每轮重扫未完成分片）")
    ap.add_argument("--retry-pause", type=int, default=120,
                    help="轮次之间的等待秒数（默认 120）")
    ap.add_argument("--workers", type=int, default=4,
                    help="分块并发预取线程数（默认 4；实测单流 0.85 MB/s、4 并发 1.41 MB/s）")
    args = ap.parse_args()

    state = {} if args.reset else load_state()
    for ds in (args.reset_dataset or []):
        if ds in state:
            state.pop(ds)
            print(f"[reset] 已清除 {ds} 的分片状态，将重新下载")
    if args.reset_dataset:
        save_state(state)
    targets = args.datasets or list(DATASETS)
    needed_by_ds = load_needed_cases() if args.with_images else {}

    if args.from_local_part:
        local_ds = targets[0]
        info = run_from_local_tar(Path(args.from_local_part), local_ds)
        state.setdefault(local_ds, {})["local_" + Path(args.from_local_part).stem] = info
        save_state(state)
        return 0

    def collect_todo() -> list[tuple[str, str, str, int, int]]:
        items: list[tuple[str, str, str, int, int]] = []
        for local_ds in targets:
            remote, tmpl = DATASETS[local_ds]
            for part_no, part_bytes, cases in fetch_part_manifest(remote):
                done = state.get(local_ds, {}).get(f"part_{part_no:03d}", {}).get("done", False)
                if args.list:
                    print(f"  {local_ds:<38} part {part_no:03d}  {part_bytes / 1e9:.2f} GB  {cases} 病例  "
                          f"[{'已完成' if done else '待下载'}]")
                if not done:
                    items.append((local_ds, remote, tmpl, part_no, part_bytes))
        return items

    if args.list:
        todo = collect_todo()
        total = sum(t[4] for t in todo)
        print(f"\n待下载 {len(todo)} 个分片，合计 {total / 1e9:.1f} GB（按 2.5 MB/s 估计 ~{_fmt(total / 2.5e6)}）")
        return 0

    for round_no in range(1, max(1, args.retry_rounds) + 1):
        todo = collect_todo()
        if not todo:
            break
        total_bytes = sum(t[4] for t in todo)
        print(f"\n=== 第 {round_no}/{args.retry_rounds} 轮：待下载 {len(todo)} 个分片 / {total_bytes / 1e9:.1f} GB"
              + ("（同时补全测试病例的完整帧）" if args.with_images else "") + " ===", flush=True)
        for local_ds, remote, tmpl, part_no, part_bytes in todo:
            try:
                info = run_part(local_ds, remote, tmpl, part_no, part_bytes,
                                needed_cases=needed_by_ds.get(local_ds, set()),
                                save_images=args.with_images, workers=args.workers)
            except Exception as exc:  # noqa: BLE001
                print(f"  [part {part_no:03d}] 失败：{type(exc).__name__}: {str(exc)[:200]}", flush=True)
                continue
            state.setdefault(local_ds, {})[f"part_{part_no:03d}"] = info
            save_state(state)
        if round_no < args.retry_rounds and collect_todo():
            print(f"  本轮有失败分片，{args.retry_pause}s 后进入下一轮重试", flush=True)
            time.sleep(args.retry_pause)

    print("\n各数据集覆盖率：", flush=True)
    sched = load_needed_cases(with_frames=True)
    for local_ds in DATASETS:
        lab = DATA_ROOT / local_ds / "label"
        img = DATA_ROOT / local_ds / "img"
        n_cases = len(list(lab.iterdir())) if lab.exists() else 0
        n_files = sum(1 for _ in lab.rglob("*.png")) if lab.exists() else 0
        want, have = sched.get(local_ds, (0, 0))
        print(f"  {local_ds:<38} 标注 {n_cases:4d} 病例 / {n_files:6d} 文件；"
              f"测试病例帧 {have}/{want}" + ("（齐全）" if want and have >= want else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
