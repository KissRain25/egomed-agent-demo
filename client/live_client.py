#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EgoMed-Agent 实时采集客户端（N4，子计划 docs/子计划_N4_摄像头实时采集.md）

把系统从"选文件发帧"升级为"摄像头实时画面 → 说话 → 实时分割回显"。

职责（对应端到端流程设计：眼镜端零智能）：
  采集 → 抽帧 → 压缩 → 发帧（后台线程）→ 收 overlay → 回显 + 防抖 → 落盘报告

设计要点：
  - FrameSource 抽象：摄像头 / 视频文件 / （占位）眼镜，主循环只认 read()
  - 发帧节奏与采集解耦：默认 5 fps（服务端 200~500ms/帧，发更快只会被 latest-wins 覆盖）
  - 网络在后台线程、并发=1、新帧覆盖旧帧（与契约"丢帧语义"一致）
  - 坏帧判定以服务端为准（overlay 为空）；空 overlay 时沿用上次显示，画面不闪跳
  - 复用 mock_client 的契约常量与 post_frame/make_session_id，不重复造轮子

用法见 docs/子计划_N4_摄像头实时采集.md 第八节。
"""
from __future__ import annotations

import argparse
import base64
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import mock_client as mc          # 复用：契约常量 / post_frame / make_session_id / natural_key

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 opencv-python，请先执行：pip install opencv-python")

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

DEFAULT_FPS = 5.0            # 发帧节奏（契约：丢帧语义，不追高帧率）
DEFAULT_SOURCE_FPS = 20.0    # 视频文件源的播放节奏（模拟摄像头帧率，让指标有意义）
DEFAULT_OUT = PROJECT_ROOT / "runs" / "live"
SAMPLE_EVERY = 10            # 每发送 N 帧落盘一组抽样帧（原帧 + 渲染帧）
LAT_SAMPLES = 500            # 延迟样本上限

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
]


# ===========================================================================
# 一、采集源抽象（换硬件只改这里）
# ===========================================================================
class FrameSource:
    """采集源抽象：主循环只依赖 read()，眼镜到位后新增 GlassesSource 即可。"""

    name = "base"

    def open(self) -> None:
        pass

    def read(self):
        """返回 (ok, frame)。ok=False 表示源结束/不可用；frame 为 None 表示跳过该帧。"""
        raise NotImplementedError

    def release(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return getattr(self, "_fps", 0.0)


class VideoFileSource(FrameSource):
    """视频文件或帧目录作为采集源（M1 自动化用；模拟"眼镜摄像头"的输入）"""

    name = "video"

    def __init__(self, path: str, loop: bool = True, fps: float = DEFAULT_SOURCE_FPS):
        self.path = Path(path)
        self.loop = loop
        self._fps = fps
        self._cap = None
        self._files: list[Path] = []
        self._idx = 0

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"输入不存在：{self.path}")
        if self.path.is_dir():
            self._files = sorted(
                [p for p in self.path.iterdir() if p.suffix.lower() in mc.IMG_EXTS],
                key=mc.natural_key,
            )
            if not self._files:
                raise RuntimeError(f"目录里没有图片（支持 {sorted(mc.IMG_EXTS)}）：{self.path}")
            mc.log(f"[source] 帧目录源就绪：{self.path}（{len(self._files)} 帧，循环={self.loop}）")
        else:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"打不开视频文件：{self.path}")
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self._fps = min(fps, 60.0)
            mc.log(f"[source] 视频文件源就绪：{self.path}（{self._fps:.1f} fps）")

    def read(self):
        if self._files:
            if self._idx >= len(self._files):
                if not self.loop:
                    return False, None
                self._idx = 0
            img = cv2.imread(str(self._files[self._idx]))
            self._idx += 1
            return True, img          # img 可能为 None（坏文件），外层跳过
        ok, frame = self._cap.read()
        if not ok and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return ok, (frame if ok else None)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class WebcamSource(FrameSource):
    """真实摄像头（M2/M3）：OpenCV 轮询拉帧"""

    name = "cam"

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index = index
        self.width = width
        self.height = height
        self._cap = None
        self._fps = 0.0

    def open(self) -> None:
        for backend, label in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_ANY, "ANY")):
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self._cap = cap
                    mc.log(f"[source] 摄像头 {self.index} 打开成功（后端 {label}）")
                    break
            cap.release()
        if self._cap is None:
            raise RuntimeError(
                f"打不开摄像头 {self.index}：可能被其他程序占用（会议/相机应用）；"
                f"可试 --index 1、2，或先关闭占用程序"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        real_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        mc.log(f"[source] 实际分辨率 {int(real_w)}x{int(real_h)}，驱动帧率 {self._fps:.1f} fps")

    def read(self):
        ok, frame = self._cap.read()
        return ok, (frame if ok else None)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class GlassesSource(FrameSource):
    """眼镜源占位：真机 SDK 到位后实现（回调 → 内部队列 → read()）。

    迁移只需实现本类，主循环、发帧、回显逻辑一行不改（决策记录 D2/D7）。
    """

    name = "glasses"

    def open(self) -> None:
        raise NotImplementedError(
            "眼镜 SDK 未接入。真机到位后在此实现：SDK 帧回调 → queue.Queue → read() 弹出"
        )


def build_source(args) -> FrameSource:
    if args.source == "cam":
        return WebcamSource(args.index, args.width, args.height)
    return VideoFileSource(args.input, loop=not args.no_loop, fps=args.source_fps)


# ===========================================================================
# 二、压缩与图像处理（契约规格，与 mock_client 保持一致）
# ===========================================================================
def compress_frame(frame, max_side: int = mc.MAX_SIDE, quality: int = mc.JPEG_QUALITY,
                   max_bytes: int = mc.MAX_BYTES):
    """内存帧 → jpg bytes（长边 ≤1280、质量 80、≤300KB，逐级降质量保证发得出去）。"""
    h, w = frame.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    data = b""
    used_q = quality
    for q in (quality, 70, 60, 50, 40):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            continue
        data = buf.tobytes()
        used_q = q
        if len(data) <= max_bytes:
            break
    return data, used_q


def decode_overlay(b64: str):
    """base64 字符串 → BGR 图；失败返回 None（不抛，保证主循环不崩）"""
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def fit_to(img, w: int, h: int):
    if img is None:
        return None
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


class Hud:
    """左上角状态文字（中文用 PIL 绘制，OpenCV putText 不支持中文）"""

    def __init__(self):
        self._font = None
        if _HAS_PIL:
            for path in _FONT_CANDIDATES:
                if Path(path).exists():
                    try:
                        self._font = ImageFont.truetype(path, 20)
                        break
                    except Exception:
                        continue

    def draw(self, frame, lines):
        if not lines:
            return frame
        if self._font is None:
            # 退化为英文（无中文字体时）
            for i, line in enumerate(lines):
                cv2.putText(frame, line.encode("ascii", "ignore").decode() or "-",
                            (12, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame
        img = Image.fromarray(frame[:, :, ::-1])       # BGR → RGB
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            y = 12 + i * 26
            draw.rectangle([8, y - 2, 8 + 12 * len(line), y + 22], fill=(0, 0, 0))
            draw.text((12, y), line, font=self._font, fill=(0, 255, 0))
        return np.array(img)[:, :, ::-1]               # RGB → BGR


# ===========================================================================
# 三、统计
# ===========================================================================
class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.sent = 0            # 提交给发送线程的帧数
        self.submitted = 0
        self.ok = 0              # HTTP 成功且业务 ok
        self.http_err = 0
        self.biz_err = 0
        self.bad_frame = 0       # 服务端判定坏帧（overlay 为空）
        self.no_target = 0       # 会话未设目标（服务端提示"请先告诉我要分割哪个器官"）
        self.set_target_text = ""    # A2 返回的识别文本
        self.set_target_target = ""  # A2 解析出的目标
        self.dropped = 0         # 被新帧覆盖而丢弃
        self.overlay_nonempty = 0
        self.latencies: list[float] = []
        self.errors: list[str] = []
        self.last_msg = ""
        self.last_target = ""
        self.last_modality = ""
        self.first_ok_at: float | None = None

    def add_latency(self, ms: float) -> None:
        self.latencies.append(ms)
        if len(self.latencies) > LAT_SAMPLES:
            del self.latencies[:len(self.latencies) - LAT_SAMPLES]

    def pct(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        data = sorted(self.latencies)
        k = max(0, min(len(data) - 1, int(round((p / 100.0) * (len(data) - 1)))))
        return data[k]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.latencies) if self.latencies else 0.0


# ===========================================================================
# 四、发送线程（并发=1，新帧覆盖旧帧）
# ===========================================================================
class FramePump:
    def __init__(self, url: str, session_id: str, timeout: float, stats: Stats,
                 on_result):
        self.url = url
        self.session = session_id
        self.timeout = timeout
        self.stats = stats
        self.on_result = on_result
        self._lock = threading.Lock()
        self._busy = False
        self._pending = None      # (frame, index)

    def submit(self, frame, index: int) -> None:
        """提交一帧；若发送线程正忙，则只保留最新一帧（覆盖）。"""
        with self._lock:
            self.stats.submitted += 1
            if self._busy:
                if self._pending is not None:
                    self.stats.dropped += 1
                self._pending = (frame, index)
                return
            self._busy = True
        threading.Thread(target=self._worker, args=(frame, index), daemon=True).start()

    def _worker(self, frame, index: int) -> None:
        while True:
            t0 = time.perf_counter()
            try:
                jpg, _q = compress_frame(frame)
            except Exception as exc:  # pragma: no cover
                self.stats.errors.append(f"压缩失败：{exc}")
                self._finish()
                return
            payload, elapsed, err = mc.post_frame(
                self.url, self.session, jpg, f"live_{index:05d}.jpg", self.timeout)
            self.stats.add_latency(elapsed)
            with self._lock:
                self.stats.sent += 1
                if self.stats.first_ok_at is None and err is None:
                    self.stats.first_ok_at = time.perf_counter() - t0
            try:
                self.on_result(payload, elapsed, err, index, jpg)
            except Exception as exc:  # pragma: no cover
                self.stats.errors.append(f"结果处理异常：{exc}")

            with self._lock:
                pending = self._pending
                self._pending = None
                if pending is None:
                    self._busy = False
                    return
                frame, index = pending      # 立刻发最新帧（旧帧已被覆盖）

    def _finish(self) -> None:
        with self._lock:
            self._busy = False


# ===========================================================================
# 五、主流程
# ===========================================================================
class LiveRunner:
    def __init__(self, args):
        self.args = args
        self.stats = Stats()
        self.hud = Hud()
        self.last_result = None          # 最近一次有效 overlay 图（防抖：空 overlay 时沿用）
        self.last_message = ""
        self._lock = threading.Lock()
        self.stop = False
        self.out_dir = Path(args.out) if args.out else (
            DEFAULT_OUT / datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.samples_dir = self.out_dir / "samples"
        self.session_id = args.session or mc.make_session_id(args.doctor)
        self.pump = FramePump(args.url, self.session_id, args.timeout, self.stats,
                              self._on_result)

    # ---------------------------------------------------------------- 结果回调
    def _on_result(self, payload, elapsed, err, index, jpg) -> None:
        with self._lock:
            if err is not None:
                self.stats.http_err += 1
                if len(self.stats.errors) < 20:
                    self.stats.errors.append(f"#{index} {err}")
                return
            if not isinstance(payload, dict):
                self.stats.biz_err += 1
                return
            if payload.get("status") != "ok":
                self.stats.biz_err += 1
                self.last_message = str(payload.get("message") or "")
                return

            overlay_b64 = payload.get("overlay") or ""
            self.last_message = str(payload.get("message") or "")
            self.stats.last_target = str(payload.get("target") or "")
            self.stats.last_modality = str(payload.get("modality") or "")
            if not overlay_b64 and "请先告诉我" in self.last_message:
                # 会话还没设目标：不是坏帧，是"没说指令"
                self.stats.no_target += 1
                self.stats.ok += 1
                return
            if overlay_b64:
                img = decode_overlay(overlay_b64)
                if img is not None:
                    self.last_result = img
                    self.stats.overlay_nonempty += 1
                    self.stats.ok += 1
                    return
            # 坏帧：服务端返空 overlay（P0.3 语义）→ 不动 last_result（画面不闪跳）
            self.stats.bad_frame += 1
            self.stats.ok += 1

    # ---------------------------------------------------------------- 设置目标（A2）
    def _set_target_from_audio(self, wav_path: Path) -> bool:
        """发一段语音设置会话目标（A2），返回是否成功。"""
        if not wav_path.exists():
            mc.log(f"[A2] 音频不存在，跳过设目标：{wav_path}")
            return False
        wav = wav_path.read_bytes()
        payload, elapsed, err = mc.post_audio(
            self.args.url, self.session_id, wav, wav_path.name, self.args.timeout)
        if err:
            mc.log(f"[A2] 设置目标失败：{err}")
            return False
        text = str(payload.get("text") or "")
        target = payload.get("target") or payload.get("targets") or ""
        self.stats.set_target_text = text
        self.stats.set_target_target = str(target)
        mc.log(f"[A2] 语音「{text}」→ 目标 {target}（{elapsed:.0f}ms）")
        if isinstance(payload.get("message"), str) and payload.get("message"):
            self.last_message = payload["message"]
        return bool(target)

    # ---------------------------------------------------------------- 渲染
    def _render(self, frame):
        with self._lock:
            last = self.last_result
            msg = self.last_message
        h, w = frame.shape[:2]
        if last is not None:
            display = fit_to(last, w, h).copy()      # 显示服务器返回的结果图（已含分割标注）
            tag = "分割结果（最近一次有效）"
        else:
            display = frame.copy()
            tag = "实时画面（暂无分割结果）"
        if self.args.display == "blend" and last is not None:
            display = cv2.addWeighted(frame, 0.45, fit_to(last, w, h), 0.55, 0)

        st = self.stats
        lines = [
            f"会话 {self.session_id}",
            f"采集帧 {self.frames_read}  发帧 {st.sent}（提交 {st.submitted}/丢 {st.dropped}）",
            f"延迟 均值 {st.mean:.0f}ms / P95 {st.pct(95):.0f}ms",
            f"模态 {st.last_modality or '-'}  目标 {st.last_target or '-'}",
            f"{tag}",
        ]
        if st.no_target and not st.set_target_target:
            lines.append("尚未设置目标：请说话（--audio 或 --mic）")
        if msg:
            lines.append(f"提示 {msg}")
        return self.hud.draw(display, lines)

    # ---------------------------------------------------------------- 主循环
    def run(self) -> int:
        args = self.args
        source = build_source(args)
        source.open()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_frames:
            self.samples_dir.mkdir(parents=True, exist_ok=True)

        mc.log(f"[main] 会话 {self.session_id} → {args.url}"
               f"（发帧 {args.fps:g} fps，显示={'开' if args.show else '关'}）")

        # 先设目标：契约语义是"先说话（A2）→ 再发帧（A1）"，否则 A1 会提示"请先告诉我要分割哪个器官"
        if args.audio:
            self._set_target_from_audio(Path(args.audio))

        loop_interval = 1.0 / max(1.0, min(source.fps or args.source_fps, 60.0))
        send_interval = 1.0 / max(0.1, args.fps)
        self.frames_read = 0
        next_send = time.perf_counter()
        next_loop = time.perf_counter()
        t_start = time.perf_counter()
        saved_groups = 0
        exit_reason = "正常结束"

        try:
            while not self.stop:
                ok, frame = source.read()
                if not ok:
                    exit_reason = "采集源结束"
                    break
                if frame is None:
                    continue
                self.frames_read += 1
                now = time.perf_counter()

                # 抽帧发送（节奏由 --fps 决定）
                if now >= next_send:
                    self.pump.submit(frame, self.frames_read)
                    next_send = now + send_interval

                # 回显（每帧都刷新，显示刷新率 = 采集节奏）
                if args.show:
                    cv2.imshow("EgoMed-Agent Live", self._render(frame))
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        exit_reason = "用户按键退出"
                        break

                # 抽样落盘（供人工/自动化读图验收）
                if args.save_frames and self.stats.sent > 0 \
                        and self.stats.sent // SAMPLE_EVERY > saved_groups:
                    saved_groups += 1
                    cv2.imwrite(str(self.samples_dir / f"g{saved_groups:03d}_raw.jpg"), frame)
                    cv2.imwrite(str(self.samples_dir / f"g{saved_groups:03d}_display.jpg"),
                                self._render(frame))

                if args.max_frames and self.stats.sent >= args.max_frames:
                    exit_reason = f"达到 --max-frames {args.max_frames}"
                    break
                if args.duration and (now - t_start) >= args.duration:
                    exit_reason = f"达到 --duration {args.duration:g}s"
                    break

                # 节奏控制：视频源按源帧率播放，避免"跑得比摄像头还快"导致指标失真
                next_loop += loop_interval
                sleep = next_loop - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_loop = time.perf_counter()
        finally:
            source.release()
            if args.show:
                cv2.destroyAllWindows()
            # 等发送线程收尾
            deadline = time.perf_counter() + max(3.0, args.timeout + 1)
            while self.stats.sent < self.stats.submitted and time.perf_counter() < deadline:
                time.sleep(0.1)

        elapsed = time.perf_counter() - t_start
        return self.report(elapsed, exit_reason)

    # ---------------------------------------------------------------- 报告
    def report(self, elapsed: float, exit_reason: str) -> int:
        st = self.stats
        send_fps = st.sent / elapsed if elapsed > 0 else 0.0
        display_fps = self.frames_read / elapsed if elapsed > 0 else 0.0
        lat_ok = st.pct(95) < 3000.0 or st.sent == 0
        fps_ok = send_fps >= self.args.fps * 0.8
        err_ok = (st.http_err + st.biz_err) == 0

        checks = [
            ("M1 有效发帧率 ≥ 80% 目标", f"{send_fps:.2f} / 目标 {self.args.fps:g} fps", fps_ok),
            ("M1 无网络/业务错误", f"HTTP 错 {st.http_err} / 业务错 {st.biz_err}", err_ok),
            ("M2 延迟 P95 < 3000ms", f"P95 {st.pct(95):.0f}ms（均值 {st.mean:.0f}ms）", lat_ok),
            ("M5 抽样帧落盘", f"{len(list(self.samples_dir.glob('*.jpg'))) if self.samples_dir.exists() else 0} 张",
             (not self.args.save_frames) or self.samples_dir.exists()),
        ]
        if self.args.expect_overlay:
            checks.append(("M1 出现有效分割结果（overlay 非空）",
                           f"overlay 非空 {st.overlay_nonempty} / 坏帧 {st.bad_frame} / 未设目标 {st.no_target}",
                           st.overlay_nonempty > 0))

        lines = [
            "# N4 实时采集验收报告",
            "",
            f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 会话：`{self.session_id}` ｜ 服务端：`{self.args.url}`",
            f"- 采集源：`{self.args.source}`"
            + (f"（`{self.args.input}`）" if self.args.source != "cam" else f"（index {self.args.index}）"),
            f"- 运行：{elapsed:.1f}s，结束原因：{exit_reason}",
            f"- 发帧节奏：{self.args.fps:g} fps ｜ 显示：{'开' if self.args.show else '关（无头）'}",
            "",
            "## 指标",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 采集帧数 | {self.frames_read} |",
            f"| 提交发帧 | {st.submitted} |",
            f"| 实际发送 | {st.sent}（被新帧覆盖丢弃 {st.dropped}） |",
            f"| 有效发帧率 | {send_fps:.2f} fps |",
            f"| 显示帧率 | {display_fps:.2f} fps |",
            f"| 延迟 均值/P50/P95 | {st.mean:.0f} / {st.pct(50):.0f} / {st.pct(95):.0f} ms |",
            f"| overlay 非空（有分割结果） | {st.overlay_nonempty} |",
            f"| 坏帧（overlay 空） | {st.bad_frame} |",
            f"| 未设目标（会话未收到语音指令） | {st.no_target} |",
            f"| A2 设目标 | 文本「{st.set_target_text or '-'}」→ `{st.set_target_target or '-'}` |",
            f"| 网络错误 / 业务错误 | {st.http_err} / {st.biz_err} |",
            f"| 末次模态 / 目标 | `{st.last_modality or '-'}` / `{st.last_target or '-'}` |",
            "",
            "## 判定",
            "",
            "| 检查项 | 实测 | 结论 |",
            "|--------|------|------|",
        ]
        for name, detail, ok in checks:
            lines.append(f"| {name} | {detail} | {'✅ PASS' if ok else '❌ FAIL'} |")

        lines += ["", "## 抽样帧（供读图确认叠加/防抖行为）", ""]
        if self.samples_dir.exists():
            for p in sorted(self.samples_dir.glob("*.jpg"))[:40]:
                lines.append(f"- `{p.relative_to(PROJECT_ROOT)}`")
        else:
            lines.append("- （未开启 --save-frames）")

        if st.errors:
            lines += ["", "## 错误样本（前 20 条）", ""]
            lines += [f"- {e}" for e in st.errors[:20]]

        self.out_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.out_dir / "report.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        mc.log("=" * 74)
        for name, detail, ok in checks:
            mc.log(f"  [{'PASS' if ok else 'FAIL'}] {name} —— {detail}")
        mc.log(f"  采集 {self.frames_read} 帧，发送 {st.sent} 帧（丢 {st.dropped}），"
               f"有效 {send_fps:.2f} fps，延迟 均值 {st.mean:.0f}ms / P95 {st.pct(95):.0f}ms")
        mc.log(f"  画面：overlay 非空 {st.overlay_nonempty} / 坏帧 {st.bad_frame} / "
               f"错误 {st.http_err + st.biz_err}")
        mc.log(f"  报告：{report_path}")
        mc.log("=" * 74)
        all_ok = all(ok for _n, _d, ok in checks)
        return 0 if all_ok else 1


# ===========================================================================
# 六、设备探测（不写任何文件，只看"能不能出帧"）
# ===========================================================================
def probe_cameras(max_index: int = 4) -> int:
    mc.log("=" * 74)
    mc.log("摄像头探测（只报告设备与画面统计，不保存图像）")
    found = 0
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            mc.log(f"  设备 {idx}: 打不开")
            cap.release()
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            mc.log(f"  设备 {idx}: 打开了但读不到帧（可能被占用）")
            cap.release()
            continue
        found += 1
        h, w = frame.shape[:2]
        bright = float(frame.mean())
        mc.log(f"  设备 {idx}: OK  {w}x{h}  平均亮度 {bright:.1f}"
               f"{'（偏暗：注意补光）' if bright < 30 else ''}")
        cap.release()
    if not found:
        mc.log("  未发现可用摄像头（检查隐私设置/占用程序）")
    mc.log("=" * 74)
    return 0 if found else 1


# ===========================================================================
# 七、CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="EgoMed-Agent 实时采集客户端（N4）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  探测设备:   python client/live_client.py --probe\n"
            "  自动化验收: python client/live_client.py --source video "
            "--input data/samples/acdc --max-frames 60 --no-show --save-frames\n"
            "  真实摄像头: python client/live_client.py --source cam --fps 5 --save-frames\n"
        ))
    p.add_argument("--source", choices=["video", "cam", "glasses"], default="video",
                   help="采集源：video=视频/帧目录（自动化），cam=摄像头（默认 video）")
    p.add_argument("--input", default="data/samples/acdc", help="video 源的输入路径（mp4 或帧目录）")
    p.add_argument("--audio", default="", help="启动时先发这段 wav 设置分割目标（A2，契约语义：先说后看）")
    p.add_argument("--expect-overlay", action="store_true",
                   help="要求必须出现有效 overlay（否则 FAIL）——用于验证'说完话能出分割结果'")
    p.add_argument("--index", type=int, default=0, help="摄像头序号（cam 源）")
    p.add_argument("--width", type=int, default=1280, help="请求采集宽度")
    p.add_argument("--height", type=int, default=720, help="请求采集高度")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="发帧节奏（默认 5）")
    p.add_argument("--source-fps", type=float, default=DEFAULT_SOURCE_FPS,
                   help="视频源播放节奏（模拟摄像头帧率，默认 20）")
    p.add_argument("--no-loop", action="store_true", help="视频源播完不循环")
    p.add_argument("--display", choices=["result", "blend"], default="result",
                   help="显示方式：result=显示服务器结果图；blend=与实时画面叠加")
    p.add_argument("--no-show", dest="show", action="store_false", help="无头模式（不弹窗）")
    p.add_argument("--save-frames", action="store_true", help="落盘抽样帧（每 10 个已发帧一组）")
    p.add_argument("--max-frames", type=int, default=0, help="发送多少帧后停止（0=不限）")
    p.add_argument("--duration", type=float, default=0.0, help="运行多少秒后停止（0=不限）")
    p.add_argument("--url", default=mc.DEFAULT_URL, help=f"服务端地址（默认 {mc.DEFAULT_URL}）")
    p.add_argument("--session", default="", help="指定 session_id（默认自动生成）")
    p.add_argument("--doctor", default="live", help="医生编号（用于生成 session_id）")
    p.add_argument("--timeout", type=float, default=mc.TIMEOUT, help="单请求超时（秒）")
    p.add_argument("--out", default="", help="输出目录（默认 runs/live/<时间戳>）")
    p.add_argument("--probe", action="store_true", help="只探测摄像头设备后退出")
    p.set_defaults(show=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.probe:
        return probe_cameras()
    if args.source == "glasses":
        mc.log("[main] glasses 源未实现（眼镜 SDK 到位后补 GlassesSource）")
        return 2
    runner = LiveRunner(args)
    try:
        return runner.run()
    except KeyboardInterrupt:
        mc.log("\n[main] 手动中断，输出部分统计")
        return runner.report(0.0, "用户中断")


if __name__ == "__main__":
    sys.exit(main())
