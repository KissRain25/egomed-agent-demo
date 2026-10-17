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
import io
import json
import queue
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
        p = Path(path)
        # 相对路径按项目根解析：无论当前工作目录在哪（例如从 C:\ 下调用）都能找到
        self.path = p if p.is_absolute() else (PROJECT_ROOT / p)
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


class NetworkStreamSource(FrameSource):
    """网络视频流采集源（手机当"模拟眼镜"摄像头）

    典型用法：手机装 IP Webcam（Android）等 app 开 MJPEG 服务，
    手机与电脑连**同一 WiFi**，然后：

        python client/live_client.py --source url --input http://192.168.1.23:8080/video --mic

    为什么要它：笔记本自带摄像头多为 640x480 且位置固定；手机分辨率高、
    可手持移动，更接近"医生戴镜走动"的第一人称视角（`端到端流程设计.md` 的模拟策略）。
    """

    name = "url"

    def __init__(self, url: str, fps: float = DEFAULT_SOURCE_FPS, max_retry: int = 3):
        self.url = url if url.startswith(("http://", "https://", "rtsp://", "rtmp://")) \
            else f"http://{url}"
        self._fps = fps
        self._cap = None
        self.max_retry = max_retry       # 断流后最多重试几次（每次退避 0.4~2s）
        self.reconnects = 0              # 成功重连次数（写进报告，便于发现手机端抖动）
        self.open_timeout_ms = 2000      # 打开超时（不设的话死地址要 ~20s 才失败）
        self.read_timeout_ms = 3000      # 读取超时

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _try_open(self, quiet: bool) -> bool:
        """按 FFMPEG → ANY 顺序尝试打开并读到第一帧。

        带打开/读取超时（实测不设超时时，死地址要 ~20s 才失败 → 重试几次就像"卡死"）。
        """
        params = [int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC), self.open_timeout_ms,
                  int(cv2.CAP_PROP_READ_TIMEOUT_MSEC), self.read_timeout_ms]
        for backend, label in ((cv2.CAP_FFMPEG, "FFMPEG"), (cv2.CAP_ANY, "ANY")):
            try:
                cap = cv2.VideoCapture(self.url, backend, params)
            except Exception:                     # 个别后端不认 params，退回不带超时
                try:
                    cap = cv2.VideoCapture(self.url, backend)
                except Exception:
                    continue
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    self._cap = cap
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps and fps > 0:
                        self._fps = min(fps, 60.0)
                    h, w = frame.shape[:2]
                    if not quiet:
                        mc.log(f"[source] 网络流就绪（后端 {label}）：{w}x{h}，"
                               f"驱动帧率 {self._fps:.1f}")
                    return True
            cap.release()
        return False

    def open(self) -> None:
        if not self._try_open(quiet=False):
            raise RuntimeError(
                f"打不开网络视频流：{self.url}\n"
                f"  排查：① 手机和电脑是否同一 WiFi ② 地址是否正确（浏览器能否打开它）"
                f" ③ Windows 防火墙是否允许 ④ app 里的服务是否已启动"
            )

    def read(self):
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if ok and frame is not None:
            return True, frame
        # N4.5 现场暴露：MJPEG 流会偶发断一次（手机 app 切后台/重连、WiFi 抖动），
        # 而 cv2 不会自己重连 → 之前整场采集就此结束（其实流还在）。
        # 这里按退避策略自动重连，避免一次抖动毁掉整场演示。
        return self._reconnect()

    def _reconnect(self):
        for attempt in range(1, self.max_retry + 1):
            self._close()
            time.sleep(min(0.4 * attempt, 2.0))
            if not self._try_open(quiet=True):
                continue
            for _ in range(3):                # 重连后缓冲要一点时间
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    self.reconnects += 1
                    mc.log(f"[source] 网络流已重连（第 {attempt} 次尝试，"
                           f"累计重连 {self.reconnects} 次）")
                    return True, frame
                time.sleep(0.2)
        mc.log(f"[source] 网络流重连失败（连续 {self.max_retry} 次尝试），结束采集")
        self._close()
        return False, None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# 眼镜接入层（N34）：SDK 适配器 + 有界队列，真机到位只写一个 Adapter
# ---------------------------------------------------------------------------
DEFAULT_GLASSES_REPLAY = PROJECT_ROOT / "data" / "samples" / "acdc"
GLASSES_QUEUE_SIZE = 2          # 队列满则丢最旧（latest-wins，避免延迟累积）


# ---------------------------------------------------------------- Rokid 通道辅助
A2_SAMPLERATE = 16000            # A2 契约：wav 16 kHz 单声道
A2_CHANNELS = 1


def nv21_to_bgr(data, width: int, height: int):
    """NV21（YUV420SP：Y 平面 + 交错 VU）→ BGR ndarray；无法解码时返回 None。

    **为什么需要这一层**：Rokid 眼镜 SDK 的画面回调给的是 **NV21 原始 YUV**
    （官方示例即 `IMessageListener.onNv21Data()`），不是 RGB/JPG。
    我们内部（FrameSource / 压缩 / Renderer）统一约定 BGR —— 所以在这里一次转干净，
    **SDK 的像素格式不外泄到主循环**（换厂商时只改适配层）。
    """
    if data is None or width is None or height is None:
        return None
    if width < 2 or height < 2 or width % 2 or height % 2:
        return None                      # YUV420 按 2×2 采样：奇数宽高必然错位
    need = width * height * 3 // 2
    if len(data) < need:
        return None
    raw = np.frombuffer(data, dtype=np.uint8, count=need)
    yuv = raw.reshape((height * 3 // 2, width))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)


def bgr_to_nv21(img) -> bytes:            # noqa: ANN001
    """BGR ndarray → NV21 字节流（与 `nv21_to_bgr` 对称）。

    用途：**样例生成与自测**（把现有样例帧转成 NV21，用来验证接入层）；
    真机链路用不到它（相机自己就出 NV21）。
    格式要点：NV21 = Y 平面 + 交错 **VU** —— 顺序写成 UV 就是 NV12，
    **U/V 搞反会让红蓝互换**（这类接入最经典的 bug，自测里专门断言了它）。
    """
    h, w = img.shape[:2]
    if h % 2 or w % 2:                    # YUV420 按 2×2 采样 → 先裁成偶数
        img = img[:h - h % 2, :w - w % 2]
        h, w = img.shape[:2]
    flat = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y = flat[:h * w]
    u = flat[h * w:h * w + h * w // 4]
    v = flat[h * w + h * w // 4:]
    vu = np.empty(u.size * 2, dtype=np.uint8)
    vu[0::2] = v
    vu[1::2] = u
    return np.concatenate([y, vu]).tobytes()


def check_audio_spec(samplerate: int, channels: int) -> tuple[bool, str]:
    """校验 SDK 给的音频是否符合 A2 契约（16 kHz 单声道）；返回 (是否可直接用, 说明)。

    Rokid 的音频走**经典蓝牙**通道，采样率/声道**不由我们决定** ——
    所以真机联调第一件事就是断言这个，不符合要**显式**处理（重采样/降混），
    而不是让它静默变成"低质 ASR → 频繁拒识"（那样会误判成 N37 门槛太严）。
    """
    if not samplerate or not channels:
        return False, "采样率/声道缺失（SDK 未给出）"
    if channels > 2:
        return False, f"{channels} 声道：超出可处理范围，需 SDK 侧改配置"
    notes = []
    if channels != A2_CHANNELS:
        notes.append(f"{channels} ch → 需降混为单声道")
    if samplerate != A2_SAMPLERATE:
        notes.append(f"{samplerate} Hz → 需重采样到 {A2_SAMPLERATE} Hz")
    return (not notes), ("；".join(notes) if notes else
                         f"符合 A2 契约（{A2_SAMPLERATE} Hz 单声道）")


class GlassesAdapter:
    """眼镜 SDK 适配器基类 —— **真机接入时只写这一个子类**。

    约定（对齐决策记录 D2/D7：眼镜端零智能）：
      · 只负责"把 SDK 推来的帧交给回调"，不做推理、不改分辨率
      · 帧格式：BGR 的 numpy.ndarray（与 OpenCV 一致）
      · 回调可能在 SDK 自己的线程里触发 → 本层用队列与主循环解耦

    真机接入示例：
        class MyGlassesAdapter(GlassesAdapter):
            def open(self, on_frame):
                self._sdk = MyVendorSDK()
                self._sdk.set_frame_callback(on_frame)   # 回调式推流
                self._sdk.start_preview()
            def close(self):
                self._sdk.stop_preview()
    """

    name = "base"

    def open(self, on_frame) -> None:      # noqa: ANN001
        raise NotImplementedError

    def close(self) -> None:
        pass


class ReplayAdapter(GlassesAdapter):
    """无硬件时的替代适配器：按 fps 把样例帧目录"推"进回调。

    用途：在**没有眼镜**的情况下把整条链路（队列 → 丢帧 → 发帧 → 回显）跑通，
    与真机共享同一套主循环代码；也让 `--source glasses` 今天就能自测。
    """

    name = "replay"

    def __init__(self, root, fps: float = DEFAULT_SOURCE_FPS, loop: bool = True):
        self.root = Path(root)
        self.fps = max(1.0, float(fps))
        self.loop = loop
        self._stop = threading.Event()

    def open(self, on_frame) -> None:      # noqa: ANN001
        threading.Thread(target=self._run, args=(on_frame,), daemon=True).start()

    def _run(self, on_frame) -> None:      # noqa: ANN001
        files = ([p for p in sorted(self.root.iterdir(), key=mc.natural_key)
                  if p.suffix.lower() in mc.IMG_EXTS] if self.root.is_dir() else [])
        if not files:
            mc.log(f"[glasses/replay] 目录里没有图片：{self.root}")
            return
        mc.log(f"[glasses/replay] 模拟眼镜推流：{len(files)} 帧 @ {self.fps:g} fps"
               f"（循环={self.loop}）")
        interval = 1.0 / self.fps
        i, nxt = 0, time.perf_counter()
        while not self._stop.is_set():
            img = cv2.imread(str(files[i]))
            if img is not None:
                on_frame(img)
            i += 1
            if i >= len(files):
                if not self.loop:
                    break
                i = 0
            nxt += interval
            sleep = nxt - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                nxt = time.perf_counter()

    def close(self) -> None:
        self._stop.set()


class RokidGlassesAdapter(GlassesAdapter):
    """**Rokid Glasses（乐奇 AI 眼镜）**接入层 —— 真机到位时只需提供一个 `bridge`。

    已核实的官方事实（2026-10-15，见 `docs/眼镜接入_供应商确认清单.md`）：
      · **画面**：SDK 回调给的是 **NV21** 原始 YUV（官方示例 `onNv21Data()`），非 RGB/JPG
        → 本类负责转成内部统一的 BGR，像素格式的差异到此为止
      · **音频**：走**经典蓝牙**音频通道 → 采样率/声道不由我们定，必须校验契约
      · ⚠️ 手机端 App **切回后台会主动停流** → 演示期必须保持前台（或确认保活开关）
      · SDK 都在 Android 侧（CXR-L 经 Rokid AI APP；或眼镜端裸机开发）→ 拓扑为
        `眼镜 →(蓝牙/P2P) 手机 App →(WiFi/HTTP) 服务器`
    本类**不做推理、不改分辨率**（D2：眼镜端零智能）。

    `bridge` 需要实现的最小协议（真机 SDK 包到位后写一个薄壳即可）：
        set_frame_callback(cb)      # cb(nv21_bytes, width, height)
        set_audio_callback(cb)      # 可选；cb(pcm_bytes, samplerate, channels)
        start() / stop()            # 对应 SDK 的启停（如 requestVideoStream/停止）
        request_stream(fps=…)       # 可选；官方帧率范围 5~30
    没有 bridge 时**直接报错退出并给出替代方案**，不静默假装在跑。
    """

    name = "rokid"

    def __init__(self, bridge=None, max_fps: int = 5, want_audio: bool = True):
        self.bridge = bridge
        self.max_fps = int(max(1, min(30, max_fps)))    # 官方文档：帧率 5~30
        self.want_audio = want_audio
        self._on_frame = None                           # noqa: ANN001
        self._warned = False
        self.stats = {"frames": 0, "decode_err": 0, "last_wh": "",
                      "audio": "未收到", "started": False}

    def open(self, on_frame) -> None:                   # noqa: ANN001
        if self.bridge is None:
            raise SystemExit(
                "[glasses/rokid] 未提供 SDK 桥（bridge）—— 真机还没接。\n"
                "  · 想先验证 NV21 这条通道：--glasses-adapter nv21 --input <含 *.nv21 的目录>\n"
                "  · 真机接入：实现 set_frame_callback / start / stop（可选 "
                "set_audio_callback + request_stream）后传给本类，其余代码一行不改")
        self._on_frame = on_frame
        self.bridge.set_frame_callback(self._on_nv21)
        if self.want_audio and hasattr(self.bridge, "set_audio_callback"):
            self.bridge.set_audio_callback(self._on_pcm)
        req = getattr(self.bridge, "request_stream", None)
        if callable(req):
            req(fps=self.max_fps)
        self.bridge.start()
        self.stats["started"] = True
        mc.log(f"[glasses/rokid] 已启动（请求 {self.max_fps} fps）"
               f"—— 注意手机端 App 切后台会停流，演示时保持前台")

    def _on_nv21(self, data, width, height) -> None:    # SDK 回调线程，勿阻塞
        img = nv21_to_bgr(data, width, height)
        if img is None:
            self.stats["decode_err"] += 1
            if not self._warned:
                self._warned = True
                n = len(data) if data is not None else 0
                mc.log(f"[glasses/rokid] NV21 解码失败：{width}x{height} / {n} 字节"
                       f"（需要偶数宽高且长度 ≥ w*h*3/2）—— 后续同类不再刷屏")
            return
        self.stats["frames"] += 1
        self.stats["last_wh"] = f"{width}x{height}"
        self._on_frame(img)

    def _on_pcm(self, data, samplerate, channels) -> None:   # noqa: ANN001
        """音频通道：先**只校验契约**（是否 16 kHz 单声道），不接 VAD。

        为什么不直接接：真实采样率/格式要真机才知；在未知格式上接 VAD 会把
        "格式不对"误判成"门槛太严"。校验通过后再把它接到 MicListener（下一步）。
        """
        ok, msg = check_audio_spec(samplerate, channels)
        self.stats["audio"] = f"{samplerate}Hz×{channels}（{'可用' if ok else '需转换'}）"
        if ok:
            mc.log(f"[glasses/rokid] 音频 {samplerate} Hz × {channels} ch：{msg}"
                   f"（尚未接入 VAD，待真机确认后接管 MicListener 的输入）")
        else:
            mc.log(f"[glasses/rokid] ⚠️ 音频不符合 A2 契约：{msg}")

    def close(self) -> None:
        if self.bridge is not None:
            try:
                self.bridge.stop()
            except Exception as exc:  # noqa: BLE001 停流失败不该影响退出
                mc.log(f"[glasses/rokid] 停流异常：{exc}")


class Nv21DirAdapter(GlassesAdapter):
    """离线替身：把目录里的 `*.nv21` 当成"SDK 推来的 NV21 帧"按 fps 回放。

    价值：**NV21 这条路径今天就能验证**（不需要眼镜）—— 与真机共用同一个
    `nv21_to_bgr`，所以"像素格式"这个坑不会等到演示现场才暴露。
    每帧可配同名 `.json`（`{"width":1920,"height":1080}`）；缺省用 `--glasses-size`。
    """

    name = "nv21"

    def __init__(self, root, size=(640, 480), fps: float = DEFAULT_SOURCE_FPS,
                 loop: bool = True):
        self.root = Path(root)
        self.size = (int(size[0]), int(size[1]))
        self.fps = max(1.0, float(fps))
        self.loop = loop
        self._stop = threading.Event()
        self.bad = 0

    def open(self, on_frame) -> None:                   # noqa: ANN001
        threading.Thread(target=self._run, args=(on_frame,), daemon=True).start()

    def _wh(self, p: Path) -> tuple[int, int]:
        side = p.with_suffix(".json")
        if side.exists():
            try:
                d = json.loads(side.read_text(encoding="utf-8"))
                return int(d["width"]), int(d["height"])
            except Exception:  # noqa: BLE001 sidecar 坏了就用缺省，不算致命
                pass
        return self.size

    def _run(self, on_frame) -> None:                   # noqa: ANN001
        files = sorted(self.root.glob("*.nv21"), key=mc.natural_key) if self.root.is_dir() else []
        if not files:
            mc.log(f"[glasses/nv21] 目录里没有 *.nv21：{self.root}"
                   f"（可用 scripts/make_nv21_samples.py 生成）")
            return
        w, h = self._wh(files[0])
        mc.log(f"[glasses/nv21] 离线回放 NV21 推流：{len(files)} 帧 @ {self.fps:g} fps"
               f"（{w}x{h}，循环={self.loop}）")
        interval, i, nxt = 1.0 / self.fps, 0, time.perf_counter()
        while not self._stop.is_set():
            p = files[i]
            w, h = self._wh(p)
            img = nv21_to_bgr(p.read_bytes(), w, h)
            if img is None:
                self.bad += 1
                mc.log(f"[glasses/nv21] 解码失败：{p.name} @ {w}x{h}")
            else:
                on_frame(img)
            i += 1
            if i >= len(files):
                if not self.loop:
                    break
                i = 0
            nxt += interval
            sleep = nxt - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                nxt = time.perf_counter()

    def close(self) -> None:
        self._stop.set()


class GlassesSource(FrameSource):
    """眼镜源：SDK 回调 → 有界队列 → 主循环 `read()`。

    · 队列长度固定（默认 2）：**满了丢最旧** —— 与契约"最新帧优先"一致，
      也能吸收"SDK 帧率 > 服务端处理能力"时的积压（否则端到端延迟会一直涨）
    · `received` / `dropped` 会写进运行报告，便于判断是"眼镜推太快"还是"服务端慢"
    · 真机接入 = 写一个 `GlassesAdapter` 子类并在这里传进来，**主循环一行不改**
    """

    name = "glasses"

    def __init__(self, adapter: GlassesAdapter | None = None,
                 queue_size: int = GLASSES_QUEUE_SIZE):
        self.adapter = adapter or ReplayAdapter(DEFAULT_GLASSES_REPLAY)
        self.queue_size = max(1, int(queue_size))
        self.received = 0
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._fps = 30.0          # 主循环节奏用；真机可从 SDK 读实际帧率

    def open(self) -> None:
        self.adapter.open(self._on_frame)

    def _on_frame(self, frame) -> None:    # noqa: ANN001
        """SDK 回调（可能在别的线程）：入队；满则丢最旧。"""
        self.received += 1
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                self.dropped += 1

    def read(self):
        try:
            return True, self._q.get(timeout=2.0)
        except queue.Empty:
            return True, None              # 暂无新帧：不结束，主循环 continue

    # 注意：fps 是 FrameSource 上的 @property（读 self._fps），这里**不要**再定义同名方法，
    # 否则会把属性覆盖成 bound method，主循环 `min(source.fps, 60)` 直接 TypeError（已踩过一次）。

    def release(self) -> None:
        self.adapter.close()


def build_glasses_source(args) -> GlassesSource:
    """按 `--glasses-adapter` 造眼镜源。

    **真机 SDK 到位后只改这里**：加一个分支 new MyGlassesAdapter(...) 即可，
    主循环 / 发帧 / 回显 / 服务端一行都不用动（决策记录 D2/D7）。
    """
    kind = getattr(args, "glasses_adapter", "replay")

    def _dir(default):
        d = Path(args.input)
        if not d.is_absolute():
            d = PROJECT_ROOT / d
        if not d.exists():
            mc.log(f"[glasses] --input 不是有效目录，回退到 {default.relative_to(PROJECT_ROOT)}")
            return default
        return d

    def _size():
        raw = str(getattr(args, "glasses_size", "640x480")).lower().replace("*", "x")
        try:
            w, h = (int(v) for v in raw.split("x", 1))
            return w, h
        except Exception:  # noqa: BLE001 写错就退回默认，别让现场卡在参数解析
            mc.log(f"[glasses] --glasses-size 无法解析：{raw}，用 640x480")
            return 640, 480

    if kind == "replay":
        adapter = ReplayAdapter(_dir(DEFAULT_GLASSES_REPLAY),
                                fps=args.source_fps, loop=not args.no_loop)
    elif kind == "nv21":
        # 离线验证 NV21 通道（与真机共用 nv21_to_bgr）
        adapter = Nv21DirAdapter(_dir(DEFAULT_GLASSES_REPLAY), size=_size(),
                                 fps=args.source_fps, loop=not args.no_loop)
    elif kind == "rokid":
        # 真机：需要一个 SDK 桥。此处**故意不造桩**——没桥就报错并提示替代方案，
        # 避免"以为在跑真机、其实在跑模拟"这种最误导人的状态。
        adapter = RokidGlassesAdapter(bridge=None, max_fps=int(args.fps))
    else:
        raise SystemExit(f"[glasses] 未知适配器：{kind}（可选 replay / nv21 / rokid）")
    return GlassesSource(adapter, queue_size=getattr(args, "glasses_queue", GLASSES_QUEUE_SIZE))


def build_source(args) -> FrameSource:
    if args.source == "cam":
        return WebcamSource(args.index, args.width, args.height)
    if args.source == "url":
        return NetworkStreamSource(args.input, fps=args.source_fps)
    if args.source == "glasses":
        return build_glasses_source(args)
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
# 二之二、麦克风监听（N4.6：真人说话 → A2 → 切换分割目标）
# ===========================================================================
# ===========================================================================
# 三·五、回显层（N35）：窗口 / 落盘 / AR 镜片 —— **换渲染不改主循环**
# ===========================================================================
class GlassesDisplayAdapter:
    """眼镜**显示**适配器基类（与采集侧 `GlassesAdapter` 对称）。

    真机接入时只需实现 `display(img)`：把图像送到镜片 / 副屏。
    约定：
      · `img` 是 BGR 的 numpy.ndarray，**已经渲染好**（含掩膜与 HUD）
      · 不要长时间阻塞（主循环会等它返回）；确有耗时请自行投递到内部队列
    """

    name = "base"

    def open(self) -> None:
        pass

    def display(self, img) -> None:        # noqa: ANN001
        raise NotImplementedError

    def close(self) -> None:
        pass


class ReplayDisplayAdapter(GlassesDisplayAdapter):
    """无硬件时的显示替代：按间隔把画面落盘（便于读图确认"镜片里会看到什么"）。"""

    name = "replay"

    def __init__(self, out_dir, interval: float = 2.0):
        self.dir = Path(out_dir)
        self.interval = max(0.1, float(interval))
        self.n = 0
        self._last = 0.0

    def open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def display(self, img) -> None:        # noqa: ANN001
        now = time.perf_counter()
        if now - self._last < self.interval:
            return
        self._last = now
        self.n += 1
        cv2.imwrite(str(self.dir / f"display_{self.n:04d}.jpg"), img)


class MonoDisplayAdapter(GlassesDisplayAdapter):
    """**单绿色镜片**显示适配器（装饰器）：把彩色 BGR 画面压成"单色 + 亮度 + 轮廓"。

    为什么必须有这一层：Rokid Glasses 的显示是**单绿色** Micro-LED + 衍射光波导
    （单光引擎"一拖二"双目），镜片能表达的只有**亮度**（色相固定为绿）——
    我们那套"不同器官用不同颜色区分"的彩色半透明掩膜，投到镜片上会**退化成一片同色**，
    信息直接丢失。对策：
      · 颜色维度失效 → 改用 **轮廓线（edges）+ 亮度分级**：掩膜边界描边（相邻器官仍可分辨）、
        掩膜内部按亮度区分
      · 最稳的演示形态仍是"镜片只显示 HUD 文字，彩色影像放平板"（N47 半真链路）

    它可以**离线验证**：包一个 `ReplayDisplayAdapter` 就能落盘看"镜片里会是什么样"。
    """

    name = "mono"

    def __init__(self, inner: GlassesDisplayAdapter, edges: bool = True,
                 gamma: float = 1.0, gain: float = 1.0):
        self.inner = inner
        self.edges = edges
        self.gamma = float(gamma)
        self.gain = float(gain)
        self.stats = {"n": 0, "last_mean": 0.0, "last_edge": 0.0}

    def open(self) -> None:
        self.inner.open()

    def to_mono(self, img):                # noqa: ANN001
        """BGR → 单绿 BGR（R=B=0，G=亮度）。抽成公有方法便于单测 / 离线看图。"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.gamma != 1.0:
            lut = ((np.arange(256) / 255.0) ** self.gamma * 255.0).astype(np.uint8)
            gray = cv2.LUT(gray, lut)
        if self.gain != 1.0:
            gray = cv2.convertScaleAbs(gray, alpha=self.gain)
        if self.edges:
            e = cv2.Canny(gray, 40, 120)
            gray = cv2.max(gray, e)        # 描边：相邻掩膜/器官的边界仍可分辨
            self.stats["last_edge"] = float(e.mean())
        mono = np.zeros_like(img)
        mono[:, :, 1] = gray               # OpenCV 通道序为 BGR → 下标 1 = 绿
        return mono

    def display(self, img) -> None:        # noqa: ANN001
        self.stats["n"] += 1
        self.stats["last_mean"] = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
        self.inner.display(self.to_mono(img))

    def close(self) -> None:
        self.inner.close()


class Renderer:
    """回显器基类：`present(img)` 返回 True 表示"用户要求退出"。"""

    name = "base"

    def open(self) -> None:
        pass

    def present(self, img) -> bool:        # noqa: ANN001
        return False

    def close(self) -> None:
        pass


class NullRenderer(Renderer):
    """不回显（无头模式：自动化 / 后台 / 控制台自己渲染）。"""

    name = "none"


class WindowRenderer(Renderer):
    """桌面窗口回显（OpenCV）：按 `q` 或 `ESC` 退出。"""

    name = "window"

    def __init__(self, title: str = "EgoMed-Agent Live"):
        self.title = title

    def present(self, img) -> bool:        # noqa: ANN001
        cv2.imshow(self.title, img)
        return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)

    def close(self) -> None:
        cv2.destroyAllWindows()


class FileRenderer(Renderer):
    """落盘回显：按间隔把**渲染后的画面**存到 `out_dir/display/`。

    用途：无人值守跑长稳/现场时也能留下"当时看到什么"的证据；
    与 `--save-frames`（原始帧+显示帧，按已发帧数抽）互补：这里按**时间**抽。
    """

    name = "file"

    def __init__(self, out_dir, interval: float = 2.0):
        self.dir = Path(out_dir)
        self.interval = max(0.1, float(interval))
        self.n = 0
        self._last = 0.0

    def present(self, img) -> bool:        # noqa: ANN001
        now = time.perf_counter()
        if now - self._last < self.interval:
            return False
        self._last = now
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.n += 1
            cv2.imwrite(str(self.dir / f"display_{self.n:04d}.jpg"), img)
        except Exception:  # noqa: BLE001 落盘失败不影响主链路
            pass
        return False


class ArGlassesRenderer(Renderer):
    """AR 镜片回显：把画面交给**显示适配器**（真机到位只写一个 `display` 适配器）。"""

    name = "glasses"

    def __init__(self, adapter: GlassesDisplayAdapter):
        self.adapter = adapter

    def open(self) -> None:
        self.adapter.open()

    def present(self, img) -> bool:        # noqa: ANN001
        self.adapter.display(img)
        return False

    def close(self) -> None:
        self.adapter.close()


def build_renderer(args, out_dir: Path) -> Renderer:
    """按参数造回显器（N35）。

    兼容旧参数：`--show`（默认 True）→ 窗口；`--no-show` → 不回显。
    显式 `--renderer` 优先：window / file / none / glasses。
    **真机 AR 接入点**：在这里把 `ReplayDisplayAdapter` 换成厂商显示适配器即可。
    """
    kind = getattr(args, "renderer", "auto")
    if kind == "auto":
        kind = "window" if getattr(args, "show", True) else "none"
    interval = float(getattr(args, "render_interval", 2.0))
    if kind == "window":
        return WindowRenderer()
    if kind == "file":
        return FileRenderer(Path(out_dir) / "display", interval)
    if kind == "glasses":
        return ArGlassesRenderer(ReplayDisplayAdapter(Path(out_dir) / "display_ar", interval))
    if kind == "glasses-mono":
        # Rokid Glasses 实为**单绿**镜片 → 外面套一层 MonoDisplayAdapter。
        # 真机显示适配器到位后，把里面的 ReplayDisplayAdapter 换成它，这层不用改。
        return ArGlassesRenderer(MonoDisplayAdapter(
            ReplayDisplayAdapter(Path(out_dir) / "display_ar_mono", interval),
            edges=not getattr(args, "mono_no_edges", False),
            gain=float(getattr(args, "mono_gain", 1.0))))
    return NullRenderer()


class MicListener:
    """实时麦克风监听：能量 VAD 自动切句，说完一句回调一次。

    机制（与摄像头的"轮询"不同，音频是驱动回调）：
      sounddevice.InputStream(callback=...) 持续推音频块 →
      计算块 RMS → 超过阈值判"在说话"，说话中持续累积 → 静音超过 silence 秒判"说完了"
      → 拼成 wav（16 kHz 单声道 PCM16）→ 回调 on_speech(wav_bytes, 时长, RMS)

    不落盘、不常驻录音；只在判定成句后把那一句交给上层（上层落盘 + 发 A2）。
    """

    def __init__(self, on_speech, samplerate: int = 16000, threshold: float | None = None,
                 silence_sec: float = 0.8, min_speech_sec: float = 0.4,
                 preroll_sec: float = 0.3, max_speech_sec: float = 6.0,
                 device: int | None = None, min_interval_sec: float = 1.0,
                 min_active_ratio: float = 0.25, margin: float = 1.5,
                 floor_min: float = 0.0012, intra_gap_sec: float = 0.25,
                 contrast_min: float = 2.5, min_active_sec: float = 0.3):
        self.on_speech = on_speech
        self.samplerate = samplerate
        self.threshold = threshold
        self.silence_sec = silence_sec
        self.min_speech_sec = min_speech_sec
        self.pre_roll = int(preroll_sec * samplerate)
        self.max_speech_frames = int(max_speech_sec * samplerate)
        self.device = device
        self.min_interval = min_interval_sec
        # N37 远场/噪声门槛（两道，见 _emit）：
        #   min_active_ratio —— 一句话里"超过阈值"的时长占比下限（瞬时尖峰占比很低 → 拒）
        #   margin           —— 峰值必须 ≥ 判定阈值 × margin（贴边触发多是环境噪声）
        self.min_active_ratio = float(min_active_ratio)
        self.margin = float(margin)
        self.floor_min = float(floor_min)                     # 自适应阈值下限（远场人声靠它）
        self.intra_gap_frames = int(intra_gap_sec * samplerate)   # "句内停顿"上限
        # N37 第三道门槛：**峰 / 均** ≥ contrast_min —— 用来区分「人说话」与「持续环境噪声」。
        # 实测（见 scripts/vad_regression.py 的数据）：8 条指令 + 远场指令的峰/均是 4.0~7.9，
        # 而持续噪声（空调）仅 1.4、远处喧哗 2.3 → 取 2.5 能干净分开。
        self.contrast_min = float(contrast_min)
        # N37 第四道门槛：**连续**发声时长 ≥ min_active_sec。
        # 动机：麦克风块是 0.1s，一次 20ms 敲击就占满一个块 → 按"活跃占比"算每 0.4s 周期
        # 能凑到 ~29%（实测 clicks.wav 就是这样蒙混过关的）。而人说话必然**连续成串**，
        # 孤立单块不可能是说话。这条门槛直接以"最长连续活跃时长"判定。
        self.min_active_sec = float(min_active_sec)
        self._run_samples = 0         # N37: 当前连续活跃样本数
        self._run_max = 0             # N37: 本句最长连续活跃样本数
        self._stream = None
        self._pre: list = []          # 预滚缓冲（说话前的少量音频，避免吃掉首字）
        self._speech: list = []       # 当前这句话
        self._speaking = False
        self._silence_frames = 0
        self._noise = None            # 自适应噪声底（启动后前 ~0.5s 估计）
        self._noise_samples = []
        self._last_emit = 0.0
        self._seg_rms_max = 0.0       # 本句说话期间的峰值 RMS（结句时用于日志，别取静音块）
        self._seg_total = 0           # N37: 本句总样本数
        self._seg_active = 0          # N37: 本句"超阈值"样本数
        self._seg_rms_sum = 0.0       # N37: 本句块级 RMS 之和（算峰/均用）
        self._seg_n = 0               # N37: 本句块数
        self._last_thr = 0.0          # N37: 本次判定用的阈值（供 _emit 算裕度）
        self.stats = {"segments": 0, "dropped": 0, "rejected_noise": 0,
                      "last_rms": 0.0, "last_sec": 0.0,
                      "rejected_ratio": 0.0, "rejected_rms": 0.0,
                      "rejected_contrast": 0.0, "rejected_run_sec": 0.0}

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        import sounddevice as sd          # 延迟导入：未装也能跑非麦克风模式

        def _cb(indata, frames, time_info, status):   # noqa: ARG001
            self._on_block(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="float32",
            blocksize=1600, callback=_cb, device=self.device,
        )
        self._stream.start()
        dev = self.device if self.device is not None else sd.default.device[0]
        mc.log(f"[mic] 开始监听（设备 {dev}，{self.samplerate} Hz 单声道；"
               f"阈值 {'自适应' if self.threshold is None else self.threshold}）——直接说话即可")

    def stop(self) -> None:
        self.flush()                      # 收尾：把最后一句发出去，别丢
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def flush(self) -> None:
        """强制结束当前句（流停止/退出/回放结束时调用，避免最后一句丢失）"""
        if self._speaking:
            self._emit(self._seg_rms_max)

    # ------------------------------------------------------------ 块处理
    def _on_block(self, block) -> None:
        import numpy as np

        rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
        # 前 0.5 s 估计噪声底，之后固定阈值（避免把环境噪声当成说话）
        if self.threshold is None:
            if len(self._noise_samples) < 10:
                self._noise_samples.append(rms)
                # N37 修正①：取 **30 分位**而不是最大值。
                #   取 max 时，若"一开口就说话"（前 0.5s 已是人声），噪声底会被估成语音电平
                #   → 阈值暴涨（实测 cmd_07 估到 0.0833、cmd_08 到 0.2327）→ 直接听不到人声。
                floor = float(np.quantile(self._noise_samples, 0.3))
                # N37 修正②：下限从 0.003 降到 floor_min（默认 0.0012）。
                #   有了「活跃占比」这道门槛，就可以放心用更低的能量门槛去听**远场人声**
                #   （实测 -20 dB 的远场指令峰值仅 ~0.003，旧的 0.003 下限根本听不到）。
                #   上限仍保留 0.03：嘈杂环境噪声底 0.015+，×4 后会盖过人声（实测踩过）。
                self._noise = min(max(floor * 4.0, self.floor_min), 0.03)
                if len(self._noise_samples) == 10:
                    mc.log(f"[mic] 噪声底 RMS≈{floor:.4f} → 判定阈值 {self._noise:.4f}"
                           f"（夹在 {self.floor_min:g}~0.03；说话触发不了就 --mic-threshold 手调）")
                return
            thr = self._noise
        else:
            thr = self.threshold
        self._last_thr = float(thr)          # N37: 供 _emit 算能量裕度

        if not self._speaking:
            self._pre.append(block)
            if sum(b.size for b in self._pre) > self.pre_roll:
                self._pre.pop(0)
            if rms >= thr:
                self._speaking = True
                self._speech = list(self._pre)      # 带上预滚，避免首字被切
                self._pre = []
                self._silence_frames = 0
                self._seg_rms_max = rms
                self._seg_rms_sum = rms             # N37: 对比度用的均值（峰/均）
                self._seg_n = 1
                self._seg_total = block.size        # N37: 从进入说话态起统计
                self._seg_active = block.size
                self._run_samples = block.size      # N37: 连续活跃计数
                self._run_max = block.size
        else:
            self._speech.append(block)
            self._seg_rms_max = max(self._seg_rms_max, rms)
            self._seg_rms_sum += rms            # N37: 均值统计覆盖"整段"（含结句静音，
            self._seg_n += 1                    #      实测这能让峰/均落在 4~8 的稳定区间）
            if rms >= thr:
                self._seg_active += block.size      # N37: 计入"活跃"时长
                self._run_samples += block.size     # N37: 连续活跃累加
                self._run_max = max(self._run_max, self._run_samples)
                self._silence_frames = 0
            else:
                self._run_samples = 0               # N37: 连续发声被静音打断
                self._silence_frames += block.size
            # N37 修正③：分母只统计「句内」（含 ≤ intra_gap 的短暂停顿），
            # **不把结句等待的那 0.8s 长静音算进去**，否则人话占比被稀释
            # （实测 cmd_05 被误判成 23% < 25% 而拒掉）。
            if self._silence_frames <= self.intra_gap_frames:
                self._seg_total += block.size
            done = self._silence_frames >= self.silence_sec * self.samplerate
            too_long = sum(b.size for b in self._speech) >= self.max_speech_frames
            if done or too_long:
                self._emit(rms, truncated=too_long)

    def _emit(self, rms: float, truncated: bool = False) -> None:
        import numpy as np

        audio = np.concatenate(self._speech) if self._speech else np.zeros(0, dtype="float32")
        self._speaking = False
        self._speech = []
        self._silence_frames = 0
        dur = len(audio) / self.samplerate
        if dur < self.min_speech_sec:
            return                                     # 太短，当噪声丢掉
        # N37 第一/二道门槛：活跃占比 + 能量裕度
        #   为什么需要：只看"有没有块超过阈值"的话，关门声/敲键盘/远处喧哗都能触发一次 A2；
        #   真人说一句话的特点是「活跃时长占比高」且「峰值明显高于环境噪声」。
        ratio = self._seg_active / max(1, self._seg_total)
        seg_peak = max(self._seg_rms_max, rms)
        need_rms = self._last_thr * self.margin
        self._seg_total = 0
        self._seg_active = 0
        seg_mean = self._seg_rms_sum / max(1, self._seg_n)
        contrast = seg_peak / max(seg_mean, 1e-9)
        run_sec = self._run_max / self.samplerate
        if ratio < self.min_active_ratio or seg_peak < need_rms \
                or contrast < self.contrast_min or run_sec < self.min_active_sec:
            self.stats["rejected_noise"] += 1
            self.stats["rejected_ratio"] = round(ratio, 3)
            self.stats["rejected_rms"] = round(seg_peak, 4)
            self.stats["rejected_contrast"] = round(contrast, 2)
            self.stats["rejected_run_sec"] = round(run_sec, 2)
            self._seg_rms_max = 0.0
            mc.log(f"[mic] 忽略环境噪声：{dur:.2f}s ｜ 活跃占比 {ratio:.0%}"
                   f"（下限 {self.min_active_ratio:.0%}）｜ 峰值 RMS {seg_peak:.4f}"
                   f"（需 ≥ {need_rms:.4f}）｜ 峰均比 {contrast:.1f}"
                   f"（需 ≥ {self.contrast_min:g}）｜ 最长连续 {run_sec:.2f}s"
                   f"（需 ≥ {self.min_active_sec:g}s）")
            return
        now = time.monotonic()
        if now - self._last_emit < self.min_interval:
            self.stats["dropped"] += 1
            return
        self._last_emit = now
        seg_rms = max(self._seg_rms_max, rms)          # 用说话期间峰值电平，而非结句静音块
        self._seg_rms_max = 0.0
        self.stats["segments"] += 1
        self.stats["last_rms"] = seg_rms
        self.stats["last_sec"] = dur
        mc.log(f"[mic] 识别到一句话：{dur:.2f}s，说话峰值 RMS {seg_rms:.4f}"
               f"{'（超长截断）' if truncated else ''} → 发送 A2")
        try:
            self.on_speech(to_wav_bytes(audio, self.samplerate), dur)
        except Exception as exc:                       # 回调线程里不能抛
            mc.log(f"[mic] 处理失败：{exc}")


def to_wav_bytes(audio, samplerate: int) -> bytes:
    """float32 [-1,1] 单声道 → 16 kHz PCM16 wav bytes"""
    import wave

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def glasses_selftest(out_dir: Path | None = None, samples: str = "") -> int:
    """眼镜通道离线自测（Rokid Glasses）：**不接真机、不发请求**，只验"我们这侧的假设"。

    每条都对应一个真机上会踩的坑：
      ① NV21 结构约束 —— 长度必须 = w·h·3/2、宽高必须偶数（奇数必然错位 → 花屏）
      ② **红蓝不互换** —— NV21(VU) 与 NV12(UV) 写反是这类接入最经典的 bug，
         用纯色块断言通道序，比"肉眼看图"可靠得多
      ③ 往返保真 —— 报告 PSNR（色度 2×2 下采样必然有损，只判"没崩"）
      ④ **单绿镜片** —— 证明"彩色掩膜投到单色屏上会丢颜色维度"（这是设计理由，不是猜测）
      ⑤ 音频契约 —— 16 kHz 单声道才可直接进 A2；否则必须先降混 / 重采样
      ⑥ NV21 样例目录离线回放（有样例就顺带验一次 `--glasses-adapter nv21` 通道）
    """
    out = Path(out_dir) if out_dir else (PROJECT_ROOT / "runs" / "glasses_selftest")
    out.mkdir(parents=True, exist_ok=True)
    mc.log("=" * 74)
    mc.log("眼镜通道离线自测（Rokid Glasses：NV21 画面 + 单绿镜片 + 音频契约）")
    mc.log(f"输出目录：{out}")
    rows: list = []

    # ① 结构约束
    good = b"\x00" * (64 * 64 * 3 // 2)
    ok1 = (nv21_to_bgr(good, 64, 64) is not None
           and nv21_to_bgr(good, 63, 64) is None
           and nv21_to_bgr(b"\x00" * 16, 64, 64) is None)
    rows.append(("① NV21 结构约束（偶数宽高 + 长度 ≥ w·h·3/2）", ok1,
                 "64×64 通过；63×64 与短缓冲被拒"))

    # ② 红蓝不互换
    blk = np.zeros((64, 64, 3), dtype=np.uint8)
    blk[0:32, 0:32] = (0, 0, 255)          # 纯红（BGR 序）
    blk[0:32, 32:64] = (255, 0, 0)         # 纯蓝
    blk[32:64, 0:32] = (0, 255, 0)         # 纯绿
    back = nv21_to_bgr(bgr_to_nv21(blk), 64, 64)
    r = back[8:24, 8:24].reshape(-1, 3).mean(axis=0)
    b = back[8:24, 40:56].reshape(-1, 3).mean(axis=0)
    g = back[40:56, 8:24].reshape(-1, 3).mean(axis=0)
    ok2 = bool(r[2] > r[0] + 30 and b[0] > b[2] + 30 and g[1] > g[0] + 30)
    rows.append(("② 红蓝不互换（NV21=VU；写反成 NV12 会红蓝颠倒）", ok2,
                 f"红块 BGR≈{np.round(r)}｜蓝块≈{np.round(b)}｜绿块≈{np.round(g)}"))

    # ③ 往返保真（用平滑图，硬边界会把 PSNR 拉低）
    gy, gx = np.mgrid[0:120, 0:160]
    smooth = np.dstack([gx * 1.5, gy * 1.6, (gx + gy) * 0.9]).astype(np.uint8)
    sm2 = nv21_to_bgr(bgr_to_nv21(smooth), 160, 120)
    mse = float(np.mean((smooth.astype(np.float32) - sm2.astype(np.float32)) ** 2))
    psnr = 99.0 if mse <= 1e-9 else 10.0 * float(np.log10(255.0 ** 2 / mse))
    rows.append(("③ NV21 往返保真（色度必有损，只判“没崩”）", psnr >= 25.0,
                 f"PSNR {psnr:.1f} dB（阈值 ≥ 25）"))
    cv2.imwrite(str(out / "nv21_roundtrip.png"), np.hstack([smooth, sm2]))

    # ④ 单绿镜片：颜色维度确实丢失
    two = np.zeros((64, 64, 3), dtype=np.uint8)
    two[8:56, 8:32] = (0, 0, 255)          # 类别 A：红
    two[8:56, 32:56] = (255, 128, 0)       # 类别 B：蓝青
    mono = MonoDisplayAdapter(ReplayDisplayAdapter(out / "mono"), edges=True)
    mono.open()
    mono.display(two)
    mono.close()
    m = mono.to_mono(two)
    ok4 = bool(m[:, :, 0].max() == 0 and m[:, :, 2].max() == 0 and m[:, :, 1].max() > 0)
    ga = float(cv2.cvtColor(two[16:48, 12:28], cv2.COLOR_BGR2GRAY).mean())
    gb = float(cv2.cvtColor(two[16:48, 36:52], cv2.COLOR_BGR2GRAY).mean())
    rows.append(("④ 单绿镜片：颜色维度丢失（R/B 通道全 0）", ok4,
                 f"类别 A 亮度 {ga:.0f} / B {gb:.0f}（差 {abs(ga - gb):.0f}）"
                 f"→ 单色后只能靠亮度与轮廓区分"))
    cv2.imwrite(str(out / "mono_compare.png"), np.hstack([two, m]))

    # ⑤ 音频契约
    a1, m1 = check_audio_spec(16000, 1)
    a2, m2 = check_audio_spec(48000, 2)
    a3, m3 = check_audio_spec(48000, 6)
    rows.append(("⑤ 音频契约校验（16k 单声道可直接进 A2）", bool(a1 and not a2 and not a3),
                 f"16k×1→{m1}｜48k×2→{m2}｜48k×6→{m3}"))

    # ⑥ NV21 样例目录离线回放
    d = Path(samples) if samples else (PROJECT_ROOT / "data" / "samples" / "glasses_nv21")
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    if d.is_dir() and list(d.glob("*.nv21")):
        ad = Nv21DirAdapter(d, fps=20.0, loop=False)
        box: list = []
        ad.open(lambda im: box.append(im))     # noqa: E731
        time.sleep(1.0)
        ad.close()
        shape = box[0].shape if box else "-"
        rows.append((f"⑥ NV21 目录离线回放（--glasses-adapter nv21）", len(box) > 0,
                     f"{d.name}/ 取到 {len(box)} 帧，尺寸 {shape}"))
    else:
        rows.append(("⑥ NV21 目录离线回放（--glasses-adapter nv21）", True,
                     f"跳过（{d} 下无 *.nv21；可跑 scripts/make_nv21_samples.py 生成）"))

    mc.log("-" * 74)
    bad = 0
    for name, ok, note in rows:
        tag = "PASS" if ok else "FAIL"
        mc.log(f"  [{tag}] {name}")
        mc.log(f"           {note}")
        bad += 0 if ok else 1
    mc.log("-" * 74)
    mc.log(f"结果：{len(rows) - bad}/{len(rows)} PASS"
           f"（样张见 {out.relative_to(PROJECT_ROOT)}/nv21_roundtrip.png、mono_compare.png）")
    return 0 if bad == 0 else 1


def mic_selftest(seconds: float = 3.0, device: int | None = None) -> int:
    """麦克风设备自测：只报 RMS 统计，不落盘、不发请求。"""
    import numpy as np
    import sounddevice as sd

    mc.log("=" * 74)
    mc.log(f"麦克风自测：录音 {seconds:g}s，只看电平统计（不保存音频）")
    try:
        rec = sd.rec(int(seconds * 16000), samplerate=16000, channels=1,
                     dtype="float32", device=device)
        sd.wait()
    except Exception as exc:
        mc.log(f"  打不开麦克风：{exc}")
        return 1
    rms = float(np.sqrt(np.mean(rec ** 2)))
    peak = float(np.max(np.abs(rec)))
    mc.log(f"  采样 {len(rec)} 点 ｜ 平均 RMS {rms:.4f} ｜ 峰值 {peak:.4f}")
    if peak < 1e-4:
        mc.log("  电平几乎为 0：检查系统麦克风权限/静音开关/设备选择（--mic-device N）")
        return 1
    mc.log(f"  设备可用。建议阈值：{max(peak * 0.3, 0.003):.3f}"
           f"（当前安静环境峰值 {peak:.4f}；说话时若触发不了就适当调低）")
    mc.log("=" * 74)
    return 0


def vad_selftest(wav_path: str, threshold: float | None = 0.005, url: str = "",
                 send: bool = False) -> int:
    """离线 VAD 回放自测：把一段 wav 按 0.1s 分块喂给 MicListener，

    验证"切句 → 打包 wav → （可选）发 A2"整条麦克风链路，不需要真人说话。
    """
    import wave

    import numpy as np

    p = Path(wav_path)
    if not p.is_file():
        mc.log(f"wav 不存在：{p}")
        return 2
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:                                   # 线性重采样到 16 kHz
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype("float32")
    segs: list[bytes] = []
    lc = MicListener(lambda wav, dur: segs.append(wav), threshold=threshold)
    block = 1600
    mc.log("=" * 74)
    mc.log(f"离线 VAD 回放：{p.name}（{len(audio) / 16000:.2f}s，阈值 "
           f"{'自适应' if threshold is None else threshold}）")
    for i in range(0, len(audio), block):
        lc._on_block(audio[i:i + block])
    # 补 1.2s 静音让最后一句正常闭合（真实麦克风场景由 stop() 的 flush 兜底）
    silence = np.zeros(int(1.2 * 16000), dtype="float32")
    for i in range(0, len(silence), block):
        lc._on_block(silence[i:i + block])
    lc.flush()
    mc.log(f"  切出 {len(segs)} 段：{[round(len(s) / 32000, 2) for s in segs]} s")
    if not segs:
        mc.log("  未切出语音段：阈值可能偏高（试 --vad-threshold 0.002）")
        return 1
    if send and url:
        sid = mc.make_session_id("vad")
        payload, ms, err = mc.post_audio(url, sid, segs[0], "vad_seg.wav", 30.0)
        if err:
            mc.log(f"  发 A2 失败：{err}")
            return 1
        mc.log(f"  A2 结果：text={payload.get('text')!r} target={payload.get('target')!r} "
               f"need_confirm={payload.get('need_confirm')}（{ms:.0f}ms）")
    mc.log("=" * 74)
    return 0


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
        self.voice_count = 0         # 实时语音指令条数（麦克风）
        self.need_confirm_count = 0  # 需澄清次数（跨模态歧义，契约边界情况 2）
        self.voice_rejected = 0      # N37: 识别不可靠（空文本/未解析出目标）被客户端拒识的语音条数
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
        self.mic: MicListener | None = None
        self.out_dir = Path(args.out) if args.out else (
            DEFAULT_OUT / datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.samples_dir = self.out_dir / "samples"
        self.session_id = args.session or mc.make_session_id(args.doctor)
        self.pump = FramePump(args.url, self.session_id, args.timeout, self.stats,
                              self._on_result)
        # N35 回显层：窗口 / 落盘 / AR 镜片，主循环只调 present()
        self.renderer = build_renderer(args, self.out_dir)

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
    def _apply_a2_payload(self, payload, elapsed: float, label: str, err=None) -> bool:
        """统一处理 A2 响应：更新目标 / 澄清提示 / 消息，返回是否设上目标。"""
        if err is not None:
            mc.log(f"[A2] {label} 失败：{err}")
            return False
        if not isinstance(payload, dict):
            mc.log(f"[A2] {label} 返回异常")
            return False
        text = str(payload.get("text") or "")
        target = payload.get("target") or ""
        candidates = payload.get("candidates") or []
        self.stats.set_target_text = text or self.stats.set_target_text
        if payload.get("status") == "error":
            self.stats.need_confirm_count += 1
            msg = str(payload.get("message") or "")
            mc.log(f"[A2] 语音「{text}」→ 业务错误：{msg}")
            self.last_message = msg
            return False
        if payload.get("need_confirm") or not target:
            # 契约边界情况 2：跨模态歧义 → 需澄清（客户端应提示医生说得更具体）
            self.stats.need_confirm_count += 1
            msg = str(payload.get("message") or f"需要澄清：{candidates}")
            mc.log(f"[A2] 语音「{text}」→ 需澄清，候选 {candidates}（{elapsed:.0f}ms）")
            self.last_message = f"{msg}（候选：{'、'.join(map(str, candidates))}）" if candidates else msg
            return False
        self.stats.set_target_target = str(target)
        mc.log(f"[A2] 语音「{text}」→ 目标 {target}（{elapsed:.0f}ms）")
        self.last_message = str(payload.get("message") or f"已切换到 {target}")
        return True

    def _set_target_from_audio(self, wav_path: Path) -> bool:
        """启动时发一段预录音频设置目标（--audio，自动化用）"""
        if not wav_path.exists():
            mc.log(f"[A2] 音频不存在，跳过设目标：{wav_path}")
            return False
        payload, elapsed, err = mc.post_audio(
            self.args.url, self.session_id, wav_path.read_bytes(), wav_path.name,
            self.args.timeout)
        return self._apply_a2_payload(payload, elapsed, f"预录 {wav_path.name}", err)

    # ---------------------------------------------------------------- 实时语音（麦克风）
    def _on_speech(self, wav: bytes, dur: float) -> None:
        """麦克风成句回调：落盘 + 发 A2 + 更新目标（在音频回调线程里，绝不能阻塞太久）"""
        self.stats.voice_count += 1
        try:
            adir = self.out_dir / "audio"
            adir.mkdir(parents=True, exist_ok=True)
            wav_path = adir / f"voice_{self.stats.voice_count:03d}.wav"
            wav_path.write_bytes(wav)
        except Exception:
            wav_path = None
        threading.Thread(target=self._send_voice, args=(wav, dur, wav_path),
                         daemon=True).start()

    def _send_voice(self, wav: bytes, dur: float, wav_path) -> None:
        payload, elapsed, err = mc.post_audio(
            self.args.url, self.session_id, wav,
            wav_path.name if wav_path else "voice.wav", self.args.timeout)
        label = f"实时语音 {dur:.2f}s" + (f"（{wav_path.name}）" if wav_path else "")
        got_target = self._apply_a2_payload(payload, elapsed, label, err)
        # N37 第三道门槛（业务层）：识别不可靠的语音**不当成有效指令**
        #   注意边界：服务端在 A2 解析成功时**已经**改了会话 target，客户端无法回滚；
        #   所以这里能做的只有「统计 + 明确提示」，让现场知道这句不作数（要重说）。
        #   真正从根上解决要靠服务端加 confirm/dry-run 语义（走 v1.x）或唤醒词/PTT。
        if err is None and isinstance(payload, dict) and not got_target:
            text = str(payload.get("text") or "").strip()
            need_confirm = bool(payload.get("need_confirm"))
            if need_confirm:
                pass                     # 歧义澄清是正常交互，不算拒识
            elif len(text) < 2:
                self.stats.voice_rejected += 1
                mc.log(f"[mic] 拒识一条语音（识别文本过短 {text!r}）："
                       f"离麦克风近些、把话说完再试")
            elif not payload.get("target"):
                self.stats.voice_rejected += 1
                mc.log(f"[mic] 拒识一条语音（未解析出目标，text={text!r}）："
                       f"换更明确的说法，如「分割心肌」")

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
        if st.set_target_text:
            lines.append(f"语音「{st.set_target_text}」（指令 {st.voice_count} 条）")
        if st.need_confirm_count and not st.set_target_target:
            lines.append("需澄清：请说得更具体（如“左心室 LV cavity”）")
        if st.no_target and not st.set_target_target:
            lines.append("尚未设置目标：请说话（--audio 或 --mic）")
        if msg:
            lines.append(f"提示 {msg}")
        return self.hud.draw(display, lines)

    # ---------------------------------------------------------------- 主循环
    def run(self) -> int:
        args = self.args
        source = build_source(args)
        self.source = source          # 供报告展示"网络流重连次数"等源侧信息
        source.open()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_frames:
            self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.renderer.open()
        mc.log(f"[main] 会话 {self.session_id} → {args.url}"
               f"（发帧 {args.fps:g} fps，回显={self.renderer.name}）")

        # 先设目标：契约语义是"先说话（A2）→ 再发帧（A1）"，否则 A1 会提示"请先告诉我要分割哪个器官"
        if args.audio:
            self._set_target_from_audio(Path(args.audio))

        # 实时语音（N4.6）：麦克风监听，说话即切换目标
        if args.mic:
            try:
                self.mic = MicListener(
                    self._on_speech, threshold=args.mic_threshold,
                    device=args.mic_device,
                    min_active_ratio=args.mic_min_active_ratio,
                    margin=args.mic_margin,
                    floor_min=args.mic_floor,
                    contrast_min=args.mic_contrast)
                self.mic.start()
            except Exception as exc:
                mc.log(f"[mic] 启动失败（降级为无麦克风模式）：{exc}")
                mc.log("[mic] 提示：可用 --mic-selftest 先自测设备；或用 --audio 播预录音频")
                self.mic = None

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

                # 回显（N35：交给 Renderer —— 换窗口/落盘/AR 都不动主循环）
                if self.renderer.present(self._render(frame)):
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
            if getattr(self, "mic", None) is not None:
                self.mic.stop()
            self.renderer.close()
            # 等发送线程收尾：被新帧覆盖丢弃的帧**永远不会发送**，所以结束条件是
            # "所有提交都有了归宿"（已发送 + 已丢弃 == 已提交），否则会白等到
            # timeout 上限（曾导致 elapsed 虚高 ~40%、发帧率指标被低估）。
            deadline = time.perf_counter() + max(3.0, args.timeout + 1)
            while (self.stats.sent + self.stats.dropped < self.stats.submitted
                   and time.perf_counter() < deadline):
                time.sleep(0.05)

        elapsed = time.perf_counter() - t_start
        return self.report(elapsed, exit_reason)

    # ---------------------------------------------------------------- 报告
    def report(self, elapsed: float, exit_reason: str) -> int:
        st = self.stats
        submit_fps = st.submitted / elapsed if elapsed > 0 else 0.0
        send_fps = st.sent / elapsed if elapsed > 0 else 0.0
        display_fps = self.frames_read / elapsed if elapsed > 0 else 0.0
        # 服务端侧上限：并发=1 时完成率 ≈ 1/单帧耗时（P0.1 丢帧语义下的物理上限）
        server_cap = (1000.0 / st.mean) if st.mean > 0 else float("inf")
        expected_fps = min(self.args.fps, server_cap)
        lat_ok = st.pct(95) < 3000.0 or st.sent == 0
        # 口径（2026-09-28 修正）：M1 原判据"完成率 ≥ 目标发帧率"不成立——
        # 完成率上限是服务端 1/延迟（实测 X 光 ≈2.3 fps、ACDC ≈0.6 fps），
        # 与客户端设定的 3~5 fps 目标无因果关系。故拆成两条：
        #   ① 提交节奏：客户端是否按 --fps 提交（不受服务端影响）
        #   ② 结果刷新率：是否达到"服务端能力上限"的 80%（在服务端能力内不额外损失）
        submit_ok = submit_fps >= self.args.fps * 0.8
        refresh_ok = send_fps >= expected_fps * 0.8
        err_ok = (st.http_err + st.biz_err) == 0

        checks = [
            ("M1 提交节奏 ≥ 80% 目标（客户端）",
             f"{submit_fps:.2f} / 目标 {self.args.fps:g} fps", submit_ok),
            ("M2 结果刷新率 ≥ 80% 服务端能力",
             f"{send_fps:.2f} fps（上限≈{server_cap:.2f} fps，均值延迟 {st.mean:.0f}ms）",
             refresh_ok),
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
            f"| 网络流重连次数 | {getattr(getattr(self, 'source', None), 'reconnects', '-')} |",
            f"| 提交发帧 | {st.submitted} |",
            f"| 实际发送 | {st.sent}（被新帧覆盖丢弃 {st.dropped}） |",
            f"| 提交节奏 | {submit_fps:.2f} fps（目标 {self.args.fps:g} fps） |",
            f"| 结果刷新率 | {send_fps:.2f} fps（服务端上限≈{server_cap:.2f} fps） |",
            f"| 显示帧率 | {display_fps:.2f} fps |",
            f"| 延迟 均值/P50/P95 | {st.mean:.0f} / {st.pct(50):.0f} / {st.pct(95):.0f} ms |",
            f"| overlay 非空（有分割结果） | {st.overlay_nonempty} |",
            f"| 坏帧（overlay 空） | {st.bad_frame} |",
            f"| 未设目标（会话未收到语音指令） | {st.no_target} |",
            f"| A2 设目标 | 文本「{st.set_target_text or '-'}」→ `{st.set_target_target or '-'}` |",
            f"| 实时语音指令 / 需澄清 | {st.voice_count} / {st.need_confirm_count} |",
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
        mc.log(f"  采集 {self.frames_read} 帧，提交 {st.submitted}（发送 {st.sent} / 丢 {st.dropped}），"
               f"提交 {submit_fps:.2f} fps、刷新 {send_fps:.2f} fps（上限≈{server_cap:.2f}），"
               f"延迟 均值 {st.mean:.0f}ms / P95 {st.pct(95):.0f}ms")
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


def probe_stream(url: str, n: int = 5) -> int:
    """探测网络视频流：连上后读几帧，只报告分辨率/亮度/帧率（不落盘）"""
    mc.log("=" * 74)
    mc.log(f"网络流探测：{url}")
    src = NetworkStreamSource(url)
    try:
        src.open()
    except Exception as exc:
        mc.log(f"  连接失败：{exc}")
        return 1
    frames, t0, bright = 0, time.perf_counter(), []
    for _ in range(n):
        ok, frame = src.read()
        if not ok or frame is None:
            continue
        frames += 1
        bright.append(float(frame.mean()))
    dt = time.perf_counter() - t0
    src.release()
    if frames == 0:
        mc.log("  连上了但读不到帧：检查 app 是否在推流、地址是否为 /video 之类的流地址")
        return 1
    h, w = frame.shape[:2]
    fps = frames / dt if dt > 0 else 0.0
    m = sum(bright) / len(bright)
    mc.log(f"  OK  {w}x{h}  平均亮度 {m:.1f}{'（偏暗，注意补光）' if m < 30 else ''}"
           f"  读取 {frames}/{n} 帧，约 {fps:.1f} fps")
    if w < 640:
        mc.log("  提示：分辨率偏低，拍屏时细节可能不足，检测容易失败")
    mc.log("=" * 74)
    return 0


# ===========================================================================
# 七、CLI
# ===========================================================================
CONFIG_DEFAULT_REL = "configs/live.yaml"   # 不指定 --config 时自动读它（存在才读）


def load_config(path: str) -> tuple[dict, str, list[str]]:
    """读 YAML 配置（N25）。

    返回 (配置字典, 实际使用的路径, 警告列表)：
      · 未指定 path → 尝试默认路径 `configs/live.yaml`（不存在就静默跳过）
      · 指定了 path 但读不到 / 解析失败 → 返回空配置 + 警告（由调用方打印）
    """
    warns: list[str] = []
    p = Path(path) if path else (mc.PROJECT_ROOT / CONFIG_DEFAULT_REL)
    if not p.is_absolute():
        p = mc.PROJECT_ROOT / p
    if not path and not p.exists():
        return {}, "", warns
    if not p.exists():
        warns.append(f"配置文件不存在：{p}")
        return {}, str(p), warns
    try:
        import yaml
    except ImportError:
        warns.append("未安装 PyYAML，--config 被忽略（pip install pyyaml）")
        return {}, str(p), warns
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        warns.append(f"配置文件解析失败：{exc}")
        return {}, str(p), warns
    if not isinstance(data, dict):
        warns.append("配置文件顶层必须是键值对（dict）")
        return {}, str(p), warns
    return {str(k): v for k, v in data.items()}, str(p), warns


def _apply_config_defaults(p, cfg: dict) -> tuple[list[str], list[str]]:
    """把配置项写进 argparse 的**默认值**槽位。

    这样命令行显式给出的参数仍然优先（**命令行 > 配置文件 > 内置默认**）。
    返回 (已应用键, 未知或类型不符的键)。
    """
    actions = {a.dest: a for a in p._actions
               if getattr(a, "dest", None) not in (None, "help")}
    applied: list[str] = []
    problems: list[str] = []
    for key, value in cfg.items():
        if key in ("config", "help"):
            continue
        act = actions.get(key)
        if act is None:
            problems.append(key)
            continue
        val = value
        conv = getattr(act, "type", None)
        if val is not None and callable(conv) and not isinstance(val, bool):
            try:
                val = conv(val)
            except Exception:  # noqa: BLE001
                problems.append(f"{key}(类型不匹配 {value!r})")
                continue
        p.set_defaults(**{key: val})
        applied.append(key)
    return applied, problems


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # ── N25 配置文件：先看用户有没有指定 --config，再把它当"默认值"来源 ──
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default="")
    _known, _ = _pre.parse_known_args(argv)
    _cfg, _cfg_path, _cfg_warns = load_config(_known.config)

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
    p.add_argument("--source", choices=["video", "cam", "url", "glasses"], default="video",
                   help="采集源：video=视频/帧目录（自动化），cam=本机摄像头，"
                        "url=网络视频流（手机 IP Webcam 当模拟眼镜）；默认 video")
    p.add_argument("--input", default="data/samples/acdc",
                   help="video 源的输入路径（mp4 或帧目录）；glasses 源的 replay 适配器也用它")
    p.add_argument("--glasses-adapter", default="replay", choices=["replay", "nv21", "rokid"],
                   help="眼镜适配器（N34/N52）：replay=用样例帧目录模拟眼镜推流（**非** NV21）；"
                        "nv21=离线回放 NV21 样例目录（验证 Rokid 像素格式，无需真机）；"
                        "rokid=真机接入（需提供 SDK 桥，未接会明确报错）")
    p.add_argument("--glasses-size", default="640x480",
                   help="NV21 样例的缺省分辨率 WxH（每帧同名 .json 里有 width/height 时优先用它）")
    p.add_argument("--glasses-queue", type=int, default=GLASSES_QUEUE_SIZE,
                   help=f"眼镜帧队列长度（满则丢最旧，默认 {GLASSES_QUEUE_SIZE}）："
                        "吸收 SDK 帧率高于服务端处理能力时的积压")
    p.add_argument("--audio", default="", help="启动时先发这段 wav 设置分割目标（A2，契约语义：先说后看）")
    p.add_argument("--expect-overlay", action="store_true",
                   help="要求必须出现有效 overlay（否则 FAIL）——用于验证'说完话能出分割结果'")
    p.add_argument("--index", type=int, default=0, help="摄像头序号（cam 源）")
    p.add_argument("--width", type=int, default=1280, help="请求采集宽度")
    p.add_argument("--height", type=int, default=720, help="请求采集高度")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help="提交节奏 fps（默认 5）——注意这是**客户端提交**频率；"
                        "结果刷新率上限由服务端单帧耗时决定（≈1/延迟），"
                        "X 光约 2 fps、ACDC MRI 约 0.6 fps，发得比它快只会被丢帧语义覆盖")
    p.add_argument("--source-fps", type=float, default=DEFAULT_SOURCE_FPS,
                   help="视频源播放节奏（模拟摄像头帧率，默认 20）")
    p.add_argument("--no-loop", action="store_true", help="视频源播完不循环")
    p.add_argument("--display", choices=["result", "blend"], default="result",
                   help="显示方式：result=显示服务器结果图；blend=与实时画面叠加")
    p.add_argument("--no-show", dest="show", action="store_false", help="无头模式（不弹窗）")
    p.add_argument("--renderer",
                   choices=["auto", "window", "file", "none", "glasses", "glasses-mono"],
                   default="auto",
                   help="回显方式（N35）：window=桌面窗口；file=按间隔把渲染画面落盘；"
                        "none=不回显（自动化/控制台自渲染）；glasses=交给眼镜显示适配器（彩色）；"
                        "glasses-mono=**单绿镜片**（Rokid Glasses：彩色掩膜会退化成同色，"
                        "故改走轮廓+亮度）；auto=跟随 --show（默认）")
    p.add_argument("--mono-gain", type=float, default=1.0,
                   help="单绿镜片的亮度增益（glasses-mono 时生效，1.0=不调）")
    p.add_argument("--mono-no-edges", action="store_true",
                   help="单绿镜片关掉轮廓描边（默认开：相邻器官的边界靠它分辨）")
    p.add_argument("--render-interval", type=float, default=2.0,
                   help="file/glasses 回显的落盘间隔秒（默认 2.0）")
    p.add_argument("--save-frames", action="store_true", help="落盘抽样帧（每 10 个已发帧一组）")
    p.add_argument("--max-frames", type=int, default=0, help="发送多少帧后停止（0=不限）")
    p.add_argument("--duration", type=float, default=0.0, help="运行多少秒后停止（0=不限）")
    p.add_argument("--url", default=mc.DEFAULT_URL, help=f"服务端地址（默认 {mc.DEFAULT_URL}）")
    p.add_argument("--session", default="", help="指定 session_id（默认自动生成）")
    p.add_argument("--doctor", default="live", help="医生编号（用于生成 session_id）")
    p.add_argument("--timeout", type=float, default=mc.TIMEOUT, help="单请求超时（秒）")
    p.add_argument("--out", default="", help="输出目录（默认 runs/live/<时间戳>）")
    p.add_argument("--probe", action="store_true", help="只探测摄像头设备后退出")
    p.add_argument("--mic", action="store_true",
                   help="开启实时麦克风监听：说话即切换分割目标（N4.6）")
    p.add_argument("--mic-threshold", type=float, default=None,
                   help="语音检测 RMS 阈值（默认自适应 = 噪声底 ×4，夹在 0.003~0.03）；"
                        "嘈杂环境可手调 0.02~0.05")
    p.add_argument("--mic-min-active-ratio", type=float, default=0.25,
                   help="N37 门槛①：一句话里「活跃（超阈值）时长」占比下限，"
                        "低于此值判为环境噪声（默认 0.25）")
    p.add_argument("--mic-margin", type=float, default=1.5,
                   help="N37 门槛②：峰值能量需 ≥ 判定阈值 × 该倍数，"
                        "否则判为贴边噪声（默认 1.5）")
    p.add_argument("--mic-floor", type=float, default=0.0012,
                   help="N37：自适应阈值下限（默认 0.0012）。远场人声电平很低，"
                        "下限越低越能听到远处；若出现噪声误触发就调高")
    p.add_argument("--mic-contrast", type=float, default=2.5,
                   help="N37 门槛③：片段「峰值 / 均值」的比值下限（默认 2.5）。"
                        "真人说话实测 4~8，持续噪声（空调）仅 1.4 → 用它把两者分开")
    p.add_argument("--mic-min-active-sec", type=float, default=0.3,
                   help="N37 门槛④：**连续**发声时长下限（默认 0.3s）。"
                        "敲键盘/关门是孤立单块，不可能是连续说话")
    p.add_argument("--mic-device", type=int, default=None, help="麦克风设备序号（默认系统默认设备）")
    p.add_argument("--mic-selftest", type=float, default=0.0, metavar="SEC",
                   help="麦克风自测：录 SEC 秒只报电平统计（不落盘、不发请求）后退出")
    p.add_argument("--vad-selftest", default="", metavar="WAV",
                   help="离线 VAD 回放自测：把 WAV 分块喂给麦克风切句逻辑（无需真人）")
    p.add_argument("--vad-threshold", type=float, default=0.005,
                   help="VAD 回放自测阈值（默认 0.005）")
    p.add_argument("--vad-send", action="store_true",
                   help="VAD 回放自测时把切出的第一段发 A2（需要服务端在跑）")
    p.add_argument("--glasses-selftest", action="store_true",
                   help="眼镜通道离线自测（Rokid Glasses：NV21 结构 / 红蓝序 / 往返 / "
                        "单绿镜片 / 音频契约），不接真机、不发请求")
    p.add_argument("--glasses-selftest-samples", default="", metavar="DIR",
                   help="自测用的 NV21 样例目录（默认 data/samples/glasses_nv21）")
    p.add_argument("--config", default=_known.config,
                   help=f"配置文件（YAML）。不指定时自动读 {CONFIG_DEFAULT_REL}（存在才读）；"
                        "配置文件只提供**默认值**，命令行参数优先")
    p.set_defaults(show=True)

    # 配置文件 → 默认值（命令行显式给出的仍然优先）
    _applied, _unknown = _apply_config_defaults(p, _cfg)
    args = p.parse_args(argv)
    args.config_path = _cfg_path if _applied else ""
    for w in _cfg_warns:
        mc.log(f"[config] [警告] {w}")
    if _unknown:
        mc.log(f"[config] [警告] 配置里这些键不认识，已忽略：{', '.join(_unknown)}")
    if _applied:
        mc.log(f"[config] 已加载 {_cfg_path}（{len(_applied)} 项默认值）：{', '.join(_applied)}")
        mc.log("[config] 提示：命令行上加的参数会覆盖配置文件里的同名项")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.probe:
        if args.source == "url" or str(args.input).startswith(("http://", "https://", "rtsp://")):
            return probe_stream(args.input)
        return probe_cameras()
    if args.mic_selftest > 0:
        return mic_selftest(args.mic_selftest, args.mic_device)
    if args.vad_selftest:
        return vad_selftest(args.vad_selftest, args.vad_threshold,
                            url=args.url, send=args.vad_send)
    if args.glasses_selftest:
        return glasses_selftest(samples=args.glasses_selftest_samples)
    runner = LiveRunner(args)
    try:
        return runner.run()
    except KeyboardInterrupt:
        mc.log("\n[main] 手动中断，输出部分统计")
        return runner.report(0.0, "用户中断")


if __name__ == "__main__":
    sys.exit(main())
