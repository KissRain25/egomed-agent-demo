#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EgoMed-Agent 服务端接口

归属：李（服务端实现）
契约：字段定义一律以 docs/API_CONTRACT.md v0.2 为准，本文件不自行发明字段。

两种模式（用开关切换，接口签名与响应字段完全一致）：
  1) 假实现（T4，默认）：frame 把请求图原样回传；audio 固定返回文本/目标。
  2) 真引擎（T7）：frame 走「滚动窗口 → YOLO 检测 → SAM2 分割/跟踪」；
     audio 走「faster-whisper 转写 → nl_parser 解析目标」。
     切换：启动时加 `--real`，或设环境变量 EGOMED_REAL_ENGINE=1。

运行环境（本机）：conda 环境 `samannot`（含 torch + fastapi + ultralytics）。
    真引擎还需要：faster-whisper（ASR）、sam2 可从本项目解析（本文件已自动加 path）。

启动：
    python server/api.py                        # 假实现，127.0.0.1:8000
    python server/api.py --real                 # 真引擎
    python server/api.py --real --host 0.0.0.0  # 真引擎 + 局域网
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# ---------------------------------------------------------------------------
# 常量（改这里 = 改契约，必须同步 docs/API_CONTRACT.md）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
STREAM_DIR = RUNS_DIR / "stream"
LOG_FILE = RUNS_DIR / "server_log.txt"

SESSION_TTL_SECONDS = 30 * 60          # 30 分钟无请求可清理（契约 5.1）
FRAME_ENDPOINT = "/api/v1/frame"
AUDIO_ENDPOINT = "/api/v1/audio"
WINDOW_SIZE = 10                        # 滚动窗口帧数（契约附录 A.5，建议 N=10）

# 假实现固定值（契约附录 A.6）
MOCK_MODALITY = "ACDC (MRI 心脏)"
MOCK_TARGET = "LV cavity"
MOCK_FRAME_MESSAGE = "假实现：已收到画面"
MOCK_AUDIO_TEXT = "帮我分割左心室"
MOCK_AUDIO_MESSAGE = "已切换到左心室"

# P0.2 预热：--real 启动时同步加载的常用模态（至少 1 个，可按需扩展）
WARMUP_MODALITIES = ("ACDC (MRI 心脏)",)

# 错误码（契约第六节）
E_PARAM = 1001        # 参数缺失或格式错误
E_NO_SESSION = 1002   # session 不存在或已过期
E_FRAME_BAD = 2001    # 画面不可用（模糊 / 过暗 / 无有效内容）
E_MODALITY = 2002     # 未识别出模态
E_CONFIRM = 2003      # 目标无法确定（需澄清）
E_INTERNAL = 5000     # 服务器内部错误

# 真引擎开关：默认假实现；--real 或 EGOMED_REAL_ENGINE=1 启用
REAL_ENGINE = os.environ.get("EGOMED_REAL_ENGINE", "0") == "1"


# ---------------------------------------------------------------------------
# 会话状态（契约 5.1：当前模态 / 目标 / 帧计数 / 最近处理时间）
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.modality: str = ""
        self.target: str = ""
        self.frames_processed: int = 0
        self.frame_counter: int = 0       # 滚动窗口文件名自增计数
        self.last_update: float = time.time()

        # P0.1 丢帧语义：每会话串行 + 队列长度=1 + 新帧覆盖旧的未处理帧。
        # 帧处理用 condition 协调（latest-wins）；音频用普通锁串行即可。
        self._frame_cv = threading.Condition()
        self._latest_frame: Optional[bytes] = None   # 最新待处理帧（队列长度=1）
        self._latest_seq: int = 0                    # 最新帧序号
        self._done_seq: int = 0                      # 已处理完的最新序号
        self._last_result: Optional[dict] = None     # _done_seq 对应的结果
        self._working: bool = False                  # 是否正有帧在处理
        self._audio_lock = threading.Lock()          # 音频串行（per-session）

    def touch(self) -> None:
        self.last_update = time.time()

    def submit_frame(self, frame: bytes, processor) -> dict:
        """每会话「最新帧优先」丢帧语义（契约 5.4 / P0.1）：

        单会话串行处理、队列长度 = 1、新帧覆盖未处理的旧帧。被覆盖的旧帧
        请求返回最新完成帧的结果（不排队、不积压）；当前帧若已落后于最新
        完成帧，直接返回缓存结果。
        """
        cv = self._frame_cv
        with cv:
            self._latest_seq += 1
            self._latest_frame = frame
            cv.notify_all()
            while True:
                if self._working:
                    cv.wait()
                    continue
                if self._latest_seq <= self._done_seq:
                    # 已有更新帧被处理完，直接返回最新结果（本帧被覆盖丢弃）
                    return dict(self._last_result)
                # 本线程成为 worker，取当前最新帧处理（更早的旧帧随之丢弃）
                self._working = True
                seq = self._latest_seq
                data = self._latest_frame
                break
        # 锁外执行耗时处理，不阻塞本会话其它等待线程、也不阻塞其它会话
        try:
            result = processor(data)
        except BaseException:
            with cv:
                self._working = False
                cv.notify_all()
            raise
        with cv:
            self._working = False
            self._done_seq = seq
            self._last_result = result
            cv.notify_all()
        return dict(result)


_SESSIONS: dict[str, Session] = {}


def _get_session(session_id: str) -> Session:
    """按 session_id 取会话，不存在则新建（契约：客户端生成 id，服务器不管理创建）。"""
    s = _SESSIONS.get(session_id)
    if s is None:
        s = _SESSIONS[session_id] = Session(session_id)
    s.touch()
    return s


def _cleanup_expired(now: float) -> None:
    """清理超过 30 分钟无请求的会话（契约 5.1）。"""
    stale = [k for k, s in _SESSIONS.items() if now - s.last_update > SESSION_TTL_SECONDS]
    for k in stale:
        _SESSIONS.pop(k, None)


# ---------------------------------------------------------------------------
# 日志（契约 5.6：每帧一行 session / 帧号 / 模态 / 目标 / 耗时 / 状态）
# ---------------------------------------------------------------------------
def _setup_logger() -> logging.Logger:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("egomed.server")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(fh)
    return logger


_log = _setup_logger()


def _log_frame(session_id: str, frame_no: int, modality: str, target: str,
               elapsed_ms: int, status: str) -> None:
    _log.info(
        f"session={session_id} | frame={frame_no} | modality={modality or '-'} | "
        f"target={target or '-'} | elapsed={elapsed_ms}ms | status={status}"
    )


# ---------------------------------------------------------------------------
# 错误响应（契约第六节统一格式）
# ---------------------------------------------------------------------------
def _error(code: int, message: str, http_status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=http_status,
                        content={"status": "error", "code": code, "message": message})


# ---------------------------------------------------------------------------
# 真引擎惰性加载（只 import 一次，模型由 demo 内部缓存）
# ---------------------------------------------------------------------------
_ENGINE_MODULES = None     # (demo, parse_target, transcribe)
_ENGINE_LOCK = threading.Lock()   # 仅用于保护引擎懒加载（请求串行已改为 per-session）
_ASR_WARMUP = None         # asr_medical.warmup（A.11），未就绪时为 None


def _get_engine():
    """惰性加载真引擎。egomed_demo 内部会把引擎(egomed-agent-iou06.py)作为模块加载，
    并配置 CUDA_VISIBLE_DEVICES；SAM2/YOLO 模型在首次 _get_models 时才真正加载。

    A.11：ASR 优先用优化版 asr_medical（接口与 asr_engine.transcribe 完全一致）；
    缺失时回退 asr_engine，保证启动不崩。
    """
    global _ENGINE_MODULES, _ASR_WARMUP
    if _ENGINE_MODULES is not None:
        return _ENGINE_MODULES
    with _ENGINE_LOCK:
        if _ENGINE_MODULES is not None:
            return _ENGINE_MODULES
        demo_dir = str(PROJECT_ROOT / "scripts" / "demo")
        if demo_dir not in sys.path:
            sys.path.insert(0, demo_dir)
        # 让 `from sam2.build_sam import ...` 从本项目解析（本机 .pth 指向的是旧目录）
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        import egomed_demo as demo
        from nl_parser import parse_target
        # A.11：换一行 import（asr_engine -> asr_medical），签名一致；缺失则回退
        try:
            from asr_medical import transcribe, warmup as _asr_warmup
        except ImportError:
            from asr_engine import transcribe
            _asr_warmup = None
        _ASR_WARMUP = _asr_warmup
        _ENGINE_MODULES = (demo, parse_target, transcribe)
        return _ENGINE_MODULES


def _warmup_real_engine() -> None:
    """--real 启动时同步预热（P0.2 + A.11）：

    - 加载真引擎模块，并把 ≥1 个常用模态（WARMUP_MODALITIES）的 SAM2/YOLO
      提前到进程启动时加载，避免首个 frame 请求触发约 66s 的冷启动；
    - 预热 ASR（asr_medical.warmup，medium 首次加载约 30s）。

    ASR 预热失败不阻塞启动：A2 已有优雅降级，仅首个 A2 请求会变慢。
    """
    print("[server] 预热真引擎（首次较慢，请稍候）...", flush=True)
    demo, _parse_target, _transcribe = _get_engine()
    for modality in WARMUP_MODALITIES:
        print(f"[server]   预热模态模型：{modality} ...", flush=True)
        demo._get_models(modality)
    if _ASR_WARMUP is not None:
        try:
            print("[server]   预热 ASR（medium whisper，约 30s）...", flush=True)
            _ASR_WARMUP()
        except Exception as exc:  # noqa: BLE001
            print(f"[server]   ASR 预热失败（不影响启动，A2 首个请求会较慢）：{exc}",
                  flush=True)
    print("[server] 预热完成。", flush=True)


# ---------------------------------------------------------------------------
# 滚动窗口工具（契约附录 A.5）
# ---------------------------------------------------------------------------
def _frame_dir(session_id: str) -> Path:
    return STREAM_DIR / session_id / "frames"


def _prune_window(frame_dir: Path, keep: int) -> None:
    """只保留最近 keep 帧，更早的删掉（文件名 {counter:06d}.jpg，零填充保证排序正确）。"""
    files = sorted(frame_dir.glob("*.jpg"), key=lambda p: p.stem)
    for f in files[:-keep]:
        f.unlink(missing_ok=True)


def _mk_base(session_id: str) -> dict:
    """frame 响应的公共字段（status/session_id/timestamp）。"""
    return {
        "status": "ok",
        "session_id": session_id,
        "timestamp": int(time.time() * 1000),
    }


# ---------------------------------------------------------------------------
# 坏帧检测（P0.3：假实现补齐"画面不可用"分支，契约 A1 边界情况 1）
# ---------------------------------------------------------------------------
BAD_MEAN_MAX = 15.0        # 灰度均值低于此 -> 黑图/过暗
BAD_STD_MAX = 15.0         # 灰度标准差低于此 -> 低对比
BAD_BLUR_LAPVAR = 80.0     # Laplacian 方差低于此 -> 模糊
BAD_NOISE_LAPVAR = 6000.0  # Laplacian 方差高于此 -> 噪声（高频噪点）
BAD_FRAME_MESSAGE = "没看清屏幕上的影像，请对准屏幕再试一次"


def _is_bad_frame(image_bytes: bytes):
    """检测坏帧（黑图/过暗/低对比/模糊/噪声）。返回 (is_bad, reason)。

    缺 cv2/numpy 时按可用帧处理（避免因检测能力缺失误伤正常联调）；
    能解码但内容不达标时才判坏帧。
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False, ""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return True, "无法解码画面"
    mean = float(img.mean())
    std = float(img.std())
    lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
    if mean < BAD_MEAN_MAX:
        return True, "画面过暗/黑屏"
    if std < BAD_STD_MAX:
        return True, "画面低对比"
    if lap_var < BAD_BLUR_LAPVAR:
        return True, "画面模糊"
    if lap_var > BAD_NOISE_LAPVAR:
        return True, "画面噪点过多"
    return False, ""


# ---------------------------------------------------------------------------
# 假实现（T4）
# ---------------------------------------------------------------------------
def _handle_frame_mock(session_id: str, image_bytes: bytes) -> dict:
    # P0.3/P0.4：坏帧返回空 overlay + 空 modality/target（契约 A1 边界情况 1）；
    # 好帧保持原假实现（回传原图 + 固定模态/目标）。
    is_bad, _reason = _is_bad_frame(image_bytes)
    if is_bad:
        return {
            **_mk_base(session_id),
            "modality": "", "target": "",
            "overlay": "",
            "message": BAD_FRAME_MESSAGE,
            "need_confirm": False, "candidates": [],
        }
    overlay_b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        **_mk_base(session_id),
        "modality": MOCK_MODALITY,
        "target": MOCK_TARGET,
        "overlay": overlay_b64,
        "message": MOCK_FRAME_MESSAGE,
        "need_confirm": False,
        "candidates": [],
    }


def _handle_audio_mock(session_id: str) -> dict:
    return {
        "status": "ok",
        "session_id": session_id,
        "text": MOCK_AUDIO_TEXT,
        "target": MOCK_TARGET,
        "modality_hint": MOCK_MODALITY,
        "need_confirm": False,
        "candidates": [],
        "message": MOCK_AUDIO_MESSAGE,
    }


# ---------------------------------------------------------------------------
# 真引擎（T7）
# ---------------------------------------------------------------------------
def _handle_frame_real(session_id: str, image_bytes: bytes, session: Session) -> dict:
    demo, _, _ = _get_engine()
    target = session.target
    if not target:
        return {
            **_mk_base(session_id),
            "modality": "", "target": "",
            "overlay": "", "message": "请先告诉我要分割哪个器官（按住说话）",
            "need_confirm": False, "candidates": [],
        }

    # 1. 写入滚动窗口，只保留最近 N 帧
    fdir = _frame_dir(session_id)
    fdir.mkdir(parents=True, exist_ok=True)
    session.frame_counter += 1
    (fdir / f"{session.frame_counter:06d}.jpg").write_bytes(image_bytes)
    _prune_window(fdir, WINDOW_SIZE)

    # 2. 目标 -> 模态（每个目标只属于一个模态）
    modality, name_to_gray = demo.resolve_modality(target)
    if modality is None:
        return {
            **_mk_base(session_id),
            "modality": "", "target": target,
            "overlay": "", "message": f"目标「{target}」不在任何已配置模态中",
            "need_confirm": False, "candidates": [],
        }

    # 3. 模型（demo 内部按模态缓存，进程内只加载一次）
    predictor, yolo_model = demo._get_models(modality)
    id_to_name = {
        k: v for k, v in demo.eng.load_yolo_class_names_from_model(yolo_model).items()
        if v in name_to_gray
    }
    class_id_to_gray = {k: name_to_gray[v] for k, v in id_to_name.items()}
    target_id = next((k for k, v in id_to_name.items() if v == target), None)
    if target_id is None:
        return {
            **_mk_base(session_id),
            "modality": modality, "target": target,
            "overlay": "", "message": f"YOLO 权重中找不到目标「{target}」",
            "need_confirm": False, "candidates": [],
        }

    # 4. 对滚动窗口跑一次检测+分割+跟踪，取最新一帧 overlay
    out_dir = STREAM_DIR / session_id / "seg"
    image_files, _ = demo.resolve_frames(fdir, out_dir)
    result = demo._segment_target(
        predictor, yolo_model, image_files, fdir, target_id,
        id_to_name, class_id_to_gray, out_dir, target,
    )
    if result["status"].startswith("未检测") or not result["overlay_files"]:
        return {
            **_mk_base(session_id),
            "modality": modality, "target": target,
            "overlay": "", "message": "画面里没检测到目标，请对准屏幕",
            "need_confirm": False, "candidates": [],
        }

    overlay_path = Path(result["overlay_files"][-1])
    overlay_b64 = base64.b64encode(overlay_path.read_bytes()).decode("ascii")
    return {
        **_mk_base(session_id),
        "modality": modality, "target": target,
        "overlay": overlay_b64,
        "message": f"已分割 {target}",
        "need_confirm": False, "candidates": [],
    }


def _handle_audio_real(session_id: str, audio_bytes: bytes, session: Session) -> dict:
    demo, parse_target, transcribe = _get_engine()

    adir = STREAM_DIR / session_id / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    wav_path = adir / f"audio_{int(time.time() * 1000)}.wav"
    wav_path.write_bytes(audio_bytes)

    try:
        text = transcribe(str(wav_path))
    except Exception as exc:
        return {
            "status": "ok", "session_id": session_id,
            "text": "", "target": "", "modality_hint": "",
            "need_confirm": False, "candidates": [],
            "message": f"语音识别不可用（缺少 faster-whisper？）：{exc}",
        }

    if not text:
        return {
            "status": "ok", "session_id": session_id,
            "text": "", "target": "", "modality_hint": "",
            "need_confirm": False, "candidates": [],
            "message": "没听清，请再说一次",
        }

    r = parse_target(text)
    if not r.ok:
        return {
            "status": "ok", "session_id": session_id,
            "text": text, "target": "", "modality_hint": "",
            "need_confirm": False, "candidates": [],
            "message": r.message,
        }

    if len(r.targets) > 1:
        # 跨模态歧义：如"左心室"在 MRI 和超声里都存在（契约 A1 边界情况 2）
        return {
            "status": "ok", "session_id": session_id,
            "text": text, "target": "", "modality_hint": "",
            "need_confirm": True, "candidates": r.targets,
            "message": f"「{text}」在多个模态里都存在，请明确是哪一个：{'、'.join(r.targets)}",
        }

    target = r.targets[0]
    modality, _ = demo.resolve_modality(target)
    session.target = target
    session.modality = modality or ""
    return {
        "status": "ok", "session_id": session_id,
        "text": text, "target": target,
        "modality_hint": modality or "",
        "need_confirm": False, "candidates": [],
        "message": f"已切换到 {target}",
    }


# ---------------------------------------------------------------------------
# 统一入口（按开关分发；真引擎走线程池 + 锁，避免阻塞事件循环与并发冲突）
# ---------------------------------------------------------------------------
def handle_frame(session_id: str, image_bytes: bytes, session: Session) -> dict:
    if REAL_ENGINE:
        # P0.1：per-session 丢帧语义（队列长度=1，新帧覆盖旧帧），不再用全局锁
        return session.submit_frame(
            image_bytes, lambda b: _handle_frame_real(session_id, b, session))
    return _handle_frame_mock(session_id, image_bytes)


def handle_audio(session_id: str, audio_bytes: bytes, session: Session) -> dict:
    if REAL_ENGINE:
        # P0.1：音频 per-session 串行即可，不再用全局锁阻塞其它会话
        with session._audio_lock:
            return _handle_audio_real(session_id, audio_bytes, session)
    return _handle_audio_mock(session_id)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(title="EgoMed-Agent Server", version="0.2")


@app.post(FRAME_ENDPOINT)
async def post_frame(
    session_id: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),  # 契约要求，暂不使用
    image: Optional[UploadFile] = File(None),
):
    _cleanup_expired(time.time())
    if not session_id:
        return _error(E_PARAM, "缺少 session_id")
    if image is None:
        return _error(E_PARAM, "缺少 image 文件")
    try:
        image_bytes = await image.read()
    except Exception as exc:
        return _error(E_INTERNAL, f"读取图片失败：{exc}", 500)
    if not image_bytes:
        return _error(E_FRAME_BAD, "画面不可用，请对准屏幕")

    s = _get_session(session_id)
    s.frames_processed += 1
    t0 = time.perf_counter()
    try:
        payload = await run_in_threadpool(handle_frame, session_id, image_bytes, s)
    except Exception as exc:
        _log_frame(session_id, s.frames_processed, "", "", 0, f"error:{exc}")
        return _error(E_INTERNAL, f"服务器内部错误：{exc}", 500)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    payload["elapsed_ms"] = elapsed_ms
    payload["frames_processed"] = s.frames_processed
    if payload.get("modality"):
        s.modality = payload["modality"]
    if payload.get("target"):
        s.target = payload["target"]

    _log_frame(session_id, s.frames_processed, payload.get("modality", ""),
               payload.get("target", ""), elapsed_ms, "ok")
    return JSONResponse(payload)


@app.post(AUDIO_ENDPOINT)
async def post_audio(
    session_id: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),  # 契约要求，暂不使用
    audio: Optional[UploadFile] = File(None),
):
    _cleanup_expired(time.time())
    if not session_id:
        return _error(E_PARAM, "缺少 session_id")
    if audio is None:
        return _error(E_PARAM, "缺少 audio 文件")
    try:
        audio_bytes = await audio.read()
    except Exception as exc:
        return _error(E_INTERNAL, f"读取音频失败：{exc}", 500)

    s = _get_session(session_id)
    t0 = time.perf_counter()
    try:
        payload = await run_in_threadpool(handle_audio, session_id, audio_bytes, s)
    except Exception as exc:
        return _error(E_INTERNAL, f"服务器内部错误：{exc}", 500)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    _log.info(f"session={session_id} | audio | target={payload.get('target') or '-'} | "
              f"elapsed={elapsed_ms}ms | status=ok")
    return JSONResponse(payload)


@app.get("/healthz")
async def healthz():
    """健康检查（联调排障用，非契约接口）。"""
    return {"status": "ok", "sessions": len(_SESSIONS), "real_engine": REAL_ENGINE}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    global REAL_ENGINE
    import uvicorn

    parser = argparse.ArgumentParser(description="EgoMed-Agent 服务端")
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址，局域网访问用 0.0.0.0（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    parser.add_argument("--real", action="store_true",
                        help="启用真引擎（默认假实现）")
    args = parser.parse_args()

    if args.real:
        REAL_ENGINE = True
        _warmup_real_engine()   # P0.2 / A.11：同步预热引擎 + 常用模态 + ASR
    mode = "真引擎" if REAL_ENGINE else "假实现"
    print(f"[server] {mode}启动：http://{args.host}:{args.port}  （Ctrl+C 停止）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
