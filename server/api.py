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
import shutil
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
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
# 滚动窗口帧数：A.5 原建议 N=10，但每次请求都会对**整个窗口**重跑
# 「检测 + SAM2 分割 + 跟踪」，实测延迟随窗口线性增长（2 帧 1.36s → 6 帧 3.39s
# → 满窗 3.8s），突破客户端 3s 超时（见 docs/联调发现_20260920_真引擎实时链路.md）。
# N13（方案 B）改为**只对最新帧推理**：窗口=1，计算量恒定、不随会话劣化；
# 代价是失去跨帧传播（每帧独立分割，静态屏幕场景无影响；画面快速移动时掩膜可能抖动）。
WINDOW_SIZE = 1

# ---------------------------------------------------------------------------
# 请求规格上限（契约第二节：请求图 ≤300KB、请求音频 ≤200KB）
# ---------------------------------------------------------------------------
# 客户端（client/mock_client.py）已保证帧 ≤300KB、音频 ≤200KB。
# 服务端这里设**宽松的硬上限**（2MB）：既能挡住异常/恶意大包把内存打爆，
# 又不会因为客户端多压缩了 10KB 就把正常请求拒掉。超限按 1001（参数错）返回。
MAX_FRAME_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_AUDIO_UPLOAD_BYTES = 2 * 1024 * 1024

# ---------------------------------------------------------------------------
# 落盘文件保留上限（2026-11-03：修「磁盘无界增长」）
# ---------------------------------------------------------------------------
# 背景：真引擎每处理一帧就往 runs/stream/<session>/seg/{overlay,masks}/ 各写一个文件，
# 每条 A2 语音写一个 wav，**从不清理** —— 实测一次 60 分钟长稳后
# runs/stream 累积 31893 个文件 / 212.6 MB，多场长稳线性叠加。
# 现在统一按"只留最近 N 个"裁剪，单会话磁盘占用有界。
KEEP_OVERLAY_FILES = 5      # seg/overlay 保留最近 5 张
KEEP_MASK_FILES = 5         # seg/masks 保留最近 5 张
KEEP_AUDIO_FILES = 3        # audio 保留最近 3 段录音
KEEP_LOG_BYTES = 10 * 1024 * 1024    # server_log.txt 单文件上限，超过滚动（保留 3 份历史）
KEEP_LOG_BACKUPS = 3

# ---------------------------------------------------------------------------
# 帧处理等待超时（2026-11-03：cv.wait() 原来无超时）
# ---------------------------------------------------------------------------
# 原来 worker 卡死（GPU hang / 引擎异常）时，同会话的其它请求在 cv.wait() 上
# **无限期**阻塞 —— 客户端只会看到"超时/连不上"，服务端线程却永久挂住。
# 现在：单次等待有超时，累计超过上限就抛错（走 500/5000），让现场能看见。
FRAME_WAIT_POLL_S = 5.0         # cv.wait 单次轮询间隔
FRAME_WAIT_MAX_S = 120.0        # 同会话单帧累计等待上限，超过 → 500

# ---------------------------------------------------------------------------
# N18（2026-09-28）：误检面积过滤阈值按模态区分
# ---------------------------------------------------------------------------
# 引擎的面积过滤默认 min_area_ratio=0.001（0.1% 帧面积），用于滤掉零星误检。但
# **CT 腹部有大量小器官**：实测检出框占帧面积比 —— 主动脉 0.011%、胃 0.014~0.069%、
# 肾 0.034~0.048%、膀胱 0.044%、脾脏标注仅 0.028%：默认阈值会把 27/28 个框当误检
# 丢弃（Amos 脾脏检出 0/8）；关闭过滤后 8/8、Dice 0.705（见 docs/指标_分割精度_在线vs离线.md §三）。
# 其余四模态最小也有 0.17%（息肉），不受影响。
# 处置：按模态放宽 —— 这里只放宽 CT 腹部到 0.0001（≈20×20 px @4MP 的亚阈值碎点仍被过滤），
# 其余模态维持 0.001，不动已验证的配置。参数由服务端传给引擎，**引擎自身不改**（红线 4）。
DEFAULT_MIN_AREA_RATIO = 0.001
MIN_AREA_RATIO_BY_MODALITY = {
    "Amos (CT 腹部)": 0.0001,
}

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
E_FRAME_BAD = 2001    # 画面不可用（模糊 / 过暗 / 无有效内容）—— 仅作诊断位，**不进响应 JSON**
E_MODALITY = 2002     # 未识别出模态 —— 同上，仅作诊断位
E_CONFIRM = 2003      # 目标无法确定（需澄清）
E_INTERNAL = 5000     # 服务器内部错误

# 真引擎开关：默认假实现；--real 或 EGOMED_REAL_ENGINE=1 启用
REAL_ENGINE = os.environ.get("EGOMED_REAL_ENGINE", "0") == "1"

# T2：--real 预热期间 state=warming（提前监听，/healthz 返回 503 + state，客户端据此显示"预热中"）。
# 假实现无预热，直接就绪。
#
# 2026-11-03 修正：原实现在预热**失败**时也在 finally 里置 ready，导致 /healthz 报 200
# 而引擎根本没加载 —— 故障被推迟到第一个请求变成 500，现场只能看到"发帧就 500"。
# 现改为三态：warming（预热中）/ ready（就绪）/ failed（预热失败）。
# failed 时 /healthz 返 503 + state=failed，A2 直接返 500/5000，不再让请求空等。
_SERVER_STATE = "ready"
_WARMUP_ERROR = ""            # 预热失败原因（state=failed 时给客户端/日志看）

# 契约 A1 边界 1 的统一文案（决议 #15）。
# 边界 1 = "画面不可用 / 未识别出模态"：一律 status:"ok" + modality/target/overlay 全空
# + 这句话；诊断码 2001/2002 只进日志，**不出现在响应 JSON**（决议 #13）。
BOUNDARY1_MESSAGE = "画面里没检测到目标，请对准屏幕"


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
        wait_t0 = time.perf_counter()      # T9：本请求"进会话→出结果"的总计时起点
        waited_s = 0.0                     # 累计等待时长（用于超时保护）
        with cv:
            self._latest_seq += 1
            self._latest_frame = frame
            cv.notify_all()
            while True:
                if self._working:
                    # 2026-11-03：原来是无超时的 cv.wait() —— worker 卡死
                    # （GPU hang / 引擎异常）会让本会话的其它请求**永久阻塞**，
                    # 客户端只看到"超时"，服务端线程却挂死不放。
                    # 现在带超时轮询，累计超 FRAME_WAIT_MAX_S 直接抛错（走 500/5000）。
                    cv.wait(timeout=FRAME_WAIT_POLL_S)
                    waited_s += FRAME_WAIT_POLL_S
                    if waited_s >= FRAME_WAIT_MAX_S:
                        raise TimeoutError(
                            f"等待同会话上一帧完成超时（>{FRAME_WAIT_MAX_S:.0f}s，"
                            f"疑似推理线程卡死）")
                    continue
                if self._latest_seq <= self._done_seq:
                    # 已有更新帧被处理完，本帧被覆盖丢弃：返回最新结果并标记 dropped（T9）
                    out = dict(self._last_result or {})
                    out["_wait_ms"] = _ms(wait_t0)
                    out["_encode_ms"] = 0
                    out["_dropped"] = True
                    return out
                # 本线程成为 worker，取当前最新帧处理（更早的旧帧随之丢弃）
                self._working = True
                seq = self._latest_seq
                data = self._latest_frame
                break
        wait_ms = _ms(wait_t0)             # T9：排队等待耗时（被覆盖的帧这里会很小）
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
            # 缓存不含 _encode_ms（该值只对本请求有效，避免污染后续 dropped 帧的日志）
            self._last_result = {k: v for k, v in result.items() if k != "_encode_ms"}
            cv.notify_all()
        out = dict(result)
        out["_wait_ms"] = wait_ms
        out["_dropped"] = False
        return out


_SESSIONS: dict[str, Session] = {}
# 2026-11-03：会话字典加锁。
# 原来 _cleanup_expired 在遍历 _SESSIONS 的同时，另一个请求可能正在
# _get_session 里插入新键 → CPython 抛 "RuntimeError: dictionary changed size
# during iteration"，而它发生在 async handler 里 → 直接变成 500。
_SESSIONS_LOCK = threading.Lock()


def _get_session(session_id: str) -> Session:
    """按 session_id 取会话，不存在则新建（契约：客户端生成 id，服务器不管理创建）。"""
    with _SESSIONS_LOCK:
        s = _SESSIONS.get(session_id)
        if s is None:
            s = _SESSIONS[session_id] = Session(session_id)
    s.touch()
    return s


def _cleanup_expired(now: float) -> list[str]:
    """清理超过 30 分钟无请求的会话（契约 5.1），返回被清理的 session_id 列表。

    2026-11-03：原来只删内存字典、**不删磁盘目录** → runs/stream/<session>/
    永远留在盘上（实测累积 31893 文件 / 212.6 MB）。现在把磁盘清理交给调用方
    放到后台线程做（rmtree 是慢操作，不能阻塞事件循环）。
    """
    with _SESSIONS_LOCK:
        stale = [k for k, s in _SESSIONS.items()
                 if now - s.last_update > SESSION_TTL_SECONDS]
        for k in stale:
            _SESSIONS.pop(k, None)
    return stale


def _purge_session_dirs(session_ids: list[str]) -> None:
    """删除过期会话的落盘目录（runs/stream/<session_id>/）。后台线程调用。"""
    for sid in session_ids:
        d = STREAM_DIR / sid
        try:
            shutil.rmtree(d, ignore_errors=True)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 ← SystemExit 也要接住
            _log.warning("清理会话目录失败（忽略）：%s（%s）", d, exc)
            continue
        _log.info("已清理过期会话目录：%s", d)


# ---------------------------------------------------------------------------
# 日志（契约 5.6：每帧一行 session / 帧号 / 模态 / 目标 / 耗时 / 状态）
# ---------------------------------------------------------------------------
def _setup_logger() -> logging.Logger:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("egomed.server")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # 2026-11-03：改为滚动写。契约 5.6 要求"每帧一行"，长稳 60 分钟就有
        # 6600+ 行；原来是无 rotation 的 FileHandler → server_log.txt 无限增长
        # （实测已 5.8 MB，多轮长稳线性叠加）。
        fh = RotatingFileHandler(LOG_FILE, maxBytes=KEEP_LOG_BYTES,
                                 backupCount=KEEP_LOG_BACKUPS, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(fh)
    return logger


_log = _setup_logger()


def _ms(t0: float) -> int:
    """从 perf_counter 起点到现在的毫秒数（取整）—— T9 分段计时用。"""
    return round((time.perf_counter() - t0) * 1000)


def _log_frame(session_id: str, frame_no: int, modality: str, target: str,
               elapsed_ms: int, status: str,
               wait_ms: int = 0, infer_ms: int = 0, encode_ms: int = 0) -> None:
    # T9：拆出分段耗时（排队 wait / 推理 infer / 编码 encode / 合计 total），
    # 用于定位 A2/A1 并发超时卡在哪一段。
    _log.info(
        f"session={session_id} | frame={frame_no} | modality={modality or '-'} | "
        f"target={target or '-'} | wait={wait_ms}ms infer={infer_ms}ms "
        f"encode={encode_ms}ms total={elapsed_ms}ms | status={status}"
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
_ASR_MODULE = None         # asr_medical 模块本身：用来查 is_ready()（推理路径预热状态）
_ASR_FALLBACK = False      # True = 已回退到旧 asr_engine（旧引擎无"就绪"概念，保持旧行为）
_NL_ALIASES = None         # nl_parser.TARGET_ALIASES（目标→别名表），用于歧义判定与中文显示名


def _asr_ready() -> bool:
    """ASR 推理路径是否已经预热过（只有真引擎才需要关心）。

    2026-11-03 修正（真缺陷）：原实现一律 fail-open（拿不到状态就返回 True），
    结果 **A.12 的 503 门控形同虚设** —— 第一个 A2 请求在 `_ASR_MODULE` 还没赋值时
    （预热线程尚未 import 完 asr_medical）就通过了检查，随后在 run_in_threadpool 里
    阻塞等 `_ENGINE_LOCK`（预热完成才放行），实测表现为"首次语音卡 60~180 秒"，
    而不是契约 A.12 设计的"立刻 503 + Retry-After: 10，客户端 10s 节奏重试"。

    改为 **fail-closed**：查不到就绪状态就当没就绪，让 A2 走 503 门控。
    唯一保留 True 的例外是**回退到旧 asr_engine**（`_ASR_FALLBACK`）——
    旧引擎既没有 is_ready() 也没有 warmup()，谈不上"就绪"，此时保持旧行为
    （A2 可用，只是首次慢），不能把它也堵死。
    """
    if _ASR_FALLBACK:
        return True
    mod = _ASR_MODULE
    if mod is None:
        return False
    fn = getattr(mod, "is_ready", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001
        return False


def _get_engine():
    """惰性加载真引擎。egomed_demo 内部会把引擎(egomed-agent-iou06.py)作为模块加载，
    并配置 CUDA_VISIBLE_DEVICES；SAM2/YOLO 模型在首次 _get_models 时才真正加载。

    A.11：ASR 优先用优化版 asr_medical（接口与 asr_engine.transcribe 完全一致）；
    缺失时回退 asr_engine，保证启动不崩。
    """
    global _ENGINE_MODULES, _ASR_WARMUP, _ASR_MODULE, _ASR_FALLBACK, _NL_ALIASES
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
        # 别名表：歧义判定（同一概念 vs 多个器官）+ 中文显示名都要用
        try:
            from nl_parser import TARGET_ALIASES as _ALIASES
            _NL_ALIASES = _ALIASES
        except ImportError:  # pragma: no cover —— 老版本 nl_parser 没有这张表
            _NL_ALIASES = {}
        # A.11：换一行 import（asr_engine -> asr_medical），签名一致；缺失则回退
        try:
            import asr_medical
            from asr_medical import transcribe, warmup as _asr_warmup
            _ASR_MODULE = asr_medical       # 留着查 is_ready()（推理路径预热状态）
        except ImportError:
            from asr_engine import transcribe
            _asr_warmup = None
            _ASR_MODULE = None
            # 回退路径：旧 asr_engine 没有 is_ready()/warmup()，就绪门控对它无意义，
            # 标记成 fallback 以便 _asr_ready() 保持 True（不把 A2 堵死）。
            _ASR_FALLBACK = True
            print("[server]   asr_medical 不可用，回退 asr_engine（无预热门控）", flush=True)
        _ASR_WARMUP = _asr_warmup
        _ENGINE_MODULES = (demo, parse_target, transcribe)
        return _ENGINE_MODULES


def _warmup_real_engine() -> None:
    """--real 启动时后台预热（P0.2 + A.11 + T2）：

    - 分阶段打印进度与每阶段耗时（[1/3] 引擎模块 → [2/3] 常用模态 SAM2/YOLO → [3/3] ASR）；
    - 预热结束（无论成败）都会把 _SERVER_STATE 置为 ready。服务**提前监听**，
      预热期间 /healthz 返回 503 + state=warming，客户端据此显示"预热中"，
      不必再盲等 60~90 秒（T2）。

    ASR 预热失败不阻塞启动：A2 已有优雅降级，仅首个 A2 请求会变慢。
    """
    global _SERVER_STATE, _WARMUP_ERROR
    t_start = time.perf_counter()
    ok = True
    try:
        t0 = time.perf_counter()
        demo, _parse_target, _transcribe = _get_engine()
        print(f"[server]   [1/3] 引擎模块加载完成（{_ms(t0)}ms）", flush=True)
        for modality in WARMUP_MODALITIES:
            print(f"[server]   [2/3] 加载模态模型（SAM2 + YOLO）：{modality} ...", flush=True)
            t1 = time.perf_counter()
            demo._get_models(modality)
            print(f"[server]         {modality} 完成（{_ms(t1)}ms）", flush=True)
        if _ASR_WARMUP is not None:
            print("[server]   [3/3] 预热 ASR（medium whisper，约 30s）...", flush=True)
            t2 = time.perf_counter()
            try:
                _ASR_WARMUP()
                print(f"[server]         ASR 完成（{_ms(t2)}ms）", flush=True)
            except Exception as exc:  # noqa: BLE001
                # A.12：ASR 预热失败**不算整体预热失败** —— asr_medical.warmup() 内部
                # 失败也会置 _ASR_READY=True（避免 A2 永久 503），故这里只记日志。
                print(f"[server]         ASR 预热失败（不影响启动，A2 首个请求会较慢）："
                      f"{exc}（{_ms(t2)}ms）", flush=True)
    except Exception as exc:  # noqa: BLE001
        ok = False
        _WARMUP_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[server] 预热失败（引擎未就绪，/healthz 将返回 503 + state=failed）："
              f"{_WARMUP_ERROR}", flush=True)
    finally:
        # 2026-11-03 修正：原来无论成败都置 ready —— /healthz 报 200 而引擎没加载，
        # 故障被推迟到第一个请求变成 500，现场只看到"发帧就 500"。
        _SERVER_STATE = "ready" if ok else "failed"
        print(f"[server] 预热结束（state={_SERVER_STATE}），总耗时 {_ms(t_start)}ms。", flush=True)


# ---------------------------------------------------------------------------
# 滚动窗口工具（契约附录 A.5）
# ---------------------------------------------------------------------------
_PRUNE_FAIL_LOGGED = [0]     # prune 删除失败已打印次数（前 3 次打印，之后静音避免刷屏）


def _frame_dir(session_id: str) -> Path:
    return STREAM_DIR / session_id / "frames"


def _prune_window(frame_dir: Path, keep: int) -> None:
    """只保留最近 keep 帧，更早的删掉（文件名 {counter:06d}.jpg，零填充保证排序正确）。

    NOTE(2026-10-08, T11/N24): 删除**必须容错**，否则会打穿整个请求。
    本机宿主环境的「安全删除守卫」会在批量删除累计到阈值（实测 count=500）时
    **抛 SystemExit（注意：它不是 Exception 的子类，`except Exception` 抓不住）**，
    若不吞掉就会让这一帧返 HTTP 500 —— 60 分钟长稳实测 355/7200 = 4.9% 的请求因此失败。
    语义：**删不掉就留着**，临时窗口帧残留不影响本帧结果。
    """
    files = sorted(frame_dir.glob("*.jpg"), key=lambda p: p.stem)
    for f in files[:-keep]:
        try:
            f.unlink(missing_ok=True)
        except (Exception, SystemExit) as exc:  # noqa: BLE001  ← SystemExit 也要接住
            if _PRUNE_FAIL_LOGGED[0] < 3:        # 前 3 次打印，之后静音（避免刷屏）
                _PRUNE_FAIL_LOGGED[0] += 1
                _log.warning("prune 删除失败（忽略，第 %d 次）：%s（%s）",
                             _PRUNE_FAIL_LOGGED[0], f.name, exc)
            continue


def _prune_files(directory: Path, keep: int,
                 suffixes: tuple[str, ...] = (".jpg", ".png", ".jpeg", ".wav")) -> None:
    """只保留目录下**最近 keep 个**文件（按修改时间），其余删掉。

    2026-11-03（修「磁盘无界增长」）：真引擎每处理一帧就往
    runs/stream/<session>/seg/{overlay,masks}/ 各写一个文件、每条语音写一个 wav，
    **从不清理** —— 实测 60 分钟长稳后 runs/stream 累积 31893 个文件 / 212.6 MB，
    多场长稳线性叠加，交付机磁盘被吃满。

    删除同样必须容错（宿主「安全删除守卫」会抛 SystemExit，同 _prune_window）：
    **删不掉就留着**，残留文件只是占点磁盘，不影响功能。
    """
    try:
        files = [p for p in directory.iterdir()
                 if p.is_file() and p.suffix.lower() in suffixes]
    except (Exception, SystemExit) as exc:  # noqa: BLE001  目录不存在 / 守卫拦截
        _log.debug("prune_files 跳过（%s）：%s", directory, exc)
        return
    if len(files) <= keep:
        return
    try:
        files.sort(key=lambda p: p.stat().st_mtime)
    except (Exception, SystemExit):  # noqa: BLE001  并发删除导致 stat 失败
        return
    for f in files[:-keep]:
        try:
            f.unlink(missing_ok=True)
        except (Exception, SystemExit) as exc:  # noqa: BLE001  ← SystemExit 也要接住
            if _PRUNE_FAIL_LOGGED[0] < 3:
                _PRUNE_FAIL_LOGGED[0] += 1
                _log.warning("prune 删除失败（忽略，第 %d 次）：%s（%s）",
                             _PRUNE_FAIL_LOGGED[0], f.name, exc)
            continue


def _mk_base(session_id: str) -> dict:
    """frame 响应的公共字段（status/session_id/timestamp）。"""
    return {
        "status": "ok",
        "session_id": session_id,
        "timestamp": int(time.time() * 1000),
    }


def _boundary1_payload(session_id: str, frames_processed: int = 0) -> dict:
    """契约 A1 边界 1 的标准响应（画面不可用 / 未识别出模态）。

    规则（契约 §4 A1 边界 1 + 决议 #13/#15）：
    一律 `status:"ok"` + modality/target/overlay 全空 + 统一 message，
    **诊断码 2001 / 2002 只进日志，不出现在响应 JSON**。
    2026-11-03 修正：空文件上传原先返回 HTTP 400 + code 2001（违约，且客户端会
    当成业务错误丢掉上一帧），现统一走边界 1。

    2026-11-03 再修正（字段对齐）：本函数被 post_frame **提前 return** 调用，
    绕过了统一补 `elapsed_ms` / `frames_processed` 的那段 → 响应只有 9 个字段，
    比契约 A1 的 11 个少 2 个（客户端虽不读，但契约示例与诊断工具都按 11 个字段）。
    现补齐：`elapsed_ms=0`（本帧未跑推理）、`frames_processed` 取会话当前计数。
    """
    return {
        **_mk_base(session_id),
        "modality": "", "target": "",
        "overlay": "", "message": BOUNDARY1_MESSAGE,
        "need_confirm": False, "candidates": [],
        "elapsed_ms": 0, "frames_processed": frames_processed,
    }


# ---------------------------------------------------------------------------
# 响应图片规格（P2.10：契约第二节 —— 长边 ≤1280、jpeg 质量 80）
# ---------------------------------------------------------------------------
OVERLAY_MAX_DIM = 1280               # 长边上限（px）
OVERLAY_JPEG_QUALITY = 80            # jpeg 质量
OVERLAY_MAX_BYTES = 300 * 1024       # 体积上限（超过才重新编码，已合规输入不重复编码）


def _encode_overlay(raw: bytes) -> bytes:
    """把 overlay 压到契约规格（长边 ≤1280、质量 80）。

    真引擎此前直接回传原尺寸结果图（实测 723 KB~1.28 MB/帧 base64），
    远超契约第二节规格；这里统一转 jpeg 并限制长边。解码失败则原样返回
    —— 宁可体积大一点，也不能因压缩失败丢结果。
    """
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return raw
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge > OVERLAY_MAX_DIM:
            scale = OVERLAY_MAX_DIM / float(long_edge)
            img = cv2.resize(
                img, (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), OVERLAY_JPEG_QUALITY])
        return buf.tobytes() if ok else raw
    except Exception as exc:
        # 2026-11-03：原来这里静默回传原图（可能 >300KB / 长边 >1280，超契约规格），
        # 且不留任何痕迹，事后无法判断"到底是哪一帧的图超了规格"。补一条日志。
        _log.warning("overlay 压缩失败，回传原图（%d 字节，可能超契约规格）：%s",
                     len(raw), exc)
        return raw


def _overlay_b64(raw: bytes) -> str:
    """overlay → base64；体积超契约上限时先按规格压缩。

    输入已 ≤300 KB（如假实现回传的客户端压缩帧）时直接编码，
    省掉一次无意义的解码+重编码（约 15 ms/帧）。
    """
    if len(raw) > OVERLAY_MAX_BYTES:
        raw = _encode_overlay(raw)
    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# 请求参数校验（P2.9：契约 §4 —— timestamp 为 Unix 毫秒整数且必填）
# ---------------------------------------------------------------------------
def _parse_timestamp(value: Optional[str]) -> tuple[Optional[int], str]:
    """返回 (timestamp, err)；err 非空表示参数不合法（应返 1001）。"""
    if value is None or str(value).strip() == "":
        return None, "缺少 timestamp（Unix 毫秒整数）"
    try:
        ts = int(str(value).strip())
    except (TypeError, ValueError):
        return None, f"timestamp 必须是 Unix 毫秒整数，收到 {value!r}"
    if ts <= 0:
        return None, "timestamp 必须是正的 Unix 毫秒整数"
    return ts, ""


# ---------------------------------------------------------------------------
# 坏帧检测（P0.3：假实现补齐"画面不可用"分支，契约 A1 边界情况 1）
# ---------------------------------------------------------------------------
# N16（2026-09-28）归一化：Laplacian 方差**对分辨率极敏感** —— 同一张正常 MRI
# 原图 lapvar≈28（旧阈值 80 → 误判"画面模糊"），客户端压到长边 1280 后 lapvar≈109
#（判正常）：判定结果取决于"图有多大"，而不是"画面清不清楚"；mock 套件恰好全走
# 压缩图，于是掩盖了这条局限。修法：判定前先把长边统一缩放到 BAD_NORM_DIM 再算
# 统计量 —— 归一化后同一画面的原图/压缩版指标几乎一致（实测 ACDC 463.1 vs 464.2），
# 阈值也才有统一的物理含义。
BAD_NORM_DIM = 640         # 判定口径：长边统一到该尺寸后再算统计量
BAD_MEAN_MAX = 15.0        # 灰度均值低于此 → 黑图/过暗（归一化后：坏帧 9.4 / 正常 ≥93.7）
BAD_STD_MAX = 20.0         # 灰度标准差低于此 → 低对比（归一化后：坏帧 11.1 / 正常 ≥57.1）
BAD_BLUR_LAPVAR = 120.0    # Laplacian 方差低于此 → 模糊（归一化后：坏帧 ≤18.8 / 正常 ≥463）
BAD_NOISE_LAPVAR = 12000.0 # Laplacian 方差高于此 → 噪声（归一化后：正常 ≤1538 / 噪点样例 110k）
# 坏帧文案统一用契约 A1 边界 1 的措辞（决议 #15：统一为"画面里没检测到目标，请对准屏幕"）。
# 2026-11-03 修正：原为旧文案「没看清屏幕上的影像，请对准屏幕再试一次」，
# 与真引擎分支文案不一致，且不符契约决议 #15 的统一要求。
BAD_FRAME_MESSAGE = BOUNDARY1_MESSAGE


def _is_bad_frame(image_bytes: bytes):
    """检测坏帧（黑图/过暗/低对比/模糊/噪声）。返回 (is_bad, reason)。

    缺 cv2/numpy 时按可用帧处理（避免因检测能力缺失误伤正常联调）；
    能解码但内容不达标时才判坏帧。判定前统一归一化长边（N16），
    使同一画面的判定不随分辨率/压缩变化。
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
    # N16：长边归一化（缩小用 INTER_AREA 保真，放大用 INTER_LINEAR 平滑）
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge and long_edge != BAD_NORM_DIM:
        scale = BAD_NORM_DIM / float(long_edge)
        img = cv2.resize(
            img, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
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
    overlay_b64 = _overlay_b64(image_bytes)
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

    # 4. 对（窗口内的）最新帧跑一次检测+分割，取结果 overlay
    #    N13：min_occurrence=1 —— 该滤波参数本意是"同一目标连续出现≥2帧才保留"
    #    以抑制误检，但它假设后面还有帧；首次交互只有 1 帧时会把结果全滤掉，
    #    表现为"每个会话第一帧必然返回'没检测到目标'"。这里显式传 1，让首帧即有结果。
    out_dir = STREAM_DIR / session_id / "seg"
    image_files, _ = demo.resolve_frames(fdir, out_dir)
    # N18：面积过滤阈值按模态取（默认 0.001；CT 腹部 0.0001，否则小器官会被当误检丢弃）
    min_area_ratio = MIN_AREA_RATIO_BY_MODALITY.get(modality, DEFAULT_MIN_AREA_RATIO)
    result = demo._segment_target(
        predictor, yolo_model, image_files, fdir, target_id,
        id_to_name, class_id_to_gray, out_dir, target,
        min_occurrence=1,
        min_area_ratio=min_area_ratio,
    )
    # P2.12：frames_processed 语义 = "该 session 累计处理帧数"。此处（真正跑过推理
    # 的路径、且已在 per-session 串行区内）自增，被 latest-wins 丢弃的帧不计数，
    # 避免并发下多个响应返回同一个"请求时刻"计数。
    session.frames_processed += 1
    if result["status"].startswith("未检测") or not result["overlay_files"]:
        return {
            **_mk_base(session_id),
            "modality": modality, "target": target,
            "overlay": "", "message": "画面里没检测到目标，请对准屏幕",
            "need_confirm": False, "candidates": [],
        }

    t_enc = time.perf_counter()                       # T9：编码分段计时
    overlay_path = Path(result["overlay_files"][-1])
    overlay_b64 = _overlay_b64(overlay_path.read_bytes())
    encode_ms = _ms(t_enc)
    # 2026-11-03（修「磁盘无界增长」）：读完之后立刻裁剪 seg/ 下的落盘产物。
    # 引擎每处理一帧就写 1 张 overlay + 1 张 mask，从不清理 —— 实测一次 60 分钟
    # 长稳后 runs/stream 累积 31893 个文件 / 212.6 MB。现在只留最近几个。
    _prune_files(out_dir / "overlay", KEEP_OVERLAY_FILES)
    _prune_files(out_dir / "masks", KEEP_MASK_FILES)
    return {
        **_mk_base(session_id),
        "modality": modality, "target": target,
        "overlay": overlay_b64,
        "message": f"已分割 {target}",
        "need_confirm": False, "candidates": [],
        "_encode_ms": encode_ms,   # T9：仅供 post_frame 记分段日志，响应前会被弹出
    }


def _audio_payload(session_id: str, text: str, target: str, modality_hint: str,
                   need_confirm: bool, candidates: list, message: str,
                   timing: dict) -> dict:
    """A2 响应体（字段与契约 A2 完全一致），另附 _timing 供 post_audio 记日志（会被弹出）。"""
    return {
        "status": "ok", "session_id": session_id,
        "text": text, "target": target, "modality_hint": modality_hint,
        "need_confirm": need_confirm, "candidates": candidates,
        "message": message,
        "_timing": timing,
    }


def _zh_target_name(target: str, aliases: dict) -> str:
    """目标的中文显示名（取第一个中文别名；没有中文别名就用标准名）。

    医生看的是中文（"左心室"），不是 YOLO 权重里的类名（"LV cavity"）。
    """
    for a in aliases.get(target, []):
        # 用 ord + 十六进制判 CJK（U+4E00 起），避免源码里写死字面字符
        if a and any(ord(ch) >= 0x4E00 for ch in a):
            return a
    return target


def _same_concept(targets: list[str], aliases: dict) -> tuple[bool, str]:
    """判定多个命中目标是不是「同一个概念」的不同写法。

    例："左心室" 在 MRI 里是 `LV cavity`、在超声里是 `Left Ventricle` ——
    医生说的是一个东西，只是落在两个模态 → **跨模态歧义**，
    要问他"是哪一个模态"（契约 A1 边界情况 2）。
    而"肝脏和左肾"是两个不同器官 → **多目标**，不是歧义。

    判据：别名表有交集，且交集里至少有一个长度 ≥2 的中文别名
    （排除 "lv"/"rv" 这类过短别名造成的误判）。
    返回 (是否同一概念, 共同中文别名)。
    """
    if len(targets) < 2:
        return False, ""
    sets = [set(aliases.get(t, [])) for t in targets]
    if not sets or any(not s for s in sets):
        return False, ""
    common = set.intersection(*sets)
    zh = sorted((a for a in common
                 if len(a) >= 2 and any(ord(c) >= 0x4E00 for c in a)),
                key=len)
    return (bool(zh), zh[0] if zh else "")


def _candidate_modalities(demo, targets: list[str]) -> list[str]:
    """candidates = 候选**模态名**（按目标顺序去重，跳过解析不出模态的目标）。

    2026-11-03 闫裁决：`candidates` 语义统一为「模态名」。
    背景：契约 §4 字段说明写"候选目标"（如 `["左心室","右心室"]`），
    而 §4 边界 2 的示例写的是模态名（`["ACDC (MRI 心脏)","CAMUS (超声)"]`）——
    **契约自身矛盾**。现按边界 2 示例统一为模态名（也更符合"请医生选模态"的交互语义）。
    """
    out: list[str] = []
    for t in targets:
        try:
            modality, _ = demo.resolve_modality(t)
        except Exception:  # noqa: BLE001
            modality = None
        if modality and modality not in out:
            out.append(modality)
    return out


def _handle_audio_real(session_id: str, audio_bytes: bytes, session: Session) -> dict:
    demo, parse_target, transcribe = _get_engine()

    # T9：分段计时（读入/落盘 read、转写 asr、解析 nlu），供 post_audio 定位慢在哪一段
    t_read = time.perf_counter()
    adir = STREAM_DIR / session_id / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    wav_path = adir / f"audio_{int(time.time() * 1000)}.wav"
    wav_path.write_bytes(audio_bytes)
    # 2026-11-03（修「磁盘无界增长」）：每条语音都落一个 wav，从不清理 →
    # 只留最近 KEEP_AUDIO_FILES 段（ASR 只需要当前这一段）。
    _prune_files(adir, KEEP_AUDIO_FILES, suffixes=(".wav",))
    timing = {"audio": wav_path.name, "read_ms": _ms(t_read), "asr_ms": 0, "nlu_ms": 0}

    t_asr = time.perf_counter()
    try:
        text = transcribe(str(wav_path))
    except Exception as exc:
        # 2026-11-03 修正（真缺陷）：原先把 ASR 异常吞成 HTTP 200 + status:"ok"
        # （客户端会当成"识别成功但没听清"），还把异常原文回给客户端。
        # 契约 §6：服务器内部错误 → status:"error" + code 5000 + HTTP 500。
        # 这里改为向上抛，由 post_audio 统一转成 500/5000。
        timing["asr_ms"] = _ms(t_asr)
        _log.exception("session=%s | ASR 转写失败：%s", session_id, exc)
        raise
    timing["asr_ms"] = _ms(t_asr)

    if not text:
        return _audio_payload(session_id, "", "", "", False, [],
                              "没听清，请再说一次", timing)

    t_nlu = time.perf_counter()
    r = parse_target(text)
    timing["nlu_ms"] = _ms(t_nlu)
    if not r.ok:
        return _audio_payload(session_id, text, "", "", False, [], r.message, timing)

    if len(r.targets) > 1:
        # 澄清分支（契约 A1 边界情况 2）。成因有两种，candidates 统一给**模态名**：
        #   ① 跨模态歧义：同一概念在多个模态都有（"左心室" ∈ MRI 的 LV cavity / 超声的 Left Ventricle）
        #      → 医生要回答"是哪个模态"，候选就是模态名
        #   ② 多目标：医生一次说了多个器官（"肝脏和左肾"），引擎一次只分割一个
        #      → 医生要回答"是哪个器官"，待选目标名放在 message 里
        # ② 原来也写"在多个模态里都存在"，是**错的**（肝肾都在 CT 腹部，不是模态歧义），
        #    现按成因分别措辞；无论哪种，candidates 都是模态名（闫 2026-11-03 裁决）。
        modalities = _candidate_modalities(demo, r.targets)
        same, zh = _same_concept(r.targets, _NL_ALIASES or {})
        if same and len(modalities) >= 2:
            msg = (f"「{zh}」在多个模态里都存在（{'、'.join(modalities)}），"
                   f"请明确是哪一个")
        else:
            names = "、".join(_zh_target_name(t, _NL_ALIASES or {}) for t in r.targets)
            msg = f"识别到多个目标：{names}（当前一次分割一个，请明确指定一个）"
            if len(modalities) == 1:
                msg += f"（都在 {modalities[0]} 里）"
        return _audio_payload(session_id, text, "", "", True, modalities, msg, timing)

    target = r.targets[0]
    modality, _ = demo.resolve_modality(target)
    session.target = target
    session.modality = modality or ""
    return _audio_payload(session_id, text, target, modality or "", False, [],
                          f"已切换到 {target}", timing)


# ---------------------------------------------------------------------------
# 统一入口（按开关分发；真引擎走线程池 + 锁，避免阻塞事件循环与并发冲突）
# ---------------------------------------------------------------------------
def handle_frame(session_id: str, image_bytes: bytes, session: Session) -> dict:
    if REAL_ENGINE:
        # P0.1：per-session 丢帧语义（队列长度=1，新帧覆盖旧帧），不再用全局锁。
        # P2.12：帧计数在 _handle_frame_real 内自增（被覆盖丢弃的帧不计数）。
        return session.submit_frame(
            image_bytes, lambda b: _handle_frame_real(session_id, b, session))
    # 假实现没有丢帧语义：每个请求都真的处理了，处理完即计数（P2.12）
    payload = _handle_frame_mock(session_id, image_bytes)
    session.frames_processed += 1
    return payload


def handle_audio(session_id: str, audio_bytes: bytes, session: Session) -> dict:
    if REAL_ENGINE:
        # P0.1：音频 per-session 串行即可，不再用全局锁阻塞其它会话
        with session._audio_lock:
            return _handle_audio_real(session_id, audio_bytes, session)
    return _handle_audio_mock(session_id)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(title="EgoMed-Agent Server", version="1.0.4")


@app.post(FRAME_ENDPOINT)
async def post_frame(
    session_id: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),  # 契约 §4：Unix 毫秒整数，必填（P2.9）
    image: Optional[UploadFile] = File(None),
):
    stale = _cleanup_expired(time.time())
    if stale:
        # 2026-11-03：过期会话的落盘目录一并清掉（原来只删内存字典）。
        # rmtree 是慢操作 → 放后台线程，不阻塞事件循环。
        threading.Thread(target=_purge_session_dirs, args=(stale,), daemon=True).start()
    if not session_id:
        return _error(E_PARAM, "缺少 session_id")
    _ts, ts_err = _parse_timestamp(timestamp)   # P2.9：缺失/非整数 → 1001
    if ts_err:
        return _error(E_PARAM, ts_err)
    if image is None:
        return _error(E_PARAM, "缺少 image 文件")
    # 2026-11-03：上传体积上限。契约 §2 规定请求图 ≤300KB（客户端已保证），
    # 这里设宽松硬上限 2MB —— 挡住异常大包把内存打爆，又不会误伤正常帧。
    if getattr(image, "size", None) and image.size > MAX_FRAME_UPLOAD_BYTES:
        _log.warning("session=%s | 上传画面超限：%d 字节 > %d",
                     session_id, image.size, MAX_FRAME_UPLOAD_BYTES)
        return _error(E_PARAM, f"画面过大（{image.size // 1024} KB），"
                               f"上限 {MAX_FRAME_UPLOAD_BYTES // 1024} KB")
    # 取会话放在读文件之前：空文件分支（边界 1）也要用 frames_processed 填字段，
    # 且空帧说明会话已经开始，建会话记录是合理的（30 分钟 TTL 后自动回收）。
    s = _get_session(session_id)
    try:
        image_bytes = await image.read()
    except Exception as exc:
        _log.exception("session=%s | 读取图片失败：%s", session_id, exc)
        return _error(E_INTERNAL, "读取图片失败，请查看服务端日志", 500)
    if len(image_bytes) > MAX_FRAME_UPLOAD_BYTES:
        return _error(E_PARAM, f"画面过大（{len(image_bytes) // 1024} KB），"
                               f"上限 {MAX_FRAME_UPLOAD_BYTES // 1024} KB")
    if not image_bytes:
        # 空文件 = 画面不可用 → 契约 A1 边界 1（status:"ok" + 空字段 + 统一 message）。
        # 2026-11-03 修正：原先返回 HTTP 400 + code 2001，违反决议 #13
        # （"2001 仅作诊断位，响应 JSON 不出现"），且客户端会当业务错误丢掉上一帧回显。
        _log.info("session=%s | frame=empty | 边界1（空文件，2001 仅作诊断位）", session_id)
        return JSONResponse(_boundary1_payload(session_id, s.frames_processed))

    # P2.12：计数改到"真正处理完该帧"的位置（真引擎在 _handle_frame_real 内、
    # 假实现无丢帧语义在 handle_frame 内自增），此处不再按"请求到达"累加。
    t0 = time.perf_counter()
    try:
        payload = await run_in_threadpool(handle_frame, session_id, image_bytes, s)
    except Exception as exc:
        # 500 的 message 不再把异常原文回给客户端（内部路径/栈信息不该外泄），
        # 完整 traceback 进服务端日志，客户端只拿到可读的一句话（契约 §6 5000）。
        _log.exception("session=%s | 帧处理异常：%s", session_id, exc)
        _log_frame(session_id, s.frames_processed, "", "", 0, f"error:{type(exc).__name__}")
        return _error(E_INTERNAL, f"服务器内部错误（{type(exc).__name__}），详见服务端日志", 500)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    # T9：拆出分段耗时（wait/infer/encode），私有字段弹掉、不进响应
    wait_ms = payload.pop("_wait_ms", 0)
    encode_ms = payload.pop("_encode_ms", 0)
    dropped = payload.pop("_dropped", False)
    infer_ms = 0 if dropped else max(0, elapsed_ms - wait_ms - encode_ms)
    payload["elapsed_ms"] = elapsed_ms
    payload["frames_processed"] = s.frames_processed
    if payload.get("modality"):
        s.modality = payload["modality"]
    if payload.get("target"):
        s.target = payload["target"]

    _log_frame(session_id, s.frames_processed, payload.get("modality", ""),
               payload.get("target", ""), elapsed_ms,
               "dropped" if dropped else "ok",
               wait_ms=wait_ms, infer_ms=infer_ms, encode_ms=encode_ms)
    return JSONResponse(payload)


@app.post(AUDIO_ENDPOINT)
async def post_audio(
    session_id: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),  # 契约 §4：Unix 毫秒整数，必填（P2.9）
    audio: Optional[UploadFile] = File(None),
):
    stale = _cleanup_expired(time.time())
    if stale:
        threading.Thread(target=_purge_session_dirs, args=(stale,), daemon=True).start()
    if not session_id:
        return _error(E_PARAM, "缺少 session_id")
    _ts, ts_err = _parse_timestamp(timestamp)   # P2.9：缺失/非整数 → 1001
    if ts_err:
        return _error(E_PARAM, ts_err)
    if audio is None:
        return _error(E_PARAM, "缺少 audio 文件")
    # 2026-11-03：上传体积上限（契约 §2：音频 ≤200KB、≤5 秒；这里硬上限 2MB）
    if getattr(audio, "size", None) and audio.size > MAX_AUDIO_UPLOAD_BYTES:
        _log.warning("session=%s | 上传音频超限：%d 字节 > %d",
                     session_id, audio.size, MAX_AUDIO_UPLOAD_BYTES)
        return _error(E_PARAM, f"音频过大（{audio.size // 1024} KB），"
                               f"上限 {MAX_AUDIO_UPLOAD_BYTES // 1024} KB")
    try:
        audio_bytes = await audio.read()
    except Exception as exc:
        _log.exception("session=%s | 读取音频失败：%s", session_id, exc)
        return _error(E_INTERNAL, "读取音频失败，请查看服务端日志", 500)
    if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        return _error(E_PARAM, f"音频过大（{len(audio_bytes) // 1024} KB），"
                               f"上限 {MAX_AUDIO_UPLOAD_BYTES // 1024} KB")

    s = _get_session(session_id)

    # ASR 推理路径还没热好 → 立刻 503，别让请求钩住 3 分钟。
    # 背景：真引擎的"预热"只加载权重，首次 forward 才做 CUDA kernel 加载 / cuDNN
    # autotune，实测 >180s。让请求干等，客户端只会报"超时/连不上"，更误导人。
    # 这里回 503 + Retry-After：语义是"服务在，但这一路暂时没准备好"，可重试。
    #
    # 2026-11-03 修正：原来只有 `not _asr_ready()` 一个条件，而 `_asr_ready()` 是
    # fail-open 的 → 门控被架空，首次 A2 反而阻塞 60~180s。现改为按服务端状态三态判断：
    #   failed  → 预热就失败了，不是"还没准备好" → 500（别让客户端无限重试）
    #   warming → 还在预热 → 503 asr_warming（契约 A.12，可重试）
    #   ready 但 ASR 未就绪 → 503 asr_warming（ASR prime 线程还在跑）
    if REAL_ENGINE:
        if _SERVER_STATE == "failed":
            _log.error("session=%s | A2 拒绝：服务端预热失败（%s）", session_id, _WARMUP_ERROR)
            return _error(E_INTERNAL,
                          f"服务端预热失败（{_WARMUP_ERROR}），请查看服务端日志", 500)
        if _SERVER_STATE == "warming" or not _asr_ready():
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "10"},
                content={
                    "status": "error",
                    "code": E_INTERNAL,
                    "message": "ASR 推理预热中（首次语音要初始化，约 10~60 秒），请稍后重试",
                    "state": "asr_warming",
                    "retry_after_s": 10,
                })

    t0 = time.perf_counter()
    try:
        payload = await run_in_threadpool(handle_audio, session_id, audio_bytes, s)
    except Exception as exc:
        _log.exception("session=%s | A2 处理异常：%s", session_id, exc)
        return _error(E_INTERNAL, f"服务器内部错误（{type(exc).__name__}），详见服务端日志", 500)
    total_ms = _ms(t0)
    # T9：A2 分段耗时（read/asr/nlu），私有 _timing 弹掉、不进响应；
    # 假实现没有 asr/nlu 阶段，audio 名回退到客户端上传文件名。
    timing = payload.pop("_timing", None) or {}
    audio_name = timing.get("audio") or (audio.filename if audio else None) or "-"
    _log.info(
        f"session={session_id} | audio={audio_name} | "
        f"text={payload.get('text') or '-'} | target={payload.get('target') or '-'} | "
        f"need_confirm={payload.get('need_confirm', False)} | "
        f"read={timing.get('read_ms', 0)}ms asr={timing.get('asr_ms', 0)}ms "
        f"nlu={timing.get('nlu_ms', 0)}ms total={total_ms}ms | ok"
    )
    return JSONResponse(payload)


def _rss_mb():
    """进程常驻内存（MB），供 N24 长稳监控用；拿不到时返回 None（T10）。"""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:  # noqa: BLE001
        return None


def _gpu_mem_mb():
    """torch 已分配的显存（MB）；仅真引擎下有意义，拿不到返回 None（T10）。"""
    if not REAL_ENGINE:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1024 / 1024)
    except Exception:  # noqa: BLE001
        return None
    return None


@app.get("/healthz")
async def healthz():
    """健康检查（联调排障用，非契约接口）。

    T2：增加 state（warming/ready/failed），预热期返回 **503** 供客户端显示"预热中"；
    T10：增加 rss_mb（进程内存）与 gpu_mem_mb（显存），供 N24 长稳直接轮询。

    2026-11-03：预热失败时（state=failed）也返回 503 —— 原来无论成败都置 ready，
    /healthz 报 200 而引擎根本没加载，故障被推迟到第一个请求变成 500。
    """
    state = _SERVER_STATE
    body = {
        "status": "ok",
        "sessions": len(_SESSIONS),
        "real_engine": REAL_ENGINE,
        "state": state,
        "rss_mb": _rss_mb(),
        "gpu_mem_mb": _gpu_mem_mb(),
    }
    if state == "failed":
        body["warmup_error"] = _WARMUP_ERROR
    return JSONResponse(status_code=503 if state in ("warming", "failed") else 200, content=body)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    global REAL_ENGINE, _SERVER_STATE
    import uvicorn

    parser = argparse.ArgumentParser(description="EgoMed-Agent 服务端")
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址，局域网访问用 0.0.0.0（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    parser.add_argument("--real", action="store_true",
                        help="启用真引擎（默认假实现）")
    args = parser.parse_args()

    # T2：提前监听 + 后台预热。预热期 /healthz 返回 503 + state=warming，
    # 客户端据此显示"预热中"，不再盲等 60~90 秒（方案 b）。
    #
    # 2026-11-03 修正：REAL_ENGINE 在**导入期**就读了环境变量 EGOMED_REAL_ENGINE，
    # 但原来只有传 --real 才启动预热线程 → 用环境变量启用真引擎时**完全没有预热**，
    # 首个 A1/A2 会撞模型冷加载（实测 >60s）。现在两种启用方式都启动预热。
    if args.real or REAL_ENGINE:
        REAL_ENGINE = True
        _SERVER_STATE = "warming"
        threading.Thread(target=_warmup_real_engine, daemon=True).start()
        print("[server] 预热已在后台开始（/healthz 先返回 503 + state=warming）", flush=True)
    mode = "真引擎" if REAL_ENGINE else "假实现"
    print(f"[server] {mode}启动：http://{args.host}:{args.port}  （Ctrl+C 停止）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
