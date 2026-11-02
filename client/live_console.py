#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EgoMed-Agent 现场控制台（N26）—— 一个窗口完成"设置 → 预检 → 运行 → 录制"

为什么要有它
------------
以前演示要记一长串命令行（源/路径/帧率/目标/显示/麦克风），现场容易出错。
本控制台把 N25 的配置化能力包成图形界面：**选源 → 选目标 → 点开始**，全程实时看指标。

界面
----
菜单栏：文件 / 运行 / 工具 / 帮助（N53 增量 2）
┌──────────────┬───────────────────────────────┐
│ ① 启动引导    │  ③ 实时画面（服务器 overlay）   │
│   源/路径/fps │                               │
│ ② 分割目标    ├───────────────────────────────┤
│   预设按钮     │  ④ 实时指标（fps/P50/P95/错误） │
│              ├───────────────────────────────┤
│              │  ⑥ 运行日志（长任务实时输出）    │
├──────────────┴───────────────────────────────┤
│ [开始] [停止] [录制] [机位预检] [打开输出目录]  │
└──────────────────────────────────────────────┘

⑥ 运行日志（N53 增量 2）
------------------------
环境自检 / 详细检查 / 一键回归 / 长稳测试都在**子进程**里跑，输出逐行进面板：
- 子进程用 `subprocess.Popen` 起，工作线程只负责把 stdout 塞进 `queue.Queue`；
- 界面用已有的 200ms `after()` 循环取队列 → **工作线程绝不碰控件**（碰了必卡）；
- 面板限 500 行滚动，超出删头部；「停止长任务」先 CTRL_BREAK 让子进程走它自己的
  清理逻辑（关服务端），5 秒不退再 taskkill 整棵进程树。

实现要点（不重复造轮子）
------------------------
- 直接复用 `live_client.LiveRunner`：同一进程跑，指标实时读它的 `stats`；`runner.stop = True` 即停
- 目标设置走 A2（契约语义"先说后看"）：预设按钮 = 播对应样例 wav；**会话 id 由控制台生成**，
  保证"先设目标、后开始发帧"作用在同一会话上
- 录制：每秒 1 张结果图 + metrics.csv（停止时可选合成 mp4，便于当演示素材）
- 不弹 OpenCV 窗口（画面显示在控制台里），关闭窗口 = 停止运行

用法
----
    现场控制台.bat                     # 双击（推荐）
    python client/live_console.py     # 同效
    python client/live_console.py --selftest              # 只构建界面自检（CI/无人值守用）
    python client/live_console.py --selftest-job doctor   # 自检 ⑥ 日志面板（跑完打印统计）

归属：闫（客户端 / 测试工具）｜ 关联：docs/总计划.md 阶段四 N26
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import mock_client as mc  # noqa: E402
import live_client as lc  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

PROJECT_ROOT = mc.PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))          # 为了 import 根目录的 launcher（版本常量）
import egomed_config as ec  # noqa: E402      # 配置唯一读写实现（增量 3）
AUDIO_DIR = PROJECT_ROOT / "data" / "samples" / "audio"
OUT_ROOT = PROJECT_ROOT / "runs" / "console"
PREVIEW_MAX = (560, 420)      # 画面预览最大尺寸
UI_TICK_MS = 200              # 界面刷新间隔
LOG_MAX_LINES = 500           # 运行日志面板最多保留的行数
LOG_DRAIN_MAX = 200           # 单次刷新最多处理的行数（防回归刷屏把界面拖住）


def _num_str(value, fallback: str = "2") -> str:
    """把配置里的 2.0 / 2 / "2" 统一成界面上好看的 "2" 或 "2.5"。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    return str(int(f)) if f == int(f) else str(f)


def _load_versions() -> tuple[str, str]:
    """软件版本 / 契约版本。

    软件版本的唯一来源是根目录 `VERSION`（与 `launcher.py --version` 一致）；
    契约版本取 `launcher.CONTRACT_VERSION`。任何一步失败都给兜底值，不为版本号崩界面。
    """
    ver, contract = "0.2.0", "v1.0.3"
    try:                                        # 首选：和启动器保持同源
        import launcher as _launcher
        ver = getattr(_launcher, "VERSION", ver)
        contract = getattr(_launcher, "CONTRACT_VERSION", contract)
        return ver, contract
    except Exception:  # noqa: BLE001
        pass
    try:
        txt = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip().splitlines()
        if txt and txt[0].strip():
            ver = txt[0].strip()
    except Exception:  # noqa: BLE001
        pass
    return ver, contract


APP_VERSION, CONTRACT_VERSION = _load_versions()

# 长任务：GUI 只负责"起子进程 + 收输出"，业务逻辑全在既有脚本里（不重写）
JOB_SPECS = {
    "doctor": ("环境自检", ["doctor"]),
    "check": ("详细环境检查", ["check"]),
    "regress-mock": ("一键回归（mock 套件，秒级）", ["regress", "--suite", "mock"]),
    "regress-all": ("一键回归（完整，含真引擎）", ["regress"]),
    "soak": ("长稳测试（60 分钟）", ["soak", "--minutes", "60"]),
}

# 返回码 → 人话（与 launcher.py 的约定一致）
RC_LABEL = {
    0: "成功",
    1: "用户取消",
    2: "环境问题（E3xxx）",
    3: "连接问题（E1xxx）",
    4: "验收失败",
    5: "功能未实现",
}

# 预设目标：按钮名 / 样例 wav / 服务端预期识别文本
TARGET_PRESETS = [
    ("心肌", "cmd_03.wav", "分割心肌"),
    ("右心室腔", "cmd_02.wav", "把右心室腔标出来"),
    ("脾脏", "cmd_04.wav", "帮我分割脾脏"),
    ("十二指肠", "cmd_05.wav", "分割十二指肠"),
    ("息肉", "cmd_06.wav", "把息肉标出来"),
    ("左肺", "cmd_07.wav", "分割左肺"),
]
TARGET_AMBIGUOUS = [
    ("左心室（会要求澄清）", "cmd_01.wav"),
    ("肝肾（会要求澄清）", "cmd_08.wav"),
]


# ---------------------------------------------------------------- 录制
class Recorder:
    """每秒 1 张结果图 + metrics.csv；停止时按需合成 mp4。"""

    def __init__(self, out_dir: Path, make_mp4: bool = True) -> None:
        self.dir = out_dir
        self.frames_dir = out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.n = 0
        self.make_mp4 = make_mp4
        self.last = 0.0

    def tick(self, img_bgr, metrics: dict) -> None:
        now = time.time()
        if now - self.last < 1.0:
            return
        self.last = now
        self.n += 1
        p = self.frames_dir / f"frame_{self.n:05d}.jpg"
        cv2.imwrite(str(p), img_bgr)
        row = {"t": datetime.now().strftime("%H:%M:%S")}
        row.update({k: v for k, v in metrics.items()})
        self.rows.append(row)

    def close(self) -> dict:
        with (self.dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            if self.rows:
                w = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
                w.writeheader()
                w.writerows(self.rows)
        out = {"frames": self.n, "dir": str(self.dir)}
        if self.make_mp4 and self.n:
            jpgs = sorted(self.frames_dir.glob("*.jpg"))
            first = cv2.imread(str(jpgs[0]))
            if first is not None:
                h, w = first.shape[:2]
                vw = cv2.VideoWriter(str(self.dir / "recording.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (w, h))
                for j in jpgs:
                    img = cv2.imread(str(j))
                    if img is None:
                        continue
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h))
                    vw.write(img)
                vw.release()
                out["mp4"] = str(self.dir / "recording.mp4")
        (self.dir / "recording.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


# ---------------------------------------------------------------- 长任务
class JobRunner:
    """把一条命令当长任务跑：stdout 灌队列，界面自己来取（线程不碰控件）。

    为什么不用 `subprocess.run(capture_output=True)`：那样要等进程结束才拿到输出，
    回归跑 5~8 分钟期间界面一片空白，看起来就像卡死了。逐行读到就逐行显示才对。
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.queue: queue.Queue = queue.Queue()
        self.label = ""
        self.kind = ""
        self.started_at = 0.0
        self.rc: int | None = None
        self.stop_requested = False

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def elapsed(self) -> float:
        return time.time() - self.started_at if self.started_at else 0.0

    def start(self, cmd: list, label: str, kind: str, cwd: Path | None = None) -> bool:
        if self.running():
            return False
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"      # 子进程输出统一 UTF-8，避免 GBK 乱码
        env["PYTHONUNBUFFERED"] = "1"          # 不缓冲，否则日志会憋到结束才出现
        # 独立进程组：停止时能发 CTRL_BREAK，让子进程走它自己的 KeyboardInterrupt 清理
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        self.queue = queue.Queue()
        self.label, self.kind = label, kind
        self.rc = None
        self.stop_requested = False
        self.started_at = time.time()
        try:
            self.proc = subprocess.Popen(
                [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, env=env, creationflags=flags)
        except Exception as exc:  # noqa: BLE001
            self.proc = None
            self.queue.put(f"[!!] 子进程启动失败：{type(exc).__name__}: {exc}")
            self.queue.put(("exit", -1))
            return False
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        return True

    def _pump(self) -> None:
        """工作线程：只做"读一行、塞队列"，不碰任何控件。"""
        p = self.proc
        if p is None or p.stdout is None:
            self.queue.put(("exit", -1))
            return
        try:
            for line in p.stdout:
                self.queue.put(line.rstrip("\r\n"))
        except Exception as exc:  # noqa: BLE001
            self.queue.put(f"[!!] 读取输出失败：{type(exc).__name__}: {exc}")
        finally:
            try:
                rc = p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                rc = -1
            self.rc = rc                                  # 先落 rc，再投哨兵（顺序别反）
            self.queue.put(("exit", rc))

    def stop(self, wait_s: float = 5.0) -> None:
        """先礼后兵：CTRL_BREAK 让子进程自己收尾（关服务端），超时才杀进程树。"""
        p = self.proc
        if p is None or p.poll() is not None:
            return
        self.stop_requested = True
        try:
            if os.name == "nt":
                os.kill(p.pid, signal.CTRL_BREAK_EVENT)   # type: ignore[attr-defined]
            else:
                p.terminate()
            p.wait(timeout=wait_s)
        except Exception:  # noqa: BLE001
            self._kill_tree()

    def _kill_tree(self) -> None:
        p = self.proc
        if p is None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True)
            else:
                p.kill()
        except Exception:  # noqa: BLE001
            pass

    def drain(self) -> list:
        """取出已就绪的输出（str 行 / ("exit", rc) 哨兵）。"""
        items = []
        while len(items) < LOG_DRAIN_MAX:
            try:
                items.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return items


# ---------------------------------------------------------------- 控制台
class Console:
    def __init__(self, root: tk.Tk, doctor: str = "console") -> None:
        self.root = root
        self.session_id = mc.make_session_id(doctor)
        self.out_root = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cfg, self.cfg_notes = self._read_cfg()   # 增量 3：配置即界面初值
        self.runner: lc.LiveRunner | None = None
        self.run_thread: threading.Thread | None = None
        self.recorder: Recorder | None = None
        self.preflight_thread: threading.Thread | None = None
        self._preview_size = (0, 0)
        self._last_counters = (0, 0, 0.0)
        self._fps_submit = 0.0
        self._fps_refresh = 0.0
        self._ticks = 0          # UI 刷新次数（自测用）
        self._ticks_img = 0      # 拿到画面的次数
        self._rec_ticks = 0      # 进入录制分支的次数
        self._ticks_runner = 0   # runner 非空的刷新次数
        self._ticks_running = 0  # runner 非空且 overlay 已超过 0 的次数
        self._fps_last_t = 0.0   # 上次算速率的时刻（≥1s 窗口）
        self.job = JobRunner()   # 长任务（doctor/check/regress/soak）子进程
        self.log_lines = 0       # 日志面板当前行数（用于限 500 行）

        root.title(f"EgoMed-Agent 现场控制台（N26）  v{APP_VERSION}")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self._build_menu()
        self.root.after(UI_TICK_MS, self._tick)   # 启动界面刷新循环（自刷新）
        self._log_line(f"[--] 控制台已启动  v{APP_VERSION}（契约 {CONTRACT_VERSION}）"
                       f"  项目根 {PROJECT_ROOT}")

    # ------------------------------------------------------------ 界面
    def _build(self) -> None:
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ① 启动引导（初值来自 config.toml 的 [ui] 段 —— 增量 3）
        ui = self.cfg.get("ui", {}) if isinstance(self.cfg, dict) else {}
        ttk.Label(left, text="① 启动引导", font=("Microsoft YaHei", 11, "bold")).pack(anchor="w")
        self.var_source = tk.StringVar(value=str(ui.get("source") or "video"))
        for text, val in (("视频/帧目录（自动化）", "video"), ("本机摄像头", "cam"),
                          ("网络流 URL（手机当镜头）", "url")):
            ttk.Radiobutton(left, text=text, value=val, variable=self.var_source).pack(anchor="w")
        ttk.Label(left, text="路径 / URL").pack(anchor="w", pady=(6, 0))
        self.var_input = tk.StringVar(value=str(ui.get("input") or "data/samples/acdc"))
        ttk.Entry(left, textvariable=self.var_input, width=34).pack(anchor="w")
        row = ttk.Frame(left)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Label(row, text="提交 fps").pack(side="left")
        self.var_fps = tk.StringVar(value=_num_str(ui.get("submit_fps")))
        ttk.Entry(row, textvariable=self.var_fps, width=5).pack(side="left", padx=4)
        ttk.Label(row, text="显示").pack(side="left")
        self.var_display = tk.StringVar(value=str(ui.get("display") or "result"))
        ttk.Combobox(row, textvariable=self.var_display, width=8, state="readonly",
                     values=["result", "blend"]).pack(side="left", padx=4)
        self.var_mic = tk.BooleanVar(value=bool(ui.get("mic", False)))
        ttk.Checkbutton(left, text="开启麦克风（说话切换目标）", variable=self.var_mic).pack(
            anchor="w", pady=(4, 0))
        ttk.Button(left, text="机位校准（引导式，8 帧）",
                   command=self.on_preflight).pack(anchor="w", pady=(6, 0))

        # ② 分割目标
        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, text="② 分割目标（契约：先设目标再发帧）",
                  font=("Microsoft YaHei", 11, "bold")).pack(anchor="w")
        grid = ttk.Frame(left)
        grid.pack(anchor="w", pady=4)
        for i, (name, wav, expect) in enumerate(TARGET_PRESETS):
            b = ttk.Button(grid, text=name, width=11,
                           command=lambda w=wav, e=expect: self.on_set_target(w, e))
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2)
        ttk.Label(left, text="（下列句子跨模态歧义，服务端会要求澄清）",
                  foreground="#888").pack(anchor="w", pady=(6, 0))
        for name, wav in TARGET_AMBIGUOUS:
            ttk.Button(left, text=name, width=23,
                       command=lambda w=wav: self.on_set_target(w, "（预期：需澄清）")
                       ).pack(anchor="w", pady=1)

        # ③ 实时画面
        self.canvas = tk.Label(right, text="等待开始…", width=72, height=22,
                               background="#202020", foreground="#aaa")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # ④ 指标
        self.var_metrics = tk.StringVar(value="指标：未开始")
        ttk.Label(right, textvariable=self.var_metrics, font=("Consolas", 10),
                  justify="left").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.var_msg = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.var_msg, foreground="#0a5").grid(
            row=2, column=0, sticky="w")

        # ⑥ 运行日志（长任务实时输出；工具菜单可折叠）
        self.log_box = ttk.Labelframe(right, text="⑥ 运行日志", padding=3)
        self.log_box.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        self.log_text = tk.Text(self.log_box, height=8, wrap="none", state="disabled",
                                font=("Consolas", 9), background="#101010",
                                foreground="#d0d0d0", insertbackground="#d0d0d0")
        sb = ttk.Scrollbar(self.log_box, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.log_box.columnconfigure(0, weight=1)
        self.log_box.rowconfigure(0, weight=1)

        # 底部按钮
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.btn_start = ttk.Button(bar, text="开始", command=self.on_start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="停止", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.var_rec = tk.BooleanVar(value=False)
        self.btn_rec = ttk.Button(bar, text="录制：关", command=self.on_toggle_record,
                                  state="disabled")
        self.btn_rec.pack(side="left", padx=4)
        ttk.Button(bar, text="打开输出目录", command=self.on_open_dir).pack(side="left", padx=4)
        ttk.Button(bar, text="导出诊断包", command=self.on_diag_pack).pack(side="left", padx=4)
        ttk.Label(bar, text=f"会话 {self.session_id}").pack(side="right")

    # ------------------------------------------------------------ 菜单
    def _build_menu(self) -> None:
        """菜单栏：把原来"要记命令行"的操作收进来，和旧 bat 一一对应。"""
        menubar = tk.Menu(self.root)

        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="设置…（服务端 / 压缩 / 回显 / 日志）", command=self.on_settings)
        m_file.add_command(label="打开配置文件（config.toml）", command=self.on_open_config)
        m_file.add_command(label="重新载入配置", command=self.on_reload_config)
        m_file.add_separator()
        m_file.add_command(label="打开输出目录", command=self.on_open_dir)
        m_file.add_command(label="导出诊断包（含版本与配置）", command=self.on_diag_pack)
        m_file.add_separator()
        m_file.add_command(label="退出（自动记住当前选择）", command=self._on_close)
        menubar.add_cascade(label="文件", menu=m_file)

        m_run = tk.Menu(menubar, tearoff=0)
        m_run.add_command(label="开始运行", command=self.on_start)
        m_run.add_command(label="停止运行", command=self.on_stop)
        m_run.add_command(label="录制开关", command=self.on_toggle_record)
        m_run.add_separator()
        for kind in ("regress-mock", "regress-all", "soak"):
            m_run.add_command(label=JOB_SPECS[kind][0],
                              command=lambda k=kind: self.on_run_job(k))
        m_run.add_separator()
        m_run.add_command(label="停止长任务", command=self.on_stop_job)
        menubar.add_cascade(label="运行", menu=m_run)

        m_tool = tk.Menu(menubar, tearoff=0)
        for kind in ("doctor", "check"):
            m_tool.add_command(label=JOB_SPECS[kind][0],
                               command=lambda k=kind: self.on_run_job(k))
        m_tool.add_command(label="机位校准（引导式，8 帧）", command=self.on_preflight)
        m_tool.add_separator()
        self.var_show_log = tk.BooleanVar(value=True)
        m_tool.add_checkbutton(label="显示运行日志面板", variable=self.var_show_log,
                               command=self.on_toggle_log_panel)
        m_tool.add_command(label="清空日志", command=self.on_clear_log)
        menubar.add_cascade(label="工具", menu=m_tool)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="关于", command=self.on_about)
        m_help.add_command(label="查看交付流程文档",
                           command=lambda: self.on_open_doc("docs/软件化交付流程.md"))
        menubar.add_cascade(label="帮助", menu=m_help)

        self.root.config(menu=menubar)

    # ------------------------------------------------------------ 配置（增量 3）
    def _read_cfg(self) -> tuple[dict, list[str]]:
        """读四层里"默认 + 文件 + 环境变量"三层；读不到也不许崩。"""
        try:
            return ec.load(PROJECT_ROOT)
        except Exception as exc:  # noqa: BLE001
            return ec.default_config(), [f"配置读取失败（{exc}），已用内置默认值"]

    def _ui_values(self) -> dict:
        try:
            fps = float(self.var_fps.get().strip() or "2")
        except ValueError:
            fps = 2.0
        return {
            "source": self.var_source.get(),
            "input": self.var_input.get().strip(),
            "submit_fps": fps,
            "display": self.var_display.get(),
            "mic": bool(self.var_mic.get()),
        }

    def _save_ui(self, quiet: bool = False) -> None:
        """把上次的界面选择写回 `[ui]`（只动这一段，不碰 server/capture）。"""
        try:
            path = ec.save_ui(PROJECT_ROOT, self._ui_values())
            if not quiet:
                self._log_line(f"[OK] 界面设置已保存：{path}")
        except Exception as exc:  # noqa: BLE001
            self._log_line(f"[!!] 保存界面设置失败：{exc}")

    def on_reload_config(self) -> None:
        self.cfg, notes = self._read_cfg()
        ui = self.cfg.get("ui", {})
        self.var_source.set(str(ui.get("source") or "video"))
        self.var_input.set(str(ui.get("input") or "data/samples/acdc"))
        self.var_fps.set(_num_str(ui.get("submit_fps")))
        self.var_display.set(str(ui.get("display") or "result"))
        self.var_mic.set(bool(ui.get("mic", False)))
        for n in notes:
            self._log_line(f"[--] {n}")
        self.var_msg.set("配置已重新载入")

    def on_open_config(self) -> None:
        path = ec.config_path(PROJECT_ROOT)
        if not path.is_file():
            self._save_ui(quiet=True)      # 没有就先落一份，别让用户对着空目录
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            self._log_line(f"[--] 配置文件路径：{path}")

    def on_settings(self) -> None:
        """设置窗口：只改"软件级"配置（服务端/压缩/回显/日志），界面选择走 `[ui]`。"""
        cfg = self.cfg if isinstance(self.cfg, dict) else {}
        win = tk.Toplevel(self.root)
        win.title("设置（写入 config.toml）")
        win.transient(self.root)
        win.resizable(False, False)

        srv = cfg.get("server", {})
        cap = cfg.get("capture", {})
        dsp = cfg.get("display", {})
        log = cfg.get("logging", {})
        fields = [
            ("服务端地址", "server.url", str(srv.get("url", "http://127.0.0.1:8000"))),
            ("请求超时（秒）", "server.timeout_s", str(srv.get("timeout_s", 3.0))),
            ("JPEG 质量（1~100）", "capture.jpeg_quality", str(cap.get("jpeg_quality", 80))),
            ("回显方式", "display.renderer", str(dsp.get("renderer", "window"))),
            ("日志级别", "logging.level", str(log.get("level", "info"))),
        ]
        vars_: dict[str, tk.StringVar] = {}
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        for i, (label, key, val) in enumerate(fields):
            ttk.Label(body, text=label).grid(row=i, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=val)
            vars_[key] = v
            if key == "display.renderer":
                ttk.Combobox(body, textvariable=v, width=22, state="readonly",
                             values=["window", "file", "glasses-mono", "null"]
                             ).grid(row=i, column=1, sticky="w", padx=6)
            elif key == "logging.level":
                ttk.Combobox(body, textvariable=v, width=22, state="readonly",
                             values=["debug", "info", "warn", "error"]
                             ).grid(row=i, column=1, sticky="w", padx=6)
            else:
                ttk.Entry(body, textvariable=v, width=24).grid(row=i, column=1, sticky="w", padx=6)
        ttk.Label(body, text=f"配置文件：{ec.config_path(PROJECT_ROOT)}",
                  foreground="#666").grid(row=len(fields), column=0, columnspan=2,
                                          sticky="w", pady=(8, 0))

        def _apply() -> None:
            try:
                sections = {
                    "server": {"url": vars_["server.url"].get().strip(),
                               "timeout_s": float(vars_["server.timeout_s"].get())},
                    "capture": {"jpeg_quality": int(vars_["capture.jpeg_quality"].get())},
                    "display": {"renderer": vars_["display.renderer"].get()},
                    "logging": {"level": vars_["logging.level"].get()},
                }
            except ValueError as exc:
                messagebox.showerror("数值不合法", f"{exc}", parent=win)
                return
            try:
                path = ec.save_sections(PROJECT_ROOT, sections)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("保存失败", f"{exc}", parent=win)
                return
            self.cfg, _notes = self._read_cfg()
            self._log_line(f"[OK] 设置已保存：{path}")
            self.var_msg.set("设置已保存（下次运行生效）")
            win.destroy()

        bar = ttk.Frame(win, padding=(10, 0, 10, 10))
        bar.pack(fill="x")
        ttk.Button(bar, text="保存", command=_apply).pack(side="right")
        ttk.Button(bar, text="取消", command=win.destroy).pack(side="right", padx=6)

    # ------------------------------------------------------------ 动作
    def _build_args(self, overrides: dict) -> argparse.Namespace:
        argv = ["--source", self.var_source.get(),
                "--input", self.var_input.get().strip() or "data/samples/acdc",
                "--fps", self.var_fps.get().strip() or "2",
                "--display", self.var_display.get(),
                "--session", self.session_id,
                "--no-show"]      # 画面显示在控制台里，不另开 OpenCV 窗口
        if self.var_mic.get():
            argv.append("--mic")
        for k, v in overrides.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return lc.parse_args(argv)

    def on_start(self) -> None:
        if self.runner is not None:
            return
        try:
            args = self._build_args({"out": str(self.out_root / "run")})
        except SystemExit:      # argparse 报错时会 exit
            return
        self.runner = lc.LiveRunner(args)
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_rec.config(state="normal")
        self.var_msg.set("运行中…（关闭窗口 = 停止）")

        def _run():
            try:
                self.runner.run()
            except Exception as exc:  # noqa: BLE001
                self.var_msg.set(f"运行异常：{exc}")
            finally:
                self.var_msg.set("已停止")

        self.run_thread = threading.Thread(target=_run, daemon=True)
        self.run_thread.start()

    def on_stop(self) -> None:
        if self.runner is not None:
            self.runner.stop = True
            self.var_msg.set("正在停止…")

    def on_toggle_record(self) -> None:
        if self.var_rec.get():
            self.var_rec.set(False)
            self.btn_rec.config(text="录制：关")
            if self.recorder is not None:
                info = self.recorder.close()
                self.var_msg.set(f"录制已保存：{info.get('mp4') or info['dir']}"
                                 f"（{info['frames']} 帧）")
                self.recorder = None
        else:
            self.var_rec.set(True)
            self.btn_rec.config(text="录制：开")
            self.recorder = Recorder(self.out_root / "recording")
            self.var_msg.set("开始录制（每秒 1 帧 + metrics.csv）")

    def on_set_target(self, wav_name: str, expect: str) -> None:
        wav = AUDIO_DIR / wav_name
        if not wav.exists():
            messagebox.showerror("缺少样例", f"找不到 {wav}")
            return
        url = (self.runner.args.url if self.runner else
               lc.parse_args(["--no-show"]).url)
        self.var_msg.set(f"正在设置目标（{wav_name} → 预期「{expect}」）…")

        def _work():
            payload, elapsed, err = mc.post_audio(url, self.session_id,
                                                   wav.read_bytes(), wav_name, 20.0)
            if err:
                self.var_msg.set(f"设置目标失败：{err}（服务端起了吗？）")
                return
            text = (payload or {}).get("text")
            target = (payload or {}).get("target")
            msg = (payload or {}).get("message")
            if target:
                self.var_msg.set(f"目标已切换 → {target}（识别「{text}」，{elapsed:.0f}ms）")
            else:
                self.var_msg.set(f"未设上目标：{msg or text}")

        threading.Thread(target=_work, daemon=True).start()

    def on_preflight(self) -> None:
        """N36：调用「引导式机位校准」工具，把结论显示在状态栏。

        复用同一个 `client/calibrate.py`（命令行也能单独跑），保证两条入口结论一致。
        """
        if self.preflight_thread is not None:
            return
        self.var_msg.set("机位校准中（抽 8 帧，约 5~10 秒）…")

        def _work():
            script = Path(__file__).resolve().parent / "calibrate.py"
            cmd = [sys.executable, str(script),
                   "--source", self.var_source.get(),
                   "--input", self.var_input.get().strip() or "data/samples/acdc",
                   "--frames", "8"]
            if self.runner is not None:
                cmd += ["--url", self.runner.args.url]
            r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                               encoding="utf-8", errors="ignore")
            tail = [l.strip() for l in (r.stdout or "").splitlines() if l.strip()]
            verdict = next((l for l in reversed(tail) if l.startswith("结论")), "")
            advice = [l for l in tail if l.startswith("→")]
            msg = verdict + (f"；{advice[0]}" if advice else "")
            self.var_msg.set(msg[:170] or f"校准失败（退出码 {r.returncode}）")
            self.preflight_thread = None

        self.preflight_thread = threading.Thread(target=_work, daemon=True)
        self.preflight_thread.start()

    def on_diag_pack(self) -> None:
        """N28：一键导出诊断包（环境/配置/日志/报告/帧 → zip）。"""
        self.var_msg.set("正在生成诊断包（约 10~20 秒）…")

        def _work():
            script = Path(__file__).resolve().parent / "make_diag_pack.py"
            r = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT),
                               capture_output=True, text=True, encoding="utf-8", errors="ignore")
            tail = [l for l in (r.stdout or "").splitlines() if l.strip()]
            self.var_msg.set(f"诊断包：{tail[-1] if tail else f'失败（退出码 {r.returncode}）'}")

        threading.Thread(target=_work, daemon=True).start()

    def on_open_dir(self) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.out_root))  # type: ignore[attr-defined]
        except Exception:
            self.var_msg.set(f"输出目录：{self.out_root}")

    # ------------------------------------------------------------ 日志与长任务
    def _log_line(self, text: str) -> None:
        """往 ⑥ 面板追加一行；超过 LOG_MAX_LINES 从头部删（只增不删会越跑越卡）。"""
        t = self.log_text
        t.configure(state="normal")
        t.insert("end", text + "\n")
        self.log_lines += 1
        if self.log_lines > LOG_MAX_LINES:
            drop = self.log_lines - LOG_MAX_LINES
            t.delete("1.0", f"{drop + 1}.0")
            self.log_lines = LOG_MAX_LINES
        t.see("end")
        t.configure(state="disabled")

    def on_clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.log_lines = 0

    def on_toggle_log_panel(self) -> None:
        if self.var_show_log.get():
            self.log_box.grid()
        else:
            self.log_box.grid_remove()

    def on_run_job(self, kind: str) -> None:
        """跑一个长任务：统一走 `launcher.py` 子命令（门面唯一，错误码才一致）。"""
        spec = JOB_SPECS.get(kind)
        if spec is None:
            return
        label, argv = spec
        if self.job.running():
            messagebox.showinfo("已有长任务在跑",
                                f"当前正在跑：{self.job.label}\n"
                                "先「运行 → 停止长任务」再试。")
            return
        launcher_py = PROJECT_ROOT / "launcher.py"
        if not launcher_py.exists():
            messagebox.showerror("缺少启动器", f"找不到 {launcher_py}")
            return
        self.var_show_log.set(True)
        self.on_toggle_log_panel()
        self._log_line("")
        self._log_line(f"==== [{datetime.now():%H:%M:%S}] {label} ====")
        self._log_line(f"$ {Path(sys.executable).name} launcher.py {' '.join(argv)}")
        if not self.job.start([sys.executable, str(launcher_py)] + argv,
                              label, kind, cwd=PROJECT_ROOT):
            return
        self.var_msg.set(f"{label}：已启动（输出见 ⑥ 运行日志）")

    def on_stop_job(self) -> None:
        if not self.job.running():
            self._log_line("[--] 当前没有在跑的长任务")
            return
        self._log_line(f"[--] 正在停止「{self.job.label}」（先 CTRL_BREAK，5 秒不退则杀进程树）…")
        self.job.stop()
        self.var_msg.set(f"{self.job.label}：正在停止…")

    def _drain_job(self) -> None:
        """界面刷新时调：把子进程已产出的行搬进面板（**只在这里碰控件**）。"""
        for item in self.job.drain():
            if isinstance(item, tuple) and item and item[0] == "exit":
                self._finish_job(int(item[1]))
            else:
                self._log_line(str(item))

    def _finish_job(self, rc: int) -> None:
        job = self.job
        verdict = self._verdict(job.kind, rc, job.stop_requested)
        self._log_line(f"[结束] {job.label}：{verdict}（退出码 {rc}，"
                       f"用时 {job.elapsed():.1f}s）")
        self.var_msg.set(f"{job.label}：{verdict}（退出码 {rc}）")

    @staticmethod
    def _verdict(kind: str, rc: int, stopped: bool) -> str:
        """返回码翻成人话：现场不需要知道数字含义，只需要知道"过了没、去哪看"。"""
        if stopped and rc != 0:
            return "已由用户停止"
        base = kind.split("-")[0]
        if rc == 0:
            return {"doctor": "自检通过", "check": "详细检查通过",
                    "regress": "回归通过", "soak": "长稳完成"}.get(base, "成功")
        if base == "doctor":
            return "自检发现问题（见上方 [!!] 行）"
        if base == "check":
            return "检查未通过（见上方 FAIL 行）"
        if base == "regress":
            return "回归未全通过，报告见 runs/acceptance/<时间戳>/report.md"
        if base == "soak":
            return "长稳未达标，报告见 runs/soak/<时间戳>/"
        return RC_LABEL.get(rc, f"退出码 {rc}")

    def on_about(self) -> None:
        cfg, note = PROJECT_ROOT / "config.toml", ""
        url = "http://127.0.0.1:8000"
        try:
            import tomllib
            if cfg.is_file():
                with cfg.open("rb") as fh:
                    url = tomllib.load(fh).get("server", {}).get("url", url)
                note = str(cfg)
            else:
                note = f"{cfg}（缺失，用内置默认值）"
        except Exception as exc:  # noqa: BLE001
            note = f"{cfg}（解析失败：{exc}）"
        messagebox.showinfo(
            "关于 EgoMed-Agent",
            "\n".join([
                "EgoMed-Agent 现场控制台（N26 / N53）",
                f"软件版本：v{APP_VERSION}",
                f"契约版本：{CONTRACT_VERSION}",
                f"Python：{sys.executable}",
                f"项目根：{PROJECT_ROOT}",
                f"服务端：{url}",
                f"配置：{note}",
                f"输出：{self.out_root}",
            ]))

    def on_open_doc(self, rel: str) -> None:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            self._log_line(f"[!!] 文档不存在：{path}")
            return
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            self._log_line(f"[--] 文档路径：{path}")

    def _on_close(self) -> None:
        # 增量 3：关窗口即记住这次的选择（下次打开自动恢复）
        self._save_ui(quiet=True)
        # 长任务不跟着界面一起"悄悄消失"：它在后台起了服务端，必须一起收掉
        if self.job.running():
            self.job.stop()
        if self.runner is not None:
            self.runner.stop = True
            time.sleep(0.3)
        if self.recorder is not None:
            self.recorder.close()
        self.root.destroy()

    # ------------------------------------------------------------ 刷新
    def _tick(self) -> None:
        """刷新循环外壳：**任何异常都不许打断刷新链**（否则界面会"卡死不再更新"）。"""
        try:
            self._tick_once()
        except Exception as exc:  # noqa: BLE001
            if not getattr(self, "_tick_err_logged", False):
                self._tick_err_logged = True
                mc.log(f"[tick] 刷新异常（已忽略，继续刷新）：{type(exc).__name__}: {exc}")
        finally:
            self.root.after(UI_TICK_MS, self._tick)

    def _tick_once(self) -> None:
        self._ticks += 1
        self._drain_job()        # ⑥ 面板：先把长任务已产出的行搬上来
        st = self.runner.stats if self.runner is not None else None
        if st is not None:
            now = time.perf_counter()
            # 速率用 ≥1s 的窗口算，避免 200ms 窗口把"某帧刚好回来"放大成 5 fps 的假读数
            if now - self._fps_last_t >= 1.0:
                sent0, ok0, t0 = self._last_counters
                if t0 and now - t0 > 0:
                    self._fps_submit = (st.sent - sent0) / (now - t0)
                    self._fps_refresh = (st.overlay_nonempty - ok0) / (now - t0)
                self._last_counters = (st.sent, st.overlay_nonempty, now)
                self._fps_last_t = now
            self.var_metrics.set(
                f"提交 {self._fps_submit:4.2f} fps | 刷新 {self._fps_refresh:4.2f} fps | "
                f"P50 {st.pct(50):5.0f} / P95 {st.pct(95):5.0f} ms | 错误 "
                f"http {st.http_err} / 业务 {st.biz_err}\n"
                f"已提交 {st.submitted} | 已发 {st.sent} | 丢弃 {st.dropped} | "
                f"overlay 非空 {st.overlay_nonempty} | 坏帧 {st.bad_frame} | 未设目标 {st.no_target} | "
                f"语音 {st.voice_count}（需澄清 {st.need_confirm_count} / "
                f"拒识 {st.voice_rejected}）\n"
                f"当前目标 {st.last_target or '—'} | 模态 {st.last_modality or '—'}")
            self.var_msg.set(self.runner.last_message or self.var_msg.get())
            # 画面
            self._ticks_runner += 1
            if st.overlay_nonempty:
                self._ticks_running += 1
            img = self.runner.last_result
            if img is not None:
                self._ticks_img += 1
                self._show(img)
            if self.recorder is not None and img is not None:
                self._rec_ticks += 1
                self.recorder.tick(img, {"fps_submit": round(self._fps_submit, 2),
                                         "fps_refresh": round(self._fps_refresh, 2),
                                         "p50": round(st.pct(50), 1), "p95": round(st.pct(95), 1),
                                         "err_http": st.http_err, "err_biz": st.biz_err,
                                         "target": st.last_target})
            if self.run_thread is not None and not self.run_thread.is_alive():
                self.btn_start.config(state="normal")
                self.btn_stop.config(state="disabled")
                self.runner = None
                self.run_thread = None

    def _show(self, img_bgr) -> None:
        h, w = img_bgr.shape[:2]
        scale = min(PREVIEW_MAX[0] / w, PREVIEW_MAX[1] / h, 1.0)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if size != (w, h):
            pil = pil.resize(size)
        self._preview = ImageTk.PhotoImage(pil)   # 保持引用，否则图会消失
        self.canvas.config(image=self._preview, width=size[0], height=size[1])


def main(argv=None) -> int:
    try:                      # .bat 启动时控制台可能是 GBK，先尽量切到 UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="EgoMed-Agent 现场控制台（N26）")
    ap.add_argument("--doctor", default="console", help="医生编号（用于生成 session_id）")
    ap.add_argument("--selftest", action="store_true", help="只构建界面自检（无人值守）")
    ap.add_argument("--autotest", type=float, default=0.0, metavar="SEC",
                    help="端到端自测：自动设目标 -> 开始 -> 录制 -> 跑 SEC 秒 -> 停止并打印指标")
    ap.add_argument("--selftest-job", default="", metavar="KIND",
                    help="无人值守验证 ⑥ 日志面板：自动跑一个长任务 "
                         "(doctor / check / regress-mock)，结束打印日志统计")
    args = ap.parse_args(argv)

    root = tk.Tk()
    con = Console(root, args.doctor)
    if args.selftest:
        info: dict = {}

        def _collect_and_quit() -> None:
            info["menu"] = bool(root["menu"])
            info["log"] = con.log_box.winfo_manager() or "未显示"
            info["lines"] = con.log_lines
            info["ui"] = (con.var_source.get(), con.var_input.get(), con.var_fps.get(),
                          con.var_display.get(), con.var_mic.get())
            info["cfg"] = str(ec.config_path(PROJECT_ROOT))
            info["notes"] = list(con.cfg_notes)
            root.destroy()

        root.after(600, _collect_and_quit)
        root.mainloop()
        mc.log(f"[selftest] 界面构建与刷新正常 | 软件 v{APP_VERSION} 契约 {CONTRACT_VERSION} | "
               f"菜单栏 {'有' if info.get('menu') else '无'} | "
               f"日志面板 {info.get('log')} | 日志 {info.get('lines')} 行")
        mc.log(f"[selftest] 配置 {info.get('cfg')} | 界面初值 source={info['ui'][0]!r} "
               f"input={info['ui'][1]!r} fps={info['ui'][2]!r} display={info['ui'][3]!r} "
               f"mic={info['ui'][4]}")
        for n in info.get("notes", []):
            mc.log(f"[selftest]   {n}")
        return 0
    if args.selftest_job:
        con.on_run_job(args.selftest_job)
        deadline = time.time() + 240

        def _watch() -> None:
            if con.job.running():
                if time.time() > deadline:
                    mc.log("[selftest-job] 超时未结束，强制停止")
                    con.job.stop()
                    root.after(1500, _report_job)
                else:
                    root.after(300, _watch)
                return
            if not con.job.started_at:
                mc.log("[selftest-job] 子进程未能启动")
                root.destroy()
                return
            root.after(800, _report_job)     # 留一个刷新周期把残余输出搬完

        def _report_job() -> None:
            tail = [ln for ln in con.log_text.get("end-6l", "end-1c").splitlines() if ln.strip()]
            mc.log(f"[selftest-job] {con.job.label} | 退出码 {con.job.rc} | "
                   f"日志 {con.log_lines} 行 | 停止标志 {con.job.stop_requested}")
            for line in tail[-4:]:
                mc.log(f"[selftest-job]   {line}")
            root.destroy()

        root.after(400, _watch)
        root.mainloop()
        return 0
    if args.autotest > 0:
        # 自动走一遍完整流程，最后打印指标（供无人值守验证 GUI 逻辑）
        con.var_source.set("video")
        con.var_input.set("data/samples/acdc")
        con.var_fps.set("2")

        def _finish() -> None:
            st = con.runner.stats if con.runner is not None else None
            imgref = con.runner.last_result if con.runner is not None else None
            if con.recorder is not None:
                con.on_toggle_record()          # 关录制（落盘 mp4/csv）
            con.on_stop()

            def _report() -> None:
                if st is None:
                    mc.log("[autotest] 未取到指标")
                else:
                    mc.log("[autotest] 提交 %d | 已发 %d | overlay 非空 %d | P50 %.0fms P95 %.0fms | "
                           "http 错 %d / 业务错 %d | 目标 %r"
                           % (st.submitted, st.sent, st.overlay_nonempty, st.pct(50), st.pct(95),
                              st.http_err, st.biz_err, st.last_target))
                mc.log("[autotest] UI 刷新 %d 次 / runner 非空 %d 次 / 已有 overlay %d 次 / "
                       "拿到画面 %d 次 / 录制分支 %d 次"
                       % (con._ticks, con._ticks_runner, con._ticks_running,
                          con._ticks_img, con._rec_ticks))
                lr = con.runner.last_result if con.runner is not None else imgref
                mc.log("[autotest] 收尾时 last_result = %s"
                       % ("None" if lr is None else f"ndarray{getattr(lr, 'shape', '')}"))
                root.destroy()

            root.after(2000, _report)

        root.after(500, lambda: con.on_set_target("cmd_03.wav", "分割心肌"))
        root.after(3000, con.on_start)
        root.after(3500, con.on_toggle_record)
        root.after(3000 + int(args.autotest * 1000), _finish)
        root.mainloop()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
