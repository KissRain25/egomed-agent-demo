#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EgoMed-Agent 现场控制台（N26）—— 一个窗口完成"设置 → 预检 → 运行 → 录制"

为什么要有它
------------
以前演示要记一长串命令行（源/路径/帧率/目标/显示/麦克风），现场容易出错。
本控制台把 N25 的配置化能力包成图形界面：**选源 → 选目标 → 点开始**，全程实时看指标。

界面
----
┌──────────────┬───────────────────────────────┐
│ ① 启动引导    │  ③ 实时画面（服务器 overlay）   │
│   源/路径/fps │                               │
│ ② 分割目标    ├───────────────────────────────┤
│   预设按钮     │  ④ 实时指标（fps/P50/P95/错误） │
├──────────────┴───────────────────────────────┤
│ [开始] [停止] [录制] [机位预检] [打开输出目录]  │
└──────────────────────────────────────────────┘

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
    python client/live_console.py --selftest   # 只构建界面自检（CI/无人值守用）

归属：闫（客户端 / 测试工具）｜ 关联：docs/总计划.md 阶段四 N26
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
AUDIO_DIR = PROJECT_ROOT / "data" / "samples" / "audio"
OUT_ROOT = PROJECT_ROOT / "runs" / "console"
PREVIEW_MAX = (560, 420)      # 画面预览最大尺寸
UI_TICK_MS = 200              # 界面刷新间隔

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


# ---------------------------------------------------------------- 控制台
class Console:
    def __init__(self, root: tk.Tk, doctor: str = "console") -> None:
        self.root = root
        self.session_id = mc.make_session_id(doctor)
        self.out_root = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
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

        root.title("EgoMed-Agent 现场控制台（N26）")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self.root.after(UI_TICK_MS, self._tick)   # 启动界面刷新循环（自刷新）

    # ------------------------------------------------------------ 界面
    def _build(self) -> None:
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ① 启动引导
        ttk.Label(left, text="① 启动引导", font=("Microsoft YaHei", 11, "bold")).pack(anchor="w")
        self.var_source = tk.StringVar(value="video")
        for text, val in (("视频/帧目录（自动化）", "video"), ("本机摄像头", "cam"),
                          ("网络流 URL（手机当镜头）", "url")):
            ttk.Radiobutton(left, text=text, value=val, variable=self.var_source).pack(anchor="w")
        ttk.Label(left, text="路径 / URL").pack(anchor="w", pady=(6, 0))
        self.var_input = tk.StringVar(value="data/samples/acdc")
        ttk.Entry(left, textvariable=self.var_input, width=34).pack(anchor="w")
        row = ttk.Frame(left)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Label(row, text="提交 fps").pack(side="left")
        self.var_fps = tk.StringVar(value="2")
        ttk.Entry(row, textvariable=self.var_fps, width=5).pack(side="left", padx=4)
        ttk.Label(row, text="显示").pack(side="left")
        self.var_display = tk.StringVar(value="result")
        ttk.Combobox(row, textvariable=self.var_display, width=8, state="readonly",
                     values=["result", "blend"]).pack(side="left", padx=4)
        self.var_mic = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="开启麦克风（说话切换目标）", variable=self.var_mic).pack(
            anchor="w", pady=(4, 0))
        ttk.Button(left, text="机位预检（发 6 帧看有没有目标）",
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
        ttk.Label(bar, text=f"会话 {self.session_id}").pack(side="right")

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
        if self.preflight_thread is not None:
            return
        try:
            args = self._build_args({"out": str(self.out_root / "preflight"), "max_frames": 6})
        except SystemExit:
            return
        self.var_msg.set("机位预检：连发 6 帧，看是否出现有效 overlay …")

        def _work():
            r = lc.LiveRunner(args)
            try:
                r.run()
            except Exception as exc:  # noqa: BLE001
                self.var_msg.set(f"预检异常：{exc}")
                return
            ok = r.stats.overlay_nonempty
            verdict = "通过 ✓" if ok > 0 else "未检测到目标 ✗（对准影像 / 调机位）"
            self.var_msg.set(f"机位预检：overlay 非空 {ok}/6 → {verdict}")
            self.preflight_thread = None

        self.preflight_thread = threading.Thread(target=_work, daemon=True)
        self.preflight_thread.start()

    def on_open_dir(self) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.out_root))  # type: ignore[attr-defined]
        except Exception:
            self.var_msg.set(f"输出目录：{self.out_root}")

    def _on_close(self) -> None:
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
                f"语音 {st.voice_count}（需澄清 {st.need_confirm_count}）\n"
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
    args = ap.parse_args(argv)

    root = tk.Tk()
    con = Console(root, args.doctor)
    if args.selftest:
        root.after(600, root.destroy)
        root.mainloop()
        mc.log("[selftest] 界面构建与刷新正常 ✓")
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
