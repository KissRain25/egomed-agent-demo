#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成演示视频（N10）：语音设目标 → 逐帧实时分割 → 坏帧防抖，输出可播放 mp4

为什么用合成而不是录屏：
    录屏依赖人手与屏幕状态，不可复现；本工具走**真实链路**（A2 设目标 + A1 逐帧请求服务端），
    把服务端返回的 overlay 逐帧拼成视频，结果可重复、可自动生成、便于嵌入文档与汇报。

视频结构（约 10~15 秒）：
    ① 开头 2 秒：字幕「语音指令：分割心肌」+ 目标切换结果
    ② 主体：每帧显示分割结果 + 状态栏（帧号 / 目标 / 单帧耗时 / 模态）
    ③ 结尾：插入 1~2 个**坏帧**（全黑）→ 服务端返空 overlay → 显示「沿用上一帧 + 请对准屏幕」
       （即客户端防抖行为，对应契约 A1 边界情况 1）

用法（服务端需 --real 启动并预热完成）：
    python client/make_demo_video.py                      # 默认 ACDC 心肌 12 帧 + 2 坏帧
    python client/make_demo_video.py --target "left lung" --wav data/samples/audio/cmd_07.wav \
        --frames-dir data/samples/cxr --frames 12
归属：闫（演示工具）｜ 关联：docs/总计划.md N10
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "client"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "demo"))

import cv2
import mock_client as mc
import live_client as lc          # 复用压缩与中文 HUD 绘制

OUT_DIR = PROJECT_ROOT / "runs" / "demo"


def log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("gbk", "replace").decode("gbk"), flush=True)


def banner(text: str, w: int, h: int, sub: str = "") -> np.ndarray:
    """开头/结尾字幕帧"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (32, 32, 32)
    hud = lc.Hud()
    lines = [text] + ([sub] if sub else [])
    img = hud.draw(img, lines)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="合成演示视频（N10）")
    ap.add_argument("--url", default=mc.DEFAULT_URL)
    ap.add_argument("--target", default="myocardium")
    ap.add_argument("--wav", default=str(PROJECT_ROOT / "data" / "samples" / "audio" / "cmd_03.wav"))
    ap.add_argument("--frames-dir", default=str(PROJECT_ROOT / "data" / "samples" / "acdc"))
    ap.add_argument("--frames", type=int, default=12, help="主体帧数")
    ap.add_argument("--bad-frames", type=int, default=2, help="结尾插入的坏帧数（演示防抖）")
    ap.add_argument("--fps", type=float, default=4.0, help="视频帧率")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--out", default="", help="输出 mp4（默认 runs/demo/sim_glasses_<时间戳>.mp4）")
    args = ap.parse_args()

    import requests
    try:
        hz = requests.get(f"{args.url}/healthz", timeout=3).json()
    except Exception as exc:
        log(f"[!!] 服务端不可用（{exc}）：请先 `python server/api.py --real` 并等预热")
        return 2
    if not hz.get("real_engine"):
        log("[!!] 服务端是假实现，请用 --real 启动")
        return 2

    frames_dir = Path(args.frames_dir)
    frames = sorted([f for f in frames_dir.glob("*") if f.suffix.lower() in mc.IMG_EXTS],
                    key=mc.natural_key)[:args.frames]
    if not frames:
        log(f"[!!] 无样例帧：{frames_dir}")
        return 2
    bad_dir = PROJECT_ROOT / "data" / "samples" / "bad"
    bad_frames = sorted(bad_dir.glob("black.jpg"))[:1] * max(0, args.bad_frames)

    sid = mc.make_session_id("demo")
    tmp = Path(tempfile.mkdtemp(prefix="egomed_demo_"))
    shots = []
    try:
        # ① A2 设目标
        wav = Path(args.wav)
        payload, ms, err = mc.post_audio(args.url, sid, wav.read_bytes(), wav.name, args.timeout)
        if err or not payload.get("target"):
            log(f"[!!] A2 设目标失败：{err or payload.get('message')}")
            return 2
        text, target = payload.get("text"), payload.get("target")
        log(f"A2 「{text}」→ {target}（{ms:.0f}ms）")

        # 取一帧原图确定画布尺寸
        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]
        scale = min(1.0, 1280 / max(h, w))
        if scale < 1.0:
            w, h = int(w * scale), int(h * scale)
        hud = lc.Hud()

        # ① 字幕
        for _ in range(2):
            shots.append(banner(f"语音指令：{text}", w, h, f"识别目标：{target}"))
        # ② 主体：逐帧请求真实服务端
        last_overlay = None
        for i, fp in enumerate(frames):
            jpg = mc.compress_image(fp)
            payload, ms, err = mc.post_frame(args.url, sid, jpg, f"{i:03d}.jpg", args.timeout)
            if err:
                log(f"  帧 {i}: 请求失败 {err}")
                continue
            overlay_b64 = payload.get("overlay") or ""
            mod, tgt = payload.get("modality") or "-", payload.get("target") or "-"
            if overlay_b64:
                img = lc.decode_overlay(overlay_b64)
                if img is not None:
                    last_overlay = img
                    shots.append(hud.draw(lc.fit_to(img, w, h).copy(),
                                          [f"第 {i + 1}/{len(frames)} 帧　{ms:.0f} ms",
                                           f"模态 {mod}",
                                           f"目标 {tgt}",
                                           "已分割（结果回显）"]))
                    log(f"  帧 {i}: overlay {ms:.0f}ms")
                    continue
            # 空 overlay：客户端防抖——沿用上一帧 + 提示
            disp = lc.fit_to(last_overlay, w, h).copy() if last_overlay is not None else first.copy()
            if last_overlay is None and disp.shape[:2] != (h, w):
                disp = lc.fit_to(disp, w, h)
            shots.append(hud.draw(disp, [f"第 {i + 1}/{len(frames)} 帧　{ms:.0f} ms",
                                         f"提示 {payload.get('message') or '未检测到目标'}",
                                         "沿用上一帧（画面不闪跳）"]))
            log(f"  帧 {i}: 空 overlay（防抖） {ms:.0f}ms")

        # ③ 结尾：坏帧防抖演示
        for j, bp in enumerate(bad_frames):
            jpg = mc.compress_image(bp)
            payload, ms, err = mc.post_frame(args.url, sid, jpg, f"bad_{j}.jpg", args.timeout)
            if err:
                continue
            disp = lc.fit_to(last_overlay, w, h).copy() if last_overlay is not None else first.copy()
            shots.append(hud.draw(disp, ["坏帧：镜头被挡/转向",
                                         f"服务端：{payload.get('message') or '未检测到目标'}",
                                         "客户端：保留上一帧，不黑屏",
                                         "网络恢复后自动继续"]))
            log(f"  坏帧 {j}: 服务端 overlay={'有' if payload.get('overlay') else '空'} → 客户端沿用上一帧")
        # 尾帧
        shots.append(banner("模拟眼镜 · 端到端实时分割", w, h, f"会话 {sid}"))

        # 落盘帧 → 编码 mp4（复用 demo 的 H.264 编码，浏览器可播）
        out_mp4 = Path(args.out) if args.out else OUT_DIR / f"sim_glasses_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        seq_dir = tmp / "shots"
        seq_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for k, img in enumerate(shots):
            p = seq_dir / f"{k:05d}.jpg"
            cv2.imwrite(str(p), img)
            files.append(p)
        import egomed_demo as demo
        demo._encode_video(files, out_mp4, args.fps, 1280)
        size_mb = out_mp4.stat().st_size / 1024 / 1024 if out_mp4.exists() else 0
        log("=" * 70)
        log(f"演示视频：{out_mp4.relative_to(PROJECT_ROOT)}（{len(files)} 帧 @ {args.fps:g}fps，{size_mb:.2f} MB）")
        log("=" * 70)
        return 0 if out_mp4.exists() and size_mb > 0 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
