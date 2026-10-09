#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""引导式机位 / 佩戴校准（N36）—— 开工前先确认"对得准、看得清"

为什么要有它
------------
实测过的坑（见 `docs/失败案例分析.md`、`docs/工作日志.md`）：
  · **正对明亮 left lung 置信度 0.956，贴身斜拍 0.000** → 机位不对，结果直接没有
  · 画面偏暗 / 偏糊 / 没对准屏幕时，服务端按契约**返空 overlay + 提示语**
    （这是设计，不是故障）——但现场的人容易误以为"系统坏了"

本工具把这件事变成**一次有结论的校准**：
  ① （可选）先设分割目标（没目标时服务端不推理，也就无从判断）
  ② 从当前采集源抽 N 帧，逐帧发 A1，统计服务端的"结果 / 坏帧 / 提示"
  ③ 同时在**客户端本地**量画面质量（亮度 / 对比 / 清晰度），给出可操作的建议
  ④ 打印 PASS / 需调整 + 具体原因（偏暗？偏糊？没对准？距离不对？）
  ⑤ 落盘报告 `runs/calibrate/<时间戳>/report.md`

用法
----
    # 摄像头（现场最常用）
    python client/calibrate.py --source cam

    # 手机当镜头（IP Webcam）
    python client/calibrate.py --source url --input http://192.168.1.23:8080/video

    # 视频/帧目录（自动化自测）
    python client/calibrate.py --source video --input data/samples/acdc

    # 指定目标用的语音（默认自动挑一段能定目标的样例语音）
    python client/calibrate.py --source cam --audio data/samples/audio/cmd_03.wav

判定口径
--------
    PASS           ：服务端有结果的帧 ≥ 62.5%（默认 5/8）且无请求错误
    需调整         ：给出原因（偏暗 / 偏糊 / 没对准 / 未设目标 / 网络错误）
退出码：0=PASS，1=需调整，2=环境/参数问题

归属：闫（客户端 / 现场工具）｜ 关联：`docs/总计划.md` 阶段五 N36、控制台「机位校准」按钮
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import mock_client as mc  # noqa: E402
import live_client as lc  # noqa: E402

PROJECT_ROOT = mc.PROJECT_ROOT
OUT_ROOT = PROJECT_ROOT / "runs" / "calibrate"

# 本地画面质量阈值：与服务端坏帧判定（N16）同口径（先归一化长边到 640 再算）
NORM_SIDE = 640
TH_BRIGHT = 15.0        # 平均亮度下限
TH_CONTRAST = 20.0      # 对比度（std）下限
TH_SHARP_LO = 120.0     # 清晰度（lapvar）下限（低于 → 偏糊）
TH_SHARP_HI = 12000.0   # 上限（高于 → 多是噪点）
PASS_RATIO = 0.625      # 有结果帧占比 ≥ 该值算通过


def local_quality(frame) -> dict:      # noqa: ANN001
    """本地量画面质量（缩放细节后归一化到 640，与服务端口径一致）。"""
    h, w = frame.shape[:2]
    scale = NORM_SIDE / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "bright": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def local_advice(q: dict) -> str:
    """把本地质量指标翻译成人话建议（空串 = 没意见）。"""
    tips = []
    if q["bright"] < TH_BRIGHT:
        tips.append(f"画面偏暗（亮度 {q['bright']:.0f} < {TH_BRIGHT:.0f}）→ 补光 / 调屏幕亮度")
    if q["contrast"] < TH_CONTRAST:
        tips.append(f"对比度偏低（{q['contrast']:.0f} < {TH_CONTRAST:.0f}）→ 避免强反光、正对屏幕")
    if q["sharp"] < TH_SHARP_LO:
        tips.append(f"画面偏糊（清晰度 {q['sharp']:.0f} < {TH_SHARP_LO:.0f}）→ 对焦 / 靠近些 / 别抖")
    elif q["sharp"] > TH_SHARP_HI:
        tips.append(f"噪点偏多（清晰度 {q['sharp']:.0f} > {TH_SHARP_HI:.0f}）→ 光线太暗导致的噪点")
    return "；".join(tips)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="引导式机位/佩戴校准（N36）")
    ap.add_argument("--source", choices=["video", "cam", "url", "glasses"], default="cam")
    ap.add_argument("--input", default="data/samples/acdc", help="video/url/glasses 源的输入")
    ap.add_argument("--index", type=int, default=0, help="摄像头序号（cam 源）")
    ap.add_argument("--source-fps", type=float, default=lc.DEFAULT_SOURCE_FPS,
                    help="源播放节奏（video/url/glasses 的 replay 用）")
    ap.add_argument("--frames", type=int, default=8, help="抽几帧做校准（默认 8）")
    ap.add_argument("--timeout", type=float, default=10.0, help="单帧请求超时（秒）")
    ap.add_argument("--url", default=mc.DEFAULT_URL, help=f"服务端地址（默认 {mc.DEFAULT_URL}）")
    ap.add_argument("--audio", default="", help="用来设目标的 wav（默认自动挑一段能定目标的样例）")
    ap.add_argument("--no-set-target", action="store_true", help="不设目标（仅测画面质量/连通性）")
    args = ap.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    session = mc.make_session_id("calib")

    print("=" * 76)
    print("引导式机位 / 佩戴校准（N36）")
    print("=" * 76)

    # ---------- ① 连通性 + 设目标 ----------
    print("[1/4] 检查服务端 …")
    try:
        import requests
        h = requests.get(f"{args.url.rstrip('/')}/healthz", timeout=2.0).json()
        print(f"      [OK] {args.url}　real_engine={h.get('real_engine')}")
    except Exception as exc:  # noqa: BLE001
        print(f"      [错误] 连不上服务端：{exc}")
        print("      → 先起服务端：python server/api.py --real（或双击「启动模拟眼镜.bat」）")
        return 2

    target = ""
    if args.no_set_target:
        print("[2/4] 跳过设目标（--no-set-target）")
    else:
        print("[2/4] 设置分割目标（没目标时服务端不推理，校准无意义）…")
        if args.audio:
            wav = Path(args.audio)
            if not wav.is_absolute():
                wav = PROJECT_ROOT / wav
            payload, el, err = mc.post_audio(args.url, session, wav.read_bytes(),
                                             wav.name, 20.0)
            target = (payload or {}).get("target") or ""
        else:
            target, used, _, err = mc.set_target(args.url, session, None, 20.0)
        if target:
            print(f"      [OK] target = {target!r}")
        else:
            print("      [警告] 没设上目标 —— 下面只能判断「画面是否可用」，"
                  "无法判断「能不能出结果」")

    # ---------- ③ 采帧（本地质量）----------
    print(f"[3/4] 请把镜头对准屏幕上的影像（保持稳定），开始抽 {args.frames} 帧 …")
    # build_source 用到的字段比本工具暴露的多，补齐默认值（避免 AttributeError）
    for k, v in (("no_loop", False), ("source_fps", lc.DEFAULT_SOURCE_FPS),
                 ("width", 1280), ("height", 720),
                 ("glasses_adapter", "replay"), ("glasses_queue", 2)):
        if not hasattr(args, k):
            setattr(args, k, v)
    src = lc.build_source(args)
    src.open()
    # 丢掉前几帧：摄像头自动曝光/对焦收敛需要一点时间
    for _ in range(3):
        src.read()
    frames = []
    for _ in range(args.frames):
        ok, frame = src.read()
        if not ok:
            break
        if frame is not None:
            frames.append(frame)
        time.sleep(0.15)
    src.release()
    if not frames:
        print("      [错误] 没取到画面（源没开起来？换个 --source 试试）")
        return 2
    qualities = [local_quality(f) for f in frames]
    avg_q = {k: sum(q[k] for q in qualities) / len(qualities) for k in qualities[0]}
    print(f"      本地画面质量（{len(frames)} 帧均值）：亮度 {avg_q['bright']:.0f}　"
          f"对比度 {avg_q['contrast']:.0f}　清晰度 {avg_q['sharp']:.0f}")
    tip_local = local_advice(avg_q)
    if tip_local:
        print(f"      [提示] {tip_local}")

    # ---------- ④ 逐帧发 A1，看服务端怎么说 ----------
    print(f"[4/4] 发送 {len(frames)} 帧到服务端 …")
    rows = []
    for i, f in enumerate(frames, 1):
        jpg = None
        try:
            import io
            from PIL import Image
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            im = Image.fromarray(rgb)
            w, h = im.size
            sc = mc.MAX_SIDE / float(max(w, h))
            if sc < 1.0:
                im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))))
            im.save(buf, format="JPEG", quality=mc.JPEG_QUALITY, optimize=True)
            jpg = buf.getvalue()
        except Exception:  # noqa: BLE001
            jpg = None
        if jpg is None:
            continue
        payload, elapsed, err = mc.post_frame(args.url, session, jpg,
                                              f"calib_{i:03d}.jpg", args.timeout)
        overlay = bool((payload or {}).get("overlay"))
        msg = str((payload or {}).get("message") or err or "")
        rows.append({"i": i, "ok": err is None and overlay, "elapsed": elapsed,
                     "err": err, "msg": msg, "overlay": overlay})
        mark = "[有结果]" if overlay else "[无结果]"
        print(f"      第 {i:>2} 帧 {mark} {elapsed:>6.0f}ms  {msg[:52]}")

    good = sum(1 for r in rows if r["overlay"])
    errs = [r for r in rows if r["err"]]
    ratio = good / max(1, len(rows))

    # ---------- 判定 ----------
    reasons: list[str] = []
    if errs:
        reasons.append(f"有 {len(errs)} 帧请求失败（网络/服务端）：{errs[0]['err']}")
    if ratio < PASS_RATIO:
        msgs = " ".join(r["msg"] for r in rows if not r["overlay"])
        if "请先告诉我要分割哪个器官" in msgs:
            reasons.append("服务端提示未设目标 → 先把目标设上（把话说清楚）")
        if "对准屏幕" in msgs or "没检测到目标" in msgs:
            reasons.append("多数帧没检测到目标 → 机位/距离/角度不对：正对影像、占画面 1/3 以上、避开反光")
        if not reasons:
            reasons.append("有结果的帧太少 → 换个更亮更清晰的机位，或换一张目标更明显的样例")
    if tip_local:
        reasons.append(f"本地画面质量：{tip_local}")

    ok = (ratio >= PASS_RATIO) and not errs
    print("-" * 76)
    print(f"结论：{'通过 ✅' if ok else '需调整 ⚠'}"
          f"（有结果 {good}/{len(rows)} = {ratio*100:.0f}%，阈值 {PASS_RATIO*100:.0f}%）")
    for r in reasons:
        print(f"  → {r}")
    print("=" * 76)

    (out_dir / "report.md").write_text("\n".join([
        f"# 机位/佩戴校准报告 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        f"- 源 `{args.source}` / `{args.input}`　服务端 `{args.url}`　目标 `{target or '（未设）'}`",
        f"- 本地画面质量：亮度 {avg_q['bright']:.0f} / 对比度 {avg_q['contrast']:.0f} / "
        f"清晰度 {avg_q['sharp']:.0f}",
        f"- **结论：{'通过 ✅' if ok else '需调整 ⚠'}**（有结果 {good}/{len(rows)}）", "",
        "## 逐帧", "", "| 帧 | 有结果 | 耗时 | 服务端提示 / 错误 |", "|---:|:---:|---:|---|",
        *[f"| {r['i']} | {'✓' if r['overlay'] else '—'} | {r['elapsed']:.0f}ms | {r['msg'][:80]} |"
          for r in rows], "",
        "## 建议", "", *([f"- {r}" for r in reasons] or ["- 无（机位正常，可以开工）"]), "",
    ]), encoding="utf-8")
    print(f"报告：{out_dir.relative_to(PROJECT_ROOT)}/report.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
