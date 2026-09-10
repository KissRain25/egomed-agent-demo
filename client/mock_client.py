#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EgoMed-Agent 客户端联调脚本（任务 T5）

作用：模拟眼镜端，按《docs/API_CONTRACT.md》v0.2 的约定把一帧画面（或一段录音）
      发给服务器，接收分割结果，并把返回的结果图存到本地。

归属：闫（客户端 / 测试工具）
契约：字段定义一律以 docs/API_CONTRACT.md 为准，本脚本不自行发明字段。

用法
----
    # 0. 不联网自测：只检查图片压缩是否满足契约规格（服务器还没好也能跑）
    python client/mock_client.py --image test.jpg --dry-run

    # 1. 发单帧（最常用）
    python client/mock_client.py --image test.jpg

    # 2. 连发多帧（模拟 5 fps 推流，测服务器的跟踪与丢帧）
    python client/mock_client.py --image-dir data/samples/acdc --fps 5 --frames 20

    # 3. 发语音（走 A2 接口）
    python client/mock_client.py --audio test.wav

    # 4. 换服务器地址 / 医生编号
    python client/mock_client.py --image test.jpg --url http://192.168.1.10:8000 --doctor 007

怎么验证（交付规范：文件在哪 / 能做什么 / 怎么验证）
----------------------------------------------------
    1) 先跑 --dry-run，确认每行都是 OK（说明压缩达标）
    2) 让李启动服务端，再跑 --image test.jpg
    3) 期望：终端打印一行 ok + modality/target/message；
       runs/stream/ 下出现 frame_0001_overlay.jpg（结果图）和 frame_0001_resp.json（响应留档）

依赖：requests、pillow（requirements.txt 已包含）
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import string
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 requests，请先执行：pip install requests")

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 pillow，请先执行：pip install pillow")


# ---------------------------------------------------------------- 契约常量（改这里就等于改契约，必须同步文档）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_OUT = PROJECT_ROOT / "runs" / "stream"

MAX_SIDE = 1280          # 长边上限（px）
JPEG_QUALITY = 80        # JPEG 质量
MAX_BYTES = 300 * 1024   # 单帧上限 300 KB
AUDIO_MAX_BYTES = 200 * 1024   # 单段录音上限 200 KB
TIMEOUT = 3.0            # 单请求超时（秒）
MAX_CONSECUTIVE_FAIL = 3  # 连续失败次数，达到则暂停探测
PROBE_INTERVAL = 5.0     # 恢复探测间隔（秒）

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def log(msg: str = "") -> None:
    print(msg, flush=True)


def make_session_id(doctor: str) -> str:
    """生成 session_id，格式 doc-<医生编号>-<yyyymmdd>-<4位随机>（契约 2 / 5.1）"""
    date = datetime.now().strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"doc-{doctor}-{date}-{suffix}"


def natural_key(path: Path):
    """让 frame_2.jpg 排在 frame_10.jpg 前面"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


# ---------------------------------------------------------------- 图片压缩
def compress_image(path: Path, max_side: int = MAX_SIDE, quality: int = JPEG_QUALITY,
                   max_bytes: int = MAX_BYTES) -> bytes:
    """按契约压缩：长边 ≤ max_side、质量 quality、单帧 ≤ max_bytes。

    压不到目标就逐级降质量（80→70→60→50→40），保证一定能发出去。
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = max_side / float(max(w, h))
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), _RESAMPLE)
            data = b""
            for q in (quality, 70, 60, 50, 40):
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=q, optimize=True)
                data = buf.getvalue()
                if len(data) <= max_bytes:
                    break
            return data
    except Exception as exc:  # 损坏文件 / 不支持的格式
        raise RuntimeError(f"无法读取图片 {path}：{exc}") from exc


# ---------------------------------------------------------------- HTTP
def post_multipart(url: str, path: str, session_id: str, field: str,
                   filename: str, content: bytes, content_type: str,
                   timeout: float = TIMEOUT):
    """发一个 multipart 请求。

    返回 (payload, elapsed_ms, error)：
      - error 为 None 表示 HTTP 层成功（payload 里可能仍是 status=error 的业务错误）
      - error 非 None 表示网络层失败（超时 / 连不上 / 返回不是 JSON）

    说明：multipart 表单值统一按字符串发送，服务器侧声明成 int 即可自动转换。
    """
    api = f"{url.rstrip('/')}{path}"
    data = {"session_id": session_id, "timestamp": str(int(time.time() * 1000))}
    files = {field: (filename, content, content_type)}
    t0 = time.perf_counter()
    try:
        resp = requests.post(api, data=data, files=files, timeout=timeout)
    except requests.exceptions.Timeout:
        return None, (time.perf_counter() - t0) * 1000, f"超时（>{timeout:g}s）"
    except requests.exceptions.ConnectionError:
        return None, (time.perf_counter() - t0) * 1000, f"连不上 {api}（服务端起了吗？端口对吗？）"
    except requests.exceptions.RequestException as exc:
        return None, (time.perf_counter() - t0) * 1000, f"请求异常：{exc}"
    elapsed = (time.perf_counter() - t0) * 1000

    try:
        payload = resp.json()
    except ValueError:
        return None, elapsed, f"HTTP {resp.status_code}，返回不是 JSON：{resp.text[:200]!r}"

    if resp.status_code != 200:
        msg = payload.get("message") if isinstance(payload, dict) else None
        return payload, elapsed, f"HTTP {resp.status_code} {msg or ''}".strip()
    return payload, elapsed, None


def post_frame(url: str, session_id: str, jpg: bytes, filename: str, timeout: float = TIMEOUT):
    return post_multipart(url, "/api/v1/frame", session_id, "image",
                          filename, jpg, "image/jpeg", timeout)


def post_audio(url: str, session_id: str, wav: bytes, filename: str, timeout: float = TIMEOUT):
    return post_multipart(url, "/api/v1/audio", session_id, "audio",
                          filename, wav, "audio/wav", timeout)


# ---------------------------------------------------------------- 结果保存
def save_result(out_dir: Path, index: int, payload: dict) -> Path | None:
    """把返回的 overlay 图落盘，同时留一份响应 JSON（剔除 base64 大字段，便于排查）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_b64 = payload.get("overlay") or ""
    img_path = None
    if overlay_b64:
        try:
            raw = base64.b64decode(overlay_b64)
        except Exception as exc:
            log(f"      [!] overlay base64 解码失败：{exc}")
            raw = b""
        if raw:
            img_path = out_dir / f"frame_{index:04d}_overlay.jpg"
            img_path.write_bytes(raw)

    slim = dict(payload)
    if slim.get("overlay"):
        slim["overlay"] = f"<base64 jpg, {len(slim['overlay'])} chars>"
    (out_dir / f"frame_{index:04d}_resp.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    return img_path


# ---------------------------------------------------------------- 帧来源
def collect_frames(args) -> list:
    if args.image:
        path = Path(args.image)
        if not path.is_file():
            sys.exit(f"找不到图片：{path}")
        return [path]

    directory = Path(args.image_dir)
    if not directory.is_dir():
        sys.exit(f"找不到目录：{directory}")
    frames = sorted((f for f in directory.rglob("*") if f.suffix.lower() in IMG_EXTS), key=natural_key)
    if not frames:
        sys.exit(f"目录里没有图片：{directory}（支持 {'/'.join(sorted(IMG_EXTS))}）")

    if args.loop and args.frames:
        base = list(frames)
        i = 0
        while len(frames) < args.frames:
            frames.append(base[i % len(base)])
            i += 1
    if args.frames:
        frames = frames[:args.frames]
    return frames


# ---------------------------------------------------------------- 模式：dry-run
def run_dry(frames, args) -> None:
    log(f"[dry-run] 不发送，只检查压缩规格（目标：长边≤{args.max_side}、质量{args.quality}、"
        f"≤{args.max_bytes // 1024} KB）")
    bad = 0
    for i, path in enumerate(frames, 1):
        try:
            data = compress_image(path, args.max_side, args.quality, args.max_bytes)
        except RuntimeError as exc:
            log(f"  !!  [{i}] {exc}")
            bad += 1
            continue
        with Image.open(path) as im:
            ow, oh = im.size
        with Image.open(io.BytesIO(data)) as im2:
            nw, nh = im2.size
        ok = len(data) <= args.max_bytes
        bad += 0 if ok else 1
        log(f"  {'OK' if ok else '!!'}  [{i}] {path.name}: {ow}x{oh} -> {nw}x{nh}, "
            f"{len(data) / 1024:.1f} KB")
    log(f"[dry-run] 完成：{len(frames) - bad} 达标 / {bad} 不达标")
    if bad:
        log("[dry-run] 有帧超限，请调小 --quality 或 --max-side")
    else:
        log("[dry-run] 全部达标，可以联调了：python client/mock_client.py --image <图>")


# ---------------------------------------------------------------- 模式：发帧
def run_stream(frames, args, session_id: str) -> None:
    api = f"{args.url.rstrip('/')}/api/v1/frame"
    interval = 1.0 / args.fps if args.fps > 0 else 0.0

    log(f"服务器   : {api}")
    log(f"session  : {session_id}")
    log(f"帧数     : {len(frames)}    帧率: {args.fps:g} fps    超时: {args.timeout:g}s")
    log(f"输出目录 : {args.out}")
    log("-" * 78)

    ok = empty = biz_err = net_err = 0
    elapsed_list = []
    saved = []
    fail_streak = 0

    for i, path in enumerate(frames, 1):
        # 节流：同步 HTTP 天然满足"上一帧未返回不发新帧"（契约 5.4）
        if interval and i > 1:
            time.sleep(interval)

        try:
            jpg = compress_image(path, args.max_side, args.quality, args.max_bytes)
        except RuntimeError as exc:
            biz_err += 1
            log(f"[{i:>3}] 跳过      {exc}")
            continue

        payload, ms, err = post_frame(args.url, session_id, jpg, path.name, args.timeout)

        if err:
            net_err += 1
            fail_streak += 1
            log(f"[{i:>3}] 连接失败  {ms:>6.0f}ms  {err}")
            if fail_streak >= MAX_CONSECUTIVE_FAIL:
                log(f"      连续失败 {fail_streak} 次，暂停 {PROBE_INTERVAL:g}s 后探测…（契约 5.5）")
                time.sleep(PROBE_INTERVAL)
                payload, ms, err = post_frame(args.url, session_id, jpg, path.name, args.timeout)
                if err:
                    log("      探测仍失败，停止发送。请确认服务端已启动、端口一致。")
                    break
                fail_streak = 0
                log("      探测成功，已恢复。")
            else:
                continue

        fail_streak = 0
        elapsed_list.append(ms)

        if payload.get("status") == "error":
            biz_err += 1
            log(f"[{i:>3}] 业务错误  {ms:>6.0f}ms  code={payload.get('code')} "
                f"{payload.get('message', '')}")
            save_result(args.out, i, payload)
            continue

        overlay = payload.get("overlay") or ""
        if overlay:
            ok += 1
            img = save_result(args.out, i, payload)
            if img:
                saved.append(img)
            note = f"overlay={len(overlay) / 1024:.0f}KB -> {img.name if img else '未保存'}"
        else:
            empty += 1
            note = "overlay 为空（客户端应保留上一帧，不要黑屏）"

        log(f"[{i:>3}] ok        {ms:>6.0f}ms  modality={payload.get('modality') or '-'}  "
            f"target={payload.get('target') or '-'}  {note}  \"{payload.get('message', '')}\"")
        if payload.get("need_confirm"):
            log(f"      需要澄清：候选={payload.get('candidates')}（客户端弹提示，等医生再说一句）")

    log("-" * 78)
    log(f"汇总：有效结果 {ok}    空结果 {empty}    业务错误 {biz_err}    连接失败 {net_err}")
    if elapsed_list:
        log(f"耗时：平均 {sum(elapsed_list) / len(elapsed_list):.0f} ms    最大 {max(elapsed_list):.0f} ms")
    if saved:
        log(f"结果图 {len(saved)} 张 -> {args.out}")
        log(f"最新一张：{saved[-1]}")
    elif not net_err:
        log("没有拿到结果图（overlay 全为空，或全部是业务错误）")
    log("联调结论请按 docs/接口.md 第六节回填（接口不一致时以 docs/API_CONTRACT.md 为准）")


# ---------------------------------------------------------------- 模式：发语音
def run_audio(args, session_id: str) -> None:
    path = Path(args.audio)
    if not path.is_file():
        sys.exit(f"找不到音频：{path}")
    raw = path.read_bytes()
    api = f"{args.url.rstrip('/')}/api/v1/audio"
    log(f"服务器   : {api}")
    log(f"session  : {session_id}")
    log(f"音频     : {path.name}  {len(raw) / 1024:.1f} KB")
    if len(raw) > AUDIO_MAX_BYTES:
        log(f"[!] 超过契约上限 {AUDIO_MAX_BYTES // 1024} KB（16kHz 单声道 ≤5 秒），服务器可能拒收")
    log("-" * 78)

    payload, ms, err = post_audio(args.url, session_id, raw, path.name, args.timeout)
    if err:
        log(f"失败  {ms:.0f}ms  {err}")
        return
    if payload.get("status") == "error":
        log(f"业务错误  code={payload.get('code')}  {payload.get('message', '')}")
        return

    log(f"ok  {ms:.0f}ms")
    log(f"  识别文本 text          : {payload.get('text', '')}")
    log(f"  分割目标 target        : {payload.get('target') or '-'}")
    log(f"  模态建议 modality_hint : {payload.get('modality_hint') or '-'}")
    log(f"  提示 message           : {payload.get('message', '')}")
    if payload.get("need_confirm"):
        log(f"  需要澄清 candidates    : {payload.get('candidates')}")


# ---------------------------------------------------------------- 入口
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="EgoMed-Agent 客户端联调脚本（字段以 docs/API_CONTRACT.md v0.2 为准）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="单张图片路径（发一帧）")
    src.add_argument("--image-dir", help="图片目录（按文件名顺序连发多帧）")
    src.add_argument("--audio", help="wav 文件路径（走 A2 语音接口）")

    parser.add_argument("--url", default=DEFAULT_URL, help=f"服务器地址，默认 {DEFAULT_URL}")
    parser.add_argument("--doctor", default="001", help="医生编号，用于生成 session_id，默认 001")
    parser.add_argument("--session-id", default=None, help="直接指定 session_id（默认自动生成）")
    parser.add_argument("--fps", type=float, default=5.0, help="发送帧率，默认 5（契约约定）")
    parser.add_argument("--frames", type=int, default=0, help="最多发多少帧，0=不限")
    parser.add_argument("--loop", action="store_true", help="目录图片循环发送，配合 --frames 用")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"结果输出目录，默认 {DEFAULT_OUT}")
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="单请求超时秒数，默认 3")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE, help="压缩后长边上限，默认 1280")
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY, help="JPEG 质量，默认 80")
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES, help="单帧字节上限，默认 307200")
    parser.add_argument("--dry-run", action="store_true", help="只压缩不发送（服务器没好也能自测）")

    args = parser.parse_args()
    args.out = Path(args.out)
    session_id = args.session_id or make_session_id(args.doctor)

    if args.audio:
        if args.dry_run:
            sys.exit("--audio 不支持 --dry-run（音频不做压缩）")
        run_audio(args, session_id)
        return

    frames = collect_frames(args)
    if args.dry_run:
        run_dry(frames, args)
    else:
        run_stream(frames, args, session_id)


if __name__ == "__main__":
    main()
