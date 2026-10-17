#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 NV21 样例（眼镜通道离线验证用，N52）。

**为什么需要**：Rokid 眼镜 SDK 推给我们的画面是 **NV21** 原始 YUV（官方示例
`IMessageListener.onNv21Data()`），而样例库里只有 JPG —— 于是"像素格式"这个坑
只能等真机到场才暴露。本脚本把现有样例帧转成 `*.nv21` + 同名 `*.json`（宽高），
`--glasses-adapter nv21` 这条通道**今天就能离线验证**，与真机走同一个 `nv21_to_bgr`。

产物是**生成物**（`data/*` 被 .gitignore 忽略），不入库、随用随生成。

用法：
    python scripts/make_nv21_samples.py                    # 默认 data/samples/acdc → data/samples/glasses_nv21
    python scripts/make_nv21_samples.py --max 12 --resize 640x480
    python client/live_client.py --source glasses --glasses-adapter nv21 --input data/samples/glasses_nv21
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "client"))

import cv2                                    # noqa: E402

import live_client as lc                      # noqa: E402  复用 bgr_to_nv21（与真机同一编码路径）
import mock_client as mc                      # noqa: E402

DEFAULT_SRC = ROOT / "data" / "samples" / "acdc"
DEFAULT_OUT = ROOT / "data" / "samples" / "glasses_nv21"


def _resolve(p: str, default: Path) -> Path:
    if not p:
        return default
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 NV21 样例（眼镜通道离线验证）")
    ap.add_argument("--src", default="", help=f"源图片目录（默认 {DEFAULT_SRC.relative_to(ROOT)}）")
    ap.add_argument("--out", default="", help=f"输出目录（默认 {DEFAULT_OUT.relative_to(ROOT)}）")
    ap.add_argument("--max", type=int, default=8, help="最多生成多少帧（默认 8）")
    ap.add_argument("--resize", default="", help="可选：缩放到 WxH（如 640x480），默认保持原尺寸")
    args = ap.parse_args(argv)

    src = _resolve(args.src, DEFAULT_SRC)
    out = _resolve(args.out, DEFAULT_OUT)
    if not src.is_dir():
        print(f"[错误] 源目录不存在：{src}")
        return 2

    want = None
    if args.resize:
        try:
            w, h = (int(v) for v in args.resize.lower().replace("*", "x").split("x", 1))
            want = (w, h)
        except Exception:  # noqa: BLE001
            print(f"[错误] --resize 无法解析：{args.resize}（示例 640x480）")
            return 2

    files = sorted((p for p in src.rglob("*") if p.suffix.lower() in mc.IMG_EXTS),
                   key=mc.natural_key)[:max(1, args.max)]
    if not files:
        print(f"[错误] {src} 下没有图片（支持 {sorted(mc.IMG_EXTS)}）")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[跳过] 读不出图：{p.name}")
            continue
        if want:
            img = cv2.resize(img, want, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        raw = lc.bgr_to_nv21(img)
        stem = f"frame_{i:03d}"
        (out / f"{stem}.nv21").write_bytes(raw)
        # 同名 sidecar：真机 SDK 会随帧给出宽高，离线样例也照这个约定走
        (out / f"{stem}.json").write_text(
            json.dumps({"width": w, "height": h, "src": p.name}, ensure_ascii=False),
            encoding="utf-8")
        total += 1
        print(f"  [生成] {stem}.nv21  {w}x{h}  {len(raw):,} 字节  ← {p.name}")

    size_mb = sum(f.stat().st_size for f in out.glob("*.nv21")) / 1024 / 1024
    print(f"[完成] {total} 帧 → {out.relative_to(ROOT)}（{size_mb:.1f} MB）"
          f"｜每帧长度 = w·h·3/2，与真机 NV21 一致")
    print(f"[提示] 原始帧很大（2304×1728 每帧 ≈ 6 MB）；只为验证通道可加 --max 3 或 --resize 1280x720；"
          f"该目录是生成物，可随时整个删掉")
    print(f"[下一步] python client/live_client.py --source glasses "
          f"--glasses-adapter nv21 --input {out.relative_to(ROOT)} --max-frames 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
