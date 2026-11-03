#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「统一尺寸」的样例帧副本（给长稳/演示减变量）
==================================================

背景
----
`data/samples/acdc/` 里 15 帧混着两种尺寸：3 帧 480x640、12 帧 1728x2304。
这本身不是问题（服务端每帧独立处理），但 2026-11-03 长稳时它**叠加**了
「宿主删除守卫导致窗口残留 2 帧」，触发过掩膜尺寸不匹配的 bug
（已修，见 `scripts/demo/egomed_demo.py`）。留着这份尺寸不一的目录，
以后排查同类问题会多一个干扰变量。

本脚本生成一份**统一尺寸**的副本，原目录**完全不动**：

    python scripts/make_uniform_samples.py
    # 输出：data/samples/acdc_uniform/（默认统一到 480x640）

注意：`data/` 在 .gitignore 内，副本不入库，交付方需自行跑一次本脚本生成。

用法
----
    python scripts/make_uniform_samples.py --size 480 640
    python scripts/make_uniform_samples.py --src data/samples/acdc --dst data/samples/acdc_uniform
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="生成统一尺寸的样例帧副本")
    ap.add_argument("--src", default=str(REPO_ROOT / "data" / "samples" / "acdc"))
    ap.add_argument("--dst", default=str(REPO_ROOT / "data" / "samples" / "acdc_uniform"))
    ap.add_argument("--size", type=int, nargs=2, default=[480, 640],
                    metavar=("H", "W"), help="目标尺寸（默认 480 640）")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 质量（默认 92）")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print("[FAIL] 需要 opencv-python，请先激活 egomed 环境")
        return 2

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        print(f"[FAIL] 源目录不存在：{src}")
        return 2
    files = sorted(src.glob("*.jpg"))
    if not files:
        print(f"[FAIL] 源目录没有 jpg：{src}")
        return 2

    dst.mkdir(parents=True, exist_ok=True)
    h, w = args.size
    resized = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[WARN] 读不出，跳过：{f.name}")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            resized += 1
        cv2.imwrite(str(dst / f.name), img, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])

    sizes = {}
    for f in sorted(dst.glob("*.jpg")):
        shape = cv2.imread(str(f)).shape[:2]
        sizes[shape] = sizes.get(shape, 0) + 1
    print(f"[OK] {dst}：共 {len(list(dst.glob('*.jpg')))} 帧"
          f"（其中 {resized} 帧被缩放到 {h}x{w}）")
    print(f"     尺寸分布：{sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
