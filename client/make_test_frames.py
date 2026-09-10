#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EgoMed-Agent 联调测试样例生成（配套 client/mock_client.py）

作用：为 T6 联调准备两类输入——
  1. 各模态的正常帧：从 data/ 里挑一个病例的若干帧，供 mock_client --image-dir 连发
  2. 坏帧：模糊 / 过暗 / 低对比 / 全黑 / 非医学影像噪声，
     用来验证服务端的边界处理。契约（docs/API_CONTRACT.md 四·A1）要求这类画面
     返回 status=ok + overlay 空 + 一句提示语，**不能返 500，也不能走错误分支**。

归属：闫（客户端 / 测试工具）
契约：字段与规格一律以 docs/API_CONTRACT.md 为准，本脚本不自行发明约定。

用法
----
    # 1. 收集各模态样例帧（需要本机有 data/ 数据）
    python client/make_test_frames.py --collect

    # 2. 生成坏帧（任何机器都能跑；没数据时自动合成一张底图）
    python client/make_test_frames.py

    # 3. 指定底图生成坏帧
    python client/make_test_frames.py --base data/samples/acdc/frame_0001.jpg

产出
----
    data/samples/acdc/   frame_0001.jpg ...     （--collect，每个模态 12 帧）
    data/samples/amos/   ...
    data/samples/camus/  ...
    data/samples/cxr/    ...
    data/samples/polyp/  ...
    data/samples/bad/    blur.jpg dark.jpg lowcontrast.jpg black.jpg noise.jpg

怎么验证
--------
    # 压缩规格是否达标（不联网）
    python client/mock_client.py --image-dir data/samples/acdc --frames 5 --dry-run
    python client/mock_client.py --image-dir data/samples/bad  --frames 5 --dry-run

    # 服务端起来后，重点看坏帧：期望每帧都是 ok + overlay 空 + 提示语
    python client/mock_client.py --image-dir data/samples/bad --frames 5

依赖：pillow（requirements.txt 已包含）
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 pillow，请先执行：pip install pillow")


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SAMPLES = DATA / "samples"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 样例目录名 -> data/ 下的模态目录名（目录名与 mock_client 用法示例保持一致）
MODALS = {
    "acdc": "ACDC",
    "amos": "Amos",
    "camus": "CAMUS",
    "cxr": "Montgomery-County-CXR-Set",
    "polyp": "PolypGen2021_MultiCenterData_v3",
}

PER_MODAL = 12          # 每个模态取多少帧
BAD_MAX_SIDE = 640      # 坏帧统一缩到长边 640，够用且体积小


def log(msg: str = "") -> None:
    print(msg, flush=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def frames_in(case_dir: Path) -> list:
    """病例目录下的所有图片，按文件名排序"""
    return sorted((f for f in case_dir.rglob("*") if f.suffix.lower() in IMG_EXTS),
                  key=lambda p: p.name)


# ---------------------------------------------------------------- 收集正常帧
def collect(per_modal: int) -> bool:
    if not DATA.is_dir():
        log(f"[collect] 找不到数据目录 {rel(DATA)}，跳过（--collect 需要在有数据的机器上跑）")
        return False

    done = 0
    for key, modal in MODALS.items():
        case_root = DATA / modal / "img"
        if not case_root.is_dir():
            log(f"[collect] {key}: 跳过（{rel(case_root)} 不存在）")
            continue

        picked = None
        for case in sorted((p for p in case_root.iterdir() if p.is_dir()), key=lambda p: p.name):
            frames = frames_in(case)
            if frames:
                picked = (case, frames)
                break
        if not picked:
            log(f"[collect] {key}: 跳过（目录里没找到图片）")
            continue

        case, frames = picked
        out = SAMPLES / key
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        chosen = frames[:per_modal]
        for i, src in enumerate(chosen, 1):
            shutil.copy2(src, out / f"frame_{i:04d}{src.suffix.lower()}")
        log(f"[collect] {key}: 病例 {case.name}，取 {len(chosen)}/{len(frames)} 帧 -> {rel(out)}")
        done += 1

    if not done:
        log("[collect] 没有收集到任何样例")
    return bool(done)


# ---------------------------------------------------------------- 坏帧
def synth_base(size=(BAD_MAX_SIDE, BAD_MAX_SIDE)) -> Image.Image:
    """没有真图时合成一张'像医学影像'的底图：灰底 + 亮椭圆 + 噪点"""
    w, h = size
    img = Image.new("L", size, 18)
    draw = ImageDraw.Draw(img)
    draw.ellipse([w * 0.22, h * 0.16, w * 0.78, h * 0.84], fill=95)
    draw.ellipse([w * 0.33, h * 0.30, w * 0.67, h * 0.70], fill=150)
    draw.ellipse([w * 0.40, h * 0.40, w * 0.60, h * 0.60], fill=200)
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = max(0, min(255, px[x, y] + random.randint(-14, 14)))
    return img.convert("RGB")


def pick_base(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            sys.exit(f"找不到底图：{path}")
        return path
    if SAMPLES.is_dir():
        cands = sorted(p for p in SAMPLES.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if cands:
            return cands[0]
    return None


def make_bad(base: Path | None, out_dir: Path) -> None:
    if base is not None:
        with Image.open(base) as im:
            src = im.convert("RGB")
        log(f"[bad] 底图：{rel(base)}")
    else:
        src = synth_base()
        log("[bad] 没有可用底图，使用合成图")

    w, h = src.size
    scale = BAD_MAX_SIDE / float(max(w, h))
    if scale < 1.0:
        src = src.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 模糊：镜头没对准 / 运动模糊
    src.filter(ImageFilter.GaussianBlur(12)).save(out_dir / "blur.jpg", quality=80)
    # 过暗：环境光不足 / 屏幕太暗
    ImageEnhance.Brightness(src).enhance(0.10).save(out_dir / "dark.jpg", quality=80)
    # 低对比：屏幕反光、白平衡失准
    ImageEnhance.Contrast(src).enhance(0.18).save(out_dir / "lowcontrast.jpg", quality=80)
    # 全黑：遮挡 / 没开屏幕
    Image.new("RGB", src.size, (0, 0, 0)).save(out_dir / "black.jpg", quality=80)
    # 非医学影像：随机噪声（模拟对着墙、对着地面）
    noise = Image.new("RGB", src.size)
    noise.putdata([(random.randint(0, 255),) * 3 for _ in range(src.size[0] * src.size[1])])
    noise.save(out_dir / "noise.jpg", quality=80)

    names = sorted(p.name for p in out_dir.glob("*.jpg"))
    log(f"[bad] 生成 {len(names)} 张坏帧 -> {rel(out_dir)}")
    log(f"      {' '.join(names)}")


# ---------------------------------------------------------------- 入口
def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="生成 EgoMed-Agent 联调测试样例（规格以 docs/API_CONTRACT.md 为准）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--collect", action="store_true",
                        help="从 data/ 各模态收集正常帧到 data/samples/<模态>/")
    parser.add_argument("--per-modal", type=int, default=PER_MODAL,
                        help=f"每个模态收集多少帧，默认 {PER_MODAL}")
    parser.add_argument("--base", default=None,
                        help="生成坏帧用的底图，默认自动找 data/samples 下的第一张")
    parser.add_argument("--out", default=str(SAMPLES / "bad"), help="坏帧输出目录")
    args = parser.parse_args()

    if args.collect:
        collect(args.per_modal)

    make_bad(pick_base(args.base), Path(args.out))
    log("")
    log("下一步：python client/mock_client.py --image-dir data/samples/bad --frames 5 --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
