#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现并定位「掩膜形状与帧尺寸不一致」导致的 HTTP 500
=====================================================

现象（2026-11-03 长稳 `runs/soak/20261103_121224`）
--------------------------------------------------
11/3504 请求返回 500，服务端日志只有一行消息（没有 traceback，因为异常被
`post_frame` 的 `except Exception` 吞掉并转成 `_error(..., 500)`）：

    error:boolean index did not match indexed array along axis 0;
           size of axis is 960 but size of corresponding boolean axis is 480

numpy 这条消息的含义是：**被索引数组**（标签图）该轴长 960，
**布尔索引数组**（掩膜）该轴长 480 —— 两者对不上。

本脚本按服务端 `_handle_frame_real` 的真实方式复现：滚动窗口只留 1 帧，
逐帧把样例帧写进去再分割，这样能撞出「同一会话里帧尺寸不一致」的时刻，
并打印**完整 traceback**（服务端日志里没有的那部分）。

用法
----
    python scripts/diag_mask_shape.py            # 默认 30 帧
    python scripts/diag_mask_shape.py --frames 60

注意：只读调用 demo 模块，不修改 server/ 任何代码。
"""

import argparse
import glob
import shutil
import sys
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "demo"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import egomed_demo as demo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="复现掩膜形状不匹配")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--keep", type=int, default=2,
                    help="滚动窗口保留帧数（默认 2）。服务端 WINDOW_SIZE=1，"
                         "但长稳实测有 28 次 prune 被宿主删除守卫拦下，"
                         "窗口残留 2 帧 —— 尺寸不一致的帧撞在一起才出问题。")
    ap.add_argument("--image-dir", default=str(REPO_ROOT / "data" / "samples" / "acdc"))
    ap.add_argument("--target", default="RV cavity")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "diag" / "mask_shape"))
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.image_dir) / "*.jpg")))
    if not files:
        print(f"[FAIL] 没有样例帧：{args.image_dir}")
        return 2

    print("[repro] 样例帧尺寸分布：")
    sizes = {}
    for f in files:
        h, w = cv2.imread(f).shape[:2]
        sizes.setdefault((h, w), []).append(Path(f).name)
    for k, v in sizes.items():
        print(f"   {k[0]}x{k[1]} × {len(v)} 个")

    frames_dir = Path(args.out) / "frames"
    seg_dir = Path(args.out) / "seg"
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(seg_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    modality, name_to_gray = demo.resolve_modality(args.target)
    predictor, yolo_model = demo._get_models(modality)
    id_to_name = {
        k: v for k, v in demo.eng.load_yolo_class_names_from_model(yolo_model).items()
        if v in name_to_gray
    }
    class_id_to_gray = {k: name_to_gray[v] for k, v in id_to_name.items()}
    target_id = next((k for k, v in id_to_name.items() if v == args.target), None)
    print(f"[repro] 模态={modality} 目标={args.target} target_id={target_id}")

    failures = []
    ok_cnt = 0
    for k in range(1, args.frames + 1):
        src = Path(files[(k - 1) % len(files)])
        dst = frames_dir / f"{k:06d}.jpg"
        shutil.copyfile(src, dst)
        for f in sorted(frames_dir.glob("*.jpg"), key=lambda p: p.stem)[:-args.keep]:
            try:
                f.unlink(missing_ok=True)
            except (Exception, SystemExit):
                pass

        image_files, _ = demo.resolve_frames(frames_dir, seg_dir)
        cur_h, cur_w = cv2.imread(str(image_files[0])).shape[:2]
        try:
            demo._segment_target(
                predictor, yolo_model, image_files, frames_dir, target_id,
                id_to_name, class_id_to_gray, seg_dir, args.target,
                min_occurrence=1, min_area_ratio=0.001,
            )
            ok_cnt += 1
        except Exception:
            tb = traceback.format_exc()
            failures.append((k, src.name, cur_h, cur_w, tb))
            print(f"\n[repro] 第 {k} 帧复现异常（源帧 {src.name} {cur_h}x{cur_w}）：\n{tb}")

    print("\n" + "=" * 70)
    print(f"[repro] 共 {args.frames} 帧：成功 {ok_cnt}，异常 {len(failures)}")
    if failures:
        print("[repro] 异常帧清单（帧号 / 源帧 / 帧尺寸）：")
        for k, name, h, w, _ in failures:
            print(f"   第 {k} 帧  {name}  {h}x{w}")
        print("\n[repro] 首条完整 traceback：")
        print(failures[0][4])
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
