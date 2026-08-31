# -*- coding: utf-8 -*-
"""
将各模态 rawdata 录屏视频提取为可分割的 JPG 帧序列。

输出结构（对齐 ACDC 的 data/ACDC/img/<病例>/）：
    data/Amos/img/<病例>/
    data/CAMUS/img/<病例>/
    data/Montgomery-County-CXR-Set/img/<病例>/
    data/PolypGen2021_MultiCenterData_v3/img/<病例>/

用法:
    python scripts/prepare_rawdata_frames.py
    python scripts/prepare_rawdata_frames.py --modalities Amos CAMUS
    python scripts/prepare_rawdata_frames.py --sample-rate 3
"""
import argparse
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"D:\BaiduNetdiskDownload\EgoMed-Screen")
OUTPUT_ROOT = Path(r"D:\EgoMed-Agent\data")

MODALITIES = {
    "Amos": {
        "rawdata_dir": DATA_ROOT / "Amos" / "rawdata",
        "output_dir": OUTPUT_ROOT / "Amos" / "img",
    },
    "CAMUS": {
        "rawdata_dir": DATA_ROOT / "CAMUS" / "rawdata",
        "output_dir": OUTPUT_ROOT / "CAMUS" / "img",
    },
    "Montgomery-County-CXR-Set": {
        "rawdata_dir": DATA_ROOT / "Montgomery-County-CXR-Set" / "rawdata",
        "output_dir": OUTPUT_ROOT / "Montgomery-County-CXR-Set" / "img",
    },
    "PolypGen2021_MultiCenterData_v3": {
        "rawdata_dir": DATA_ROOT / "PolypGen2021_MultiCenterData_v3" / "rawdata",
        "output_dir": OUTPUT_ROOT / "PolypGen2021_MultiCenterData_v3" / "img",
    },
}


def extract_video_frames(video_path, output_dir, case_name, sample_rate=3):
    """Extract frames from a video file, saving every N-th frame.

    Returns (num_extracted, total_frames) or (0, 0) on error.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ERROR: Cannot open video {video_path}")
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % sample_rate == 0:
            out_path = output_dir / f"{case_name}_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        n += 1

    cap.release()
    return saved, n


def process_modality(modality_name, cfg, sample_rate=3):
    """Process all videos for one modality."""
    rawdata_dir = cfg["rawdata_dir"]
    output_dir = cfg["output_dir"]

    if not rawdata_dir.exists():
        print(f"  [{modality_name}] rawdata 目录不存在: {rawdata_dir}")
        return 0, 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = [d for d in rawdata_dir.iterdir() if d.is_dir()]
    case_dirs.sort(key=lambda x: int(x.name) if x.name.isdigit() else x.name)

    processed = 0
    skipped = 0
    total_frames = 0

    for case_dir in case_dirs:
        case_name = case_dir.name
        case_output_dir = output_dir / case_name

        # Skip if already exists and has frames
        if case_output_dir.exists() and any(case_output_dir.glob("*.jpg")):
            existing = len(list(case_output_dir.glob("*.jpg")))
            skipped += 1
            if skipped <= 3 or skipped % 20 == 0:
                print(f"  [{modality_name}] {case_name}: 已存在 {existing} 帧，跳过")
            continue

        # Find video file
        mp4_files = list(case_dir.glob("*.mp4")) + list(case_dir.glob("*.MP4"))
        if not mp4_files:
            print(f"  [{modality_name}] {case_name}: 无 MP4 文件")
            continue

        video_path = mp4_files[0]
        saved, total = extract_video_frames(
            video_path, case_output_dir, case_name, sample_rate
        )

        if saved > 0:
            print(f"  [{modality_name}] {case_name}: {saved}/{total} 帧 → {case_output_dir}")
            processed += 1
            total_frames += saved
        else:
            print(f"  [{modality_name}] {case_name}: 提取失败")

    return processed, skipped, total_frames


def main():
    parser = argparse.ArgumentParser(description="从 rawdata 视频提取帧")
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=list(MODALITIES.keys()),
        default=list(MODALITIES.keys()),
        help="要处理的模态（默认全部）",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=3,
        help="抽帧间隔（每 N 帧保存 1 帧，默认 3）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Rawdata 视频 → JPG 帧提取")
    print(f"数据来源: {DATA_ROOT}")
    print(f"输出位置: {OUTPUT_ROOT}")
    print(f"抽帧间隔: 每 {args.sample_rate} 帧保存 1 帧")
    print("=" * 60)

    total_processed = 0
    total_skipped = 0
    total_frames = 0

    for modality_name in args.modalities:
        cfg = MODALITIES[modality_name]
        print(f"\n>>> 处理模态: {modality_name}")
        processed, skipped, frames = process_modality(
            modality_name, cfg, args.sample_rate
        )
        total_processed += processed
        total_skipped += skipped
        total_frames += frames
        print(f"    完成: 新处理 {processed} 病例, 跳过 {skipped} 病例, {frames} 帧")

    print("\n" + "=" * 60)
    print(f"全部完成: 新处理 {total_processed} 病例, 跳过 {total_skipped} 病例, {total_frames} 帧")
    print("=" * 60)


if __name__ == "__main__":
    main()
