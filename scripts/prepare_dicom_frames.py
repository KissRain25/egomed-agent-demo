# -*- coding: utf-8 -*-
"""
将各模态的 DICOM 医学影像转换为可分割的 JPG 帧序列。

输出结构（对齐 ACDC 的 data/ACDC/img/<病例>/）：
    data/Amos/img/<病例>/
    data/CAMUS/img/<病例>/
    data/Montgomery-County-CXR-Set/img/<病例>/
    data/PolypGen2021_MultiCenterData_v3/img/<病例>/

用法:
    python scripts/prepare_dicom_frames.py
"""
import os
import sys
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    print("请先安装 pydicom:  conda activate egomed && pip install pydicom")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"D:\BaiduNetdiskDownload\EgoMed-Screen")
OUTPUT_ROOT = Path(r"D:\EgoMed-Agent\data")

MODALITIES = {
    "Amos": {
        "dicom_dir": DATA_ROOT / "Amos" / "dicom",
        "output_dir": OUTPUT_ROOT / "Amos" / "img",
        # Amos dicom 子文件夹命名如 amos_0001, amos_0004...
        "case_pattern": "*",
        "file_pattern": "*.dcm",
    },
    "CAMUS": {
        "dicom_dir": DATA_ROOT / "CAMUS" / "dicom",
        "output_dir": OUTPUT_ROOT / "CAMUS" / "img",
        "case_pattern": "*",
        "file_pattern": "*.dcm",
    },
    "Montgomery-County-CXR-Set": {
        "dicom_dir": DATA_ROOT / "Montgomery-County-CXR-Set" / "dicom",
        "output_dir": OUTPUT_ROOT / "Montgomery-County-CXR-Set" / "img",
        "case_pattern": "*",
        "file_pattern": "*.dcm",
    },
    "PolypGen2021_MultiCenterData_v3": {
        "dicom_dir": DATA_ROOT / "PolypGen2021_MultiCenterData_v3" / "dicom",
        "output_dir": OUTPUT_ROOT / "PolypGen2021_MultiCenterData_v3" / "img",
        "case_pattern": "*",
        "file_pattern": "*.dcm",
    },
}


def dcm_to_image(dcm_path):
    """读取单个 DICOM 文件，返回归一化后的 uint8 numpy 数组 (H, W) 或 (H, W, 3)。"""
    ds = pydicom.dcmread(str(dcm_path))
    pixel_array = ds.pixel_array

    # 处理多帧 DICOM（如超声视频）
    if pixel_array.ndim == 3:
        # 取第一帧，或后续可以扩展为多帧
        pixel_array = pixel_array[0]

    # 归一化到 0-255
    if pixel_array.dtype != np.uint8:
        arr = pixel_array.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
        pixel_array = arr.astype(np.uint8)

    return pixel_array


def convert_modality(modality_name, cfg):
    """转换一个模态的所有 DICOM 到 JPG 帧。"""
    dicom_dir = cfg["dicom_dir"]
    output_dir = cfg["output_dir"]
    file_pattern = cfg["file_pattern"]

    if not dicom_dir.exists():
        print(f"  [{modality_name}] dicom 目录不存在: {dicom_dir}")
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    case_count = 0
    frame_count = 0

    # 遍历病例子文件夹
    for case_dir in sorted(dicom_dir.glob(cfg["case_pattern"])):
        if not case_dir.is_dir():
            continue

        case_name = case_dir.name
        case_output_dir = output_dir / case_name

        # 如果已存在则跳过（可重新跑，不会重复）
        if case_output_dir.exists() and any(case_output_dir.iterdir()):
            existing = len(list(case_output_dir.glob("*.jpg")))
            print(f"  [{modality_name}] {case_name}: 已存在 {existing} 帧，跳过")
            case_count += 1
            frame_count += existing
            continue

        case_output_dir.mkdir(parents=True, exist_ok=True)

        # 收集该病例的所有 dcm 文件
        dcm_files = sorted(case_dir.glob(file_pattern))
        if not dcm_files:
            print(f"  [{modality_name}] {case_name}: 无 DICOM 文件")
            continue

        case_frame_count = 0
        for idx, dcm_file in enumerate(dcm_files):
            try:
                img_arr = dcm_to_image(dcm_file)

                # 保存为 jpg
                frame_name = f"{case_name}_{idx:04d}.jpg"
                out_path = case_output_dir / frame_name

                if img_arr.ndim == 2:
                    Image.fromarray(img_arr, mode="L").save(out_path, quality=95)
                else:
                    Image.fromarray(img_arr).save(out_path, quality=95)

                case_frame_count += 1
            except Exception as e:
                print(f"    错误处理 {dcm_file.name}: {e}")
                continue

        print(f"  [{modality_name}] {case_name}: {case_frame_count} 帧 → {case_output_dir}")
        case_count += 1
        frame_count += case_frame_count

    return case_count, frame_count


def main():
    print("=" * 60)
    print("DICOM → JPG 帧转换")
    print(f"数据来源: {DATA_ROOT}")
    print(f"输出位置: {OUTPUT_ROOT}")
    print("=" * 60)

    total_cases = 0
    total_frames = 0

    for modality_name, cfg in MODALITIES.items():
        print(f"\n>>> 处理模态: {modality_name}")
        cases, frames = convert_modality(modality_name, cfg)
        total_cases += cases
        total_frames += frames
        print(f"    完成: {cases} 病例, {frames} 帧")

    print("\n" + "=" * 60)
    print(f"全部完成: {total_cases} 病例, {total_frames} 帧")
    print("=" * 60)


if __name__ == "__main__":
    main()
