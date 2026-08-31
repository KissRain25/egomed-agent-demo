#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 EgoMed-IEMIS models 里的 6 个权重到项目对应位置（走 hf-mirror 国内镜像）"""
import os
import sys
import time
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
DATASET = "daizywang/EgoMed-IEMIS"
REVISION = "main"

ROOT = Path(r"D:\EgoMed-Agent")
YOLO_RUN = ROOT / "runs" / "yolo26_det"

# 目标文件 -> (在线路径, 本地落盘路径)
# 本地路径与引擎 DATASET_CONFIGS 的权重路径保持一致
WEIGHTS = {
    "imaging_modality_classifier.pt": (
        f"models/classifier/imaging_modality_classifier.pt",
        ROOT / "runs" / "tool_selection_cls" / "tool_selection_yolo26m_cls_imgsz320" / "weights" / "best.pt",
    ),
    "ct_modality_target_detector.pt": (
        f"models/detectors/ct_modality_target_detector.pt",
        YOLO_RUN / "Amos_yolo26m_imgsz1024" / "weights" / "best.pt",
    ),
    "mri_modality_target_detector.pt": (
        f"models/detectors/mri_modality_target_detector.pt",
        YOLO_RUN / "ACDC_yolo26m_imgsz1024" / "weights" / "best.pt",
    ),
    "ultrasound_modality_target_detector.pt": (
        f"models/detectors/ultrasound_modality_target_detector.pt",
        YOLO_RUN / "CAMUS_yolo26m_imgsz1024" / "weights" / "best.pt",
    ),
    "xray_modality_target_detector.pt": (
        f"models/detectors/xray_modality_target_detector.pt",
        YOLO_RUN / "Montgomery-County-CXR-Set_yolo26m_imgsz1024" / "weights" / "best.pt",
    ),
    "endoscopy_modality_target_detector.pt": (
        f"models/detectors/endoscopy_modality_target_detector.pt",
        YOLO_RUN / "PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024" / "weights" / "best.pt",
    ),
}

# 期望大小（字节），用于校验下载是否完整
EXPECTED_SIZE = {
    "imaging_modality_classifier.pt": 20886230,
    "ct_modality_target_detector.pt": 44131417,
    "mri_modality_target_detector.pt": 44083289,
    "ultrasound_modality_target_detector.pt": 44084313,
    "xray_modality_target_detector.pt": 44078361,
    "endoscopy_modality_target_detector.pt": 44082265,
}


def download_one(url, dest, expected):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print(f"[*] 下载 {dest.parent.parent.name}")
    print(f"    来源: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done / total * 100
                print(f"\r    {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
    print()
    size = os.path.getsize(tmp)
    if size != expected:
        print(f"    [警告] 大小不符: 实际 {size} != 期望 {expected}")
        tmp.unlink()
        return False
    os.replace(tmp, dest)
    print(f"    [OK] 已保存: {dest} ({size/1e6:.1f} MB)")
    return True


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ok, fail = 0, 0
    for name, (rel, dest) in WEIGHTS.items():
        if only and name != only:
            continue
        url = f"{MIRROR}/datasets/{DATASET}/resolve/{REVISION}/{rel}"
        try:
            if download_one(url, dest, EXPECTED_SIZE[name]):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"    [失败] {name}: {e}")
            fail += 1
    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
