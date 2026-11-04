#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""帧级模态分类（N48）—— 把仓库里**已经躺着**的分类权重真正接进服务端。

背景（为什么要做）
------------------
契约 §5.3 第 8 条早就写明：「**模态切换：连续 3 帧一致才切，低置信保持原模态**」，
附录 A.10 **P1.5** 也把"帧级模态分类"列为待办，但一直没做：
现在的 `modality` 是 `resolve_modality(target)` **从目标反推**出来的，
医生还没说话时，服务端只能回一句"请先告诉我要分割哪个器官"。

本模块提供契约 §5.3 要求的能力：**帧 → 模态**。
关键事实（2026-11-04 实测）：`models/imaging_modality_classifier.pt`（19.9 MB）
是 **Ultralytics 分类模型**（`train_args.task='classify'`、`model='yolo26m-cls.pt'`），
类别名 `{0:'CT', 1:'Endoscopy', 2:'MRI', 3:'Ultrasound', 4:'Xray'}`，
**正好一一对应**服务端 `scripts/demo/egomed_demo.py::MODALITIES` 的 5 个模态键。

⚠️ **身份说明（防答辩被问住）**：该权重的训练数据是 `yolo_cls_tool_selection`
（"工具选择"分类器），**不能断言它就是论文里的"帧级模态分类器"**。
本模块只承诺工程口径的准确率（五模态各 20 帧判对 ≥90%），不宣称复现论文那一项。

设计约束
--------
1. **绝不抛异常**：权重缺失 / 加载失败 / 解码失败 / 推理异常 → 一律返回 `name=None`，
   由调用方走契约 A1 边界 1（清空字段 + 统一文案）。**分类器故障绝不能把 A1 打成 500。**
2. **零新增依赖**：只用 ultralytics（服务端已有）+ PIL / OpenCV（二选一，都没有就降级）。
3. **懒加载 + 进程内缓存**：与 `demo._get_models` 同款思路，模型只加载一次；
   加载失败会被记住，不会每帧重试（避免每帧都卡在加载上）。
4. **默认用 GPU 推理**（可用时）：分类器很小（19.9 MB），GPU 上十几毫秒一帧。
   ⚠️ **不能用 CPU 模式**——ultralytics 的 select_device() 在 CPU 模式下会把
   `CUDA_VISIBLE_DEVICES` 置空，而 torch 只在首次初始化 CUDA 时读它一次，
   于是同进程里分割链路的 GPU 推理会全部抛 "Invalid device id"（A1 全 500）。
   本模块用 `_cvd_guard()` 进出还原环境变量，确保不会污染进程里其它模块。
   需要时仍可设环境变量 `EGOMED_MODALITY_CLS_DEVICE` 覆盖（`cuda:0` / `cpu`）。

归属：闫 ｜ 关联：docs/API_CONTRACT.md §5.3 / 附录 A.10 P1.5、docs/总计划.md N48
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "imaging_modality_classifier.pt"

# ---------------------------------------------------------------------------
# 类别名 → 服务端模态键（严格对应 scripts/demo/egomed_demo.py::MODALITIES）
# ---------------------------------------------------------------------------
# 分类器的 5 类与服务端 5 个模态**一一对应**，映射是硬对应关系，不做模糊匹配：
#   MRI        → ACDC (MRI 心脏)        权重 runs/yolo26_det/ACDC_yolo26m_imgsz1024
#   Ultrasound → CAMUS (超声 心脏)      权重 runs/yolo26_det/CAMUS_yolo26m_imgsz1024
#   CT         → Amos (CT 腹部)         权重 runs/yolo26_det/Amos_yolo26m_imgsz1024
#   Xray       → Montgomery (X光 胸片)   权重 runs/yolo26_det/Montgomery-County-CXR-Set_yolo26m_imgsz1024
#   Endoscopy  → PolypGen (内窥镜)      权重 runs/yolo26_det/PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024
LABEL_TO_MODALITY: dict[str, str] = {
    "MRI": "ACDC (MRI 心脏)",
    "Ultrasound": "CAMUS (超声 心脏)",
    "CT": "Amos (CT 腹部)",
    "Xray": "Montgomery (X光 胸片)",
    "Endoscopy": "PolypGen (内窥镜)",
}

# 推理设备：环境变量 `EGOMED_MODALITY_CLS_DEVICE` 优先；否则**能用 GPU 就用 GPU**。
#
# ⚠️ 为什么默认不是 CPU（2026-11-04 实测踩坑）：
#   ultralytics 的 select_device() 在 CPU 模式下会执行
#       os.environ["CUDA_VISIBLE_DEVICES"] = ""    # 注释原文: force torch.cuda.is_available() = False
#   torch 只在**首次初始化 CUDA 时**读一次这个变量 → 之后 device_count() 恒为 0
#   → 分割链路 torch.cuda.get_device_properties(0) 抛 AssertionError("Invalid device id")
#   → 已设目标的 A1 全部 500。分类器若跑 CPU，等于**在服务端进程里埋雷**。
#   所以默认用 GPU（顺带更快：CPU 71~136ms/帧 → GPU 十几毫秒），
#   并且 classify() 全程用 _cvd_guard() 保护环境变量，见该函数注释。
DEVICE_ENV = "EGOMED_MODALITY_CLS_DEVICE"


def _resolve_device() -> str:
    """实际推理设备（延迟解析，避免模块导入时就初始化 CUDA）。"""
    env = os.environ.get(DEVICE_ENV, "").strip()
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # noqa: BLE001 没 torch / 驱动异常 → CPU
        pass
    return "cpu"


@contextlib.contextmanager
def _cvd_guard():
    """保护 `CUDA_VISIBLE_DEVICES`：进出都还原。

    ultralytics 的 select_device() 会**改写**这个环境变量（CPU 模式直接置空）。
    torch 只在首次初始化 CUDA 时读它一次，一旦被置空，后续
    `torch.cuda.get_device_properties(0)` 就会抛 "Invalid device id"，
    把同进程里其它模块（分割链路）的 GPU 推理一起搞挂。

    放在这里做还原，本模块就**不会破坏进程里其它模块的 GPU 可用性**。
    """
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved


@dataclass(frozen=True)
class ModalityGuess:
    """一次帧级模态猜测的结果。"""

    name: str | None      # 模态键（如 "ACDC (MRI 心脏)"）；None = 判不出 → 走契约 A1 边界 1
    confidence: float     # 分类器 top-1 置信度（0~1）
    raw_label: str        # 原始类别名（CT/Endoscopy/MRI/Ultrasound/Xray），仅供日志排查

    @property
    def ok(self) -> bool:
        return bool(self.name)


# ---------------------------------------------------------------------------
# 模型懒加载（进程内只加载一次；失败会被记住）
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_READY = False          # 尝试过没有（成功或失败都置 True，避免每帧重试）
_MODEL_ERROR = ""             # 失败原因（写进日志，便于排障）
_LOAD_LOCK = threading.Lock()
_DEVICE_CACHE: str | None = None


def _device() -> str:
    """实际设备（进程内缓存一次）。"""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        _DEVICE_CACHE = _resolve_device()
    return _DEVICE_CACHE


def _load_model():
    """懒加载分类模型。返回 None 表示不可用（**不抛异常**）。"""
    global _MODEL, _MODEL_READY, _MODEL_ERROR
    if _MODEL_READY:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL_READY:                 # 双重检查（多线程下别重复加载）
            return _MODEL
        _MODEL_READY = True
        if not WEIGHTS_PATH.is_file():
            _MODEL_ERROR = f"权重不存在：{WEIGHTS_PATH}"
            return None
        try:
            from ultralytics import YOLO
            _MODEL = YOLO(str(WEIGHTS_PATH))
            names = getattr(_MODEL, "names", None) or {}
            unknown = set(names.values()) - set(LABEL_TO_MODALITY)
            if unknown:
                # 类别名对不上映射表 → 宁可不猜（返回 None），也不要给出错误的模态
                _MODEL_ERROR = (f"分类器类别 {sorted(names.values())} 与映射表不匹配"
                                f"（未知：{sorted(unknown)}）")
                _MODEL = None
        except Exception as exc:  # noqa: BLE001  缺依赖 / 权重损坏 / 版本不兼容
            _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
            _MODEL = None
    return _MODEL


def _decode(image_bytes: bytes):
    """把上传的字节解成 ultralytics 能吃的图。返回 None 表示解不了。

    PIL 分支给 RGB（ultralytics 对 PIL 走 `mode="RGB"`），
    OpenCV 分支给 BGR（ultralytics 对 ndarray 走 `mode="BGR"`）—— 两者都对。
    """
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as im:
            return im.convert("RGB")
    except Exception:  # noqa: BLE001  换 OpenCV
        pass
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img if img is not None else None
    except Exception:  # noqa: BLE001  两个解码器都没有 / 都失败
        return None


def classify(image_bytes: bytes, min_conf: float = 0.0) -> ModalityGuess:
    """帧 → 模态猜测。**任何异常都降级为 `name=None`，绝不抛出。**

    `min_conf`
    ---------
    置信度阈值（0~1）。低于阈值视为"判不出"→ 返回 `name=None`。
    分类器**只有 5 类、没有"不是医学影像"这一类**，所以非影像输入会被强行归到某一类；
    **阈值是这个功能的核心参数，不是可选项**（默认值由离线评测实测确定，见
    `client/eval_modality_cls.py`）。
    """
    if not image_bytes:
        return ModalityGuess(None, 0.0, "")
    try:
        model = _load_model()
        if model is None:
            return ModalityGuess(None, 0.0, "")
        img = _decode(image_bytes)
        if img is None:
            return ModalityGuess(None, 0.0, "")
        with _cvd_guard():
            results = model.predict(img, device=_device(), verbose=False)
        if not results:
            return ModalityGuess(None, 0.0, "")
        probs = results[0].probs
        idx = int(probs.top1)
        conf = float(probs.top1conf)
        names = getattr(results[0], "names", None) or {}
        raw = str(names.get(idx, idx))
        name = LABEL_TO_MODALITY.get(raw, "")
        if not name or conf < min_conf:
            return ModalityGuess(None, conf, raw)
        return ModalityGuess(name, conf, raw)
    except Exception:  # noqa: BLE001  推理异常一律降级，绝不让 A1 变成 500
        return ModalityGuess(None, 0.0, "")


def short_name(modality_key: str) -> str:
    """把模态键变成给医生看的短名。

    "ACDC (MRI 心脏)" → "MRI 心脏"；"Amos (CT 腹部)" → "CT 腹部"；
    "Montgomery (X光 胸片)" → "X光 胸片"；"PolypGen (内窥镜)" → "内窥镜"
    """
    if "(" in modality_key and modality_key.endswith(")"):
        return modality_key[modality_key.index("(") + 1:-1].strip()
    return modality_key


def status() -> dict:
    """分类器可用性（给 /healthz 或 doctor 用）。"""
    model = _load_model()
    return {
        "available": model is not None,
        "weights": str(WEIGHTS_PATH),
        "weights_exists": WEIGHTS_PATH.is_file(),
        "device": _device(),
        "error": _MODEL_ERROR,
        "labels": sorted(LABEL_TO_MODALITY),
    }


# ---------------------------------------------------------------------------
# 自测 CLI：五模态各取样例帧跑一遍，打印混淆矩阵与置信度分布
# ---------------------------------------------------------------------------
# data/samples 下的样例目录 → 期望的模态键
SAMPLE_DIRS: dict[str, str] = {
    "acdc": "ACDC (MRI 心脏)",
    "camus": "CAMUS (超声 心脏)",
    "amos": "Amos (CT 腹部)",
    "cxr": "Montgomery (X光 胸片)",
    "polyp": "PolypGen (内窥镜)",
}


def _selftest(frames_per_modality: int) -> int:
    import collections

    samples_root = PROJECT_ROOT / "data" / "samples"
    print("=" * 74)
    print(f"帧级模态分类自测（每模态最多 {frames_per_modality} 帧，device={_device()}）")
    print(f"权重：{WEIGHTS_PATH}  存在={WEIGHTS_PATH.is_file()}")
    st = status()
    if not st["available"]:
        print(f"[!!] 分类器不可用：{st['error']}")
        return 2
    print("=" * 74)

    confusion: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int))
    confs: list[tuple[str, str, float]] = []
    total = ok = 0

    for folder, expect in SAMPLE_DIRS.items():
        d = samples_root / folder
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in
                       (".jpg", ".jpeg", ".png")) if d.is_dir() else []
        for p in files[:frames_per_modality]:
            g = classify(p.read_bytes())
            total += 1
            guess = g.name or "(判不出)"
            confusion[expect][guess] += 1
            confs.append((expect, guess, g.confidence))
            if g.name == expect:
                ok += 1
            print(f"  {folder:<7} {p.name[:26]:<28} → {guess:<22} conf={g.confidence:.3f}")

    print("-" * 74)
    print("混淆矩阵（行=真实，列=预测）：")
    cols = sorted({c for r in confusion.values() for c in r})
    short = {m: m.split(" ")[0] for m in cols}
    print(f"{'真实\\预测':<20}" + "".join(f"{short[c]:<12}" for c in cols))
    for real in SAMPLE_DIRS.values():
        row = confusion.get(real, {})
        print(f"{short.get(real, real):<20}" +
              "".join(f"{row.get(c, 0):<12}" for c in cols))

    print("-" * 74)
    by_mod: dict[str, list[float]] = collections.defaultdict(list)
    for real, _guess, c in confs:
        by_mod[real].append(c)
    print("置信度分布（按真实模态）：")
    for real, cs in by_mod.items():
        cs_sorted = sorted(cs)
        p5 = cs_sorted[max(0, int(len(cs_sorted) * 0.05))]
        p50 = cs_sorted[len(cs_sorted) // 2]
        print(f"  {real:<22} n={len(cs):<3} min={cs_sorted[0]:.3f} "
              f"P5={p5:.3f} P50={p50:.3f} max={cs_sorted[-1]:.3f}")

    print("-" * 74)
    print(f"Top-1 判对：{ok}/{total} = {ok / total * 100:.1f}%" if total else "无样本")
    return 0 if total and ok == total else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="帧级模态分类模块（N48）")
    ap.add_argument("--selftest", action="store_true", help="对 data/samples 五模态样例帧自测")
    ap.add_argument("--frames", type=int, default=8, help="每模态取几帧（默认 8）")
    ap.add_argument("--status", action="store_true", help="只打印分类器可用性")
    args = ap.parse_args()
    if args.status:
        for k, v in status().items():
            print(f"{k}: {v}")
        return 0
    if args.selftest:
        return _selftest(args.frames)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
