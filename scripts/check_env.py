#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""环境自检（N29）—— 换机器/交付时先跑它，一次看清"缺什么、怎么补"

检查项（每项给 PASS / WARN / FAIL + 修复建议）
---------------------------------------------
  ① 解释器与 Python 版本        ⑦ 数据样例（帧 / 音频 / 坏帧）
  ② 关键依赖包（版本 / 是否缺）  ⑧ 端口 8000 是否被占用
  ③ torch 与 CUDA（设备、显存）  ⑨ 磁盘剩余空间
  ④ 模型权重清单                ⑩ tkinter（现场控制台需要）
  ⑤ SAM2 源码与配置             ⑪ site-packages 里的残留坏目录（~开头）
  ⑥ 配置文件                    ⑫ git 工作区状态

用法
----
    python scripts/check_env.py            # 常规自检（秒级）
    python scripts/check_env.py --deep     # 额外导入引擎依赖链（较慢，约 10~20s）

输出：终端表格 + 结论；有 FAIL 时退出码 1（便于脚本串起来）

归属：闫（交付 / 运维工具）｜ 关联：docs/总计划.md 阶段四 N29、docs/上手指南.md
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECT_PY = (3, 12)
MIN_FREE_GB = 10.0

# (pip 名, 用途, 是否必需)
PACKAGES = [
    ("torch", "推理后端（YOLO/SAM2/ASR）", True),
    ("ultralytics", "YOLO 检测器", True),
    ("fastapi", "服务端框架", True),
    ("uvicorn", "服务端 ASGI", True),
    ("python-multipart", "接收 multipart 上传（契约 A1/A2 必需）", True),
    ("opencv-python", "图像读写 / 摄像头", True),
    ("numpy", "数值计算", True),
    ("Pillow", "图像与中文 HUD 渲染", True),
    ("requests", "客户端 HTTP", True),
    ("PyYAML", "客户端配置文件（N25）", True),
    ("faster-whisper", "A2 语音识别（不做语音可缺）", False),
    ("sounddevice", "实时麦克风（N4.6，可缺）", False),
    ("psutil", "长稳采内存 / 诊断包", False),
]
STDLIB_MODULES = [("tkinter", "现场控制台界面（N26）", True)]


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str, str]] = []   # (级别, 名称, 说明, 建议)

    def add(self, ok: bool, name: str, detail: str, hint: str = "", level: str = "FAIL") -> None:
        self.items.append(("PASS" if ok else level, name, detail, "" if ok else hint))

    def show(self) -> int:
        icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
        print("=" * 78)
        print("环境自检（N29）")
        print("=" * 78)
        for lv, name, detail, hint in self.items:
            print(f"{icon[lv]} {name}：{detail}")
            if hint:
                print(f"        → {hint}")
        fails = [i for i in self.items if i[0] == "FAIL"]
        warns = [i for i in self.items if i[0] == "WARN"]
        print("-" * 78)
        print(f"结论：{len(self.items)} 项检查 → PASS {len(self.items)-len(fails)-len(warns)}"
              f" / WARN {len(warns)} / FAIL {len(fails)}")
        if fails:
            print("存在 FAIL：按上面的 → 提示处理后重跑本脚本")
        else:
            print("可以开工：双击「现场控制台.bat」或「启动模拟眼镜.bat」")
        print("=" * 78)
        return 1 if fails else 0


def pkg_version(pip_name: str) -> str | None:
    try:
        return version(pip_name)
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return None


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="EgoMed-Agent 环境自检（N29）")
    ap.add_argument("--deep", action="store_true", help="额外导入引擎依赖链（较慢）")
    args = ap.parse_args(argv)

    rep = Report()

    # ① 解释器与版本
    pv = sys.version_info
    rep.add((pv.major, pv.minor) == EXPECT_PY, "Python 版本",
            f"{platform.python_version()}（期望 {EXPECT_PY[0]}.{EXPECT_PY[1]}）",
            "换用 egomed 环境：双击「安装环境.bat」，或设 EGOMED_PY 指向它的 python.exe",
            level="WARN")
    in_conda = "envs" in str(sys.executable).replace("\\", "/")
    rep.add(in_conda, "解释器位置", sys.executable,
            "建议用 conda 环境内的解释器（egomed），避免系统 Python 缺依赖")

    # ② 依赖包
    missing_req, missing_opt, vers = [], [], []
    for pip_name, why, required in PACKAGES:
        v = pkg_version(pip_name)
        if v:
            vers.append(f"{pip_name} {v}")
        elif required:
            missing_req.append(f"{pip_name}（{why}）")
        else:
            missing_opt.append(f"{pip_name}（{why}）")
    rep.add(not missing_req, "必需依赖",
            f"{len(PACKAGES)-len(missing_req)-len(missing_opt)}/{len(PACKAGES)} 就绪"
            + (f"；缺 {len(missing_req)} 个" if missing_req else ""),
            "缺：" + "、".join(missing_req) + " → `pip install -r requirements.txt`")
    if missing_opt:
        rep.add(True, "可选依赖", f"缺 {len(missing_opt)} 个（不影响主链路）",
                "缺：" + "、".join(missing_opt), level="WARN")

    # 标准库模块
    for mod, why, required in STDLIB_MODULES:
        ok = importlib.util.find_spec(mod) is not None
        rep.add(ok, f"{mod} 可用", why if ok else f"缺少 {mod}",
                f"{mod} 属于标准库，缺失通常是精简版 Python → 换官方/conda 版 Python")

    # ③ torch 与 CUDA
    try:
        import torch  # noqa: PLC0415
        cuda = torch.cuda.is_available()
        detail = (f"torch {torch.__version__}｜CUDA {'可用' if cuda else '不可用'}")
        if cuda:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            detail += f"｜{name}（{total:.1f} GB）"
        rep.add(cuda, "GPU 推理环境", detail,
                "服务端需要 CUDA；确认显卡驱动 + 装的是 CUDA 版 torch（不是 CPU 版）")
    except Exception as exc:  # noqa: BLE001
        rep.add(False, "GPU 推理环境", f"torch 导入失败：{exc}",
                "`pip install -r requirements.txt` 后重试")

    # ④ 模型权重
    ckpt = sorted((PROJECT_ROOT / "checkpoints").glob("*.pt"))
    mdl = sorted((PROJECT_ROOT / "models").rglob("*.pt"))
    if ckpt:
        rep.add(True, "SAM2 权重", f"{len(ckpt)} 个：" + "、".join(p.name for p in ckpt))
    else:
        rep.add(False, "SAM2 权重", "checkpoints/ 下没有 .pt",
                "`python scripts/download_models.py`（或手动放入 "
                "checkpoints/sam2.1_hiera_base_plus.pt）")
    rep.add(bool(mdl), "检测/分类权重", f"models/ 下 {len(mdl)} 个："
            + "、".join(p.name for p in mdl[:4]) + ("…" if len(mdl) > 4 else ""),
            "缺少检测器权重 → `python scripts/download_models.py`")

    # ⑤ SAM2 源码
    sam2_dir = PROJECT_ROOT / "sam2"
    cfg = list((sam2_dir / "configs").glob("**/*.yaml")) if sam2_dir.exists() else []
    rep.add(sam2_dir.exists() and bool(cfg), "SAM2 源码与配置",
            f"{'有' if sam2_dir.exists() else '缺'} sam2/，配置 {len(cfg)} 个",
            "`pip install -r requirements.txt` 会按 REQ 里的 git 地址拉取 SAM2")

    # ⑥ 配置
    live_cfg = PROJECT_ROOT / "configs" / "live.yaml"
    rep.add(live_cfg.exists(), "客户端配置", str(live_cfg.relative_to(PROJECT_ROOT)) if
            live_cfg.exists() else "缺 configs/live.yaml",
            "从仓库同步该文件（N25 配置化）", level="WARN")

    # ⑦ 数据样例
    acdc = PROJECT_ROOT / "data" / "samples" / "acdc"
    audio = PROJECT_ROOT / "data" / "samples" / "audio"
    n_img = len(list(acdc.glob("*.jpg"))) if acdc.exists() else 0
    n_wav = len(list(audio.glob("*.wav"))) if audio.exists() else 0
    rep.add(n_img > 0 and n_wav > 0, "数据样例（自动化/演示用）",
            f"帧 {n_img} 张 / 语音 {n_wav} 段",
            "缺样例→ 控制台可改用真实摄像头（可选源 cam）或指定自己的帧目录")

    # ⑧ 端口
    busy = port_in_use(8000)
    rep.add(not busy, "端口 8000", "空闲" if not busy else "已被占用",
            "占用多为上一轮服务端没关：`netstat -ano | findstr :8000` 找到 PID 后结束它",
            level="WARN")

    # ⑨ 磁盘
    free = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    rep.add(free >= MIN_FREE_GB, "磁盘剩余", f"{free:.1f} GB（建议 ≥{MIN_FREE_GB:g} GB）",
            "清理 runs/ 下旧产物（长稳/评测报告与帧会累积）", level="WARN")

    # ⑩ site-packages 残留坏目录（~ 开头，会让 pip 打警告）
    sp = Path(sys.prefix) / "Lib" / "site-packages"
    junk = [p.name for p in sp.glob("~*")] if sp.exists() else []
    rep.add(not junk, "site-packages 干净", "无残留" if not junk else f"残留 {junk}",
            "手动删掉这些 ~ 开头的目录（上次安装中断留下的），否则 pip 会持续告警",
            level="WARN")

    # ⑪ git
    try:
        r = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--short"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="ignore", timeout=10)
        dirty = len([l for l in r.stdout.splitlines() if l.strip()])
        head = subprocess.run(["git", "-C", str(PROJECT_ROOT), "log", "-1", "--oneline"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="ignore", timeout=10).stdout.strip()
        rep.add(dirty == 0, "git 工作区", f"{'干净' if not dirty else f'{dirty} 个改动'}｜{head}",
                "交付前先提交（`git add -A && git commit`）", level="WARN")
    except Exception:  # noqa: BLE001
        rep.add(False, "git 工作区", "未检测（非 git 目录或未装 git）", level="WARN")

    # ⑫ 可选：深度导入
    if args.deep:
        for mod in ("ultralytics", "fastapi", "uvicorn", "cv2", "yaml"):
            try:
                __import__(mod)
                rep.add(True, f"导入 {mod}", "OK")
            except Exception as exc:  # noqa: BLE001
                rep.add(False, f"导入 {mod}", f"{type(exc).__name__}: {exc}",
                        "依赖链不完整 → 重装 requirements.txt")

    return rep.show()


if __name__ == "__main__":
    sys.exit(main())
