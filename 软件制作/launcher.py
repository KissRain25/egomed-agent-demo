#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EgoMed-Agent 统一启动器（骨架 v0.1）

把原来散在 11 个 .bat 里的"解释器解析 + 参数拼装"收敛到一处，
并提供统一子命令与返回码，供 GUI / 脚本 / CI 调用。

本文件当前是**骨架**：解释器探测、配置加载、环境自检（doctor）已实装；
demo / regress / soak 等长任务先给"未实现"占位，按增量 2~4 逐步接入。
计划见同目录 `计划_软件化实施.md`。

用法
----
    python launcher.py                 # 打开 GUI（默认）
    python launcher.py doctor          # 环境自检（返回码 0 通过 / 2 有问题）
    python launcher.py console         # 直接开 GUI
    python launcher.py calibrate       # 机位校准
    python launcher.py diag            # 生成诊断包
    python launcher.py regress         # 一键回归（占位）
    python launcher.py soak -m 60      # 长稳测试（占位）

返回码
------
    0 成功 | 1 用户取消 | 2 环境问题(E3xxx) | 3 连接问题(E1xxx)
    4 验收失败 | 5 功能未实现（骨架阶段）

归属：闫 ｜ 关联：docs/软件化交付流程.md、软件制作/计划_软件化实施.md
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- 常量

VERSION = "0.1.0"
CONTRACT_VERSION = "v1.0.3"

RC_OK = 0
RC_CANCELLED = 1
RC_ENV = 2
RC_CONN = 3
RC_ACCEPT = 4
RC_NOTIMPL = 5

STAGE_DIR = Path(__file__).resolve().parent

# 候选解释器（收敛了原来 11 个 bat 里重复的 6 行逻辑）
PY_CANDIDATES = [
    os.environ.get("EGOMED_PY", ""),
    r"D:\ScienceApp\anaconda\envs\egomed\python.exe",
    str(Path.home() / "anaconda3/envs/egomed/python.exe"),
    str(Path.home() / "miniconda3/envs/egomed/python.exe"),
    str(Path(os.environ.get("LOCALAPPDATA", "")) / "anaconda3/envs/egomed/python.exe"),
    r"C:\ProgramData\anaconda3\envs\egomed\python.exe",
]

# doctor 需要检查的权重目录（存在即可，不深究内容）
WEIGHT_DIRS = [
    "runs/yolo26_det/Amos_yolo26m_imgsz1024/weights",
    "runs/yolo26_det/CAMUS_yolo26m_imgsz1024/weights",
    "runs/yolo26_det/Montgomery-County-CXR-Set_yolo26m_imgsz1024/weights",
    "runs/yolo26_det/PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024/weights",
    "runs/yolo26_det/ACDC_yolo26m_imgsz1024/weights",
]

REQUIRED_PACKAGES = [("cv2", "opencv-python"), ("numpy", "numpy"), ("PIL", "pillow")]


# ---------------------------------------------------------------- 基础工具

def setup_stdout() -> None:
    """中文控制台兜底：绝不因编码问题崩掉（历史上 emoji 直接崩过）。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def log(tag: str, message: str) -> None:
    """统一输出格式：[OK] / [!!] / [--] / [??]  —— 只用 ASCII 标记。"""
    print(f"{tag} {message}")


def find_project_root() -> Path:
    """定位项目根：本文件既可能在根目录，也可能在子目录（草稿区）。"""
    for cand in (STAGE_DIR, STAGE_DIR.parent):
        if (cand / "client" / "live_client.py").exists():
            return cand
    return STAGE_DIR.parent


def find_python() -> str | None:
    """返回第一个可用的解释器路径；找不到返回 None（对应 E3001）。"""
    for cand in PY_CANDIDATES:
        if cand and Path(cand).is_file():
            return cand
    # 最后兜底：PATH 里的 python
    from shutil import which
    return which("python")


def load_config(root: Path) -> tuple[dict, str]:
    """读 config.toml（四层优先级里只负责"文件"这一层）。

    返回 (配置字典, 说明)。文件缺失/损坏都不抛异常——启动不该被配置卡住。
    """
    path = root / "config.toml"
    if not path.exists():
        path = STAGE_DIR / "config.toml"
    if not path.exists():
        return default_config(), "未找到 config.toml，使用内置默认值"

    try:
        import tomllib  # Python 3.11+
        with path.open("rb") as f:
            return tomllib.load(f), f"已加载 {path}"
    except Exception as exc:
        return default_config(), f"config.toml 无法解析（{exc}），使用内置默认值"


def default_config() -> dict:
    return {
        "server": {"url": "http://127.0.0.1:8000", "timeout_s": 3.0},
        "capture": {"source": "video", "fps": 5.0, "max_side": 1280,
                    "jpeg_quality": 80, "max_bytes": 307200},
        "display": {"renderer": "window"},
        "logging": {"level": "info", "dir": "runs/logs"},
    }


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def server_healthy(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------- 子命令

def cmd_doctor(args) -> int:
    """环境自检：只读检查，不改任何东西。对应错误码 E3001~E3004。"""
    root = find_project_root()
    problems = 0

    log("[--]", f"EgoMed-Agent 环境自检  v{VERSION}（契约 {CONTRACT_VERSION}）")
    log("[--]", f"项目根：{root}")

    # 1) 解释器
    py = find_python()
    if py is None:
        log("[!!]", "E3001 找不到可用 Python 解释器")
        log("     ", "解决：先跑「安装环境.bat」，或设置环境变量 EGOMED_PY")
        problems += 1
    else:
        log("[OK]", f"解释器：{py}")
        if sys.version_info < (3, 10):
            log("[!!]", f"E3001 Python 版本过低：{sys.version.split()[0]}（需 ≥3.10）")
            problems += 1

    # 2) 依赖
    for mod, pkg in REQUIRED_PACKAGES:
        try:
            __import__(mod)
            log("[OK]", f"依赖：{pkg}")
        except Exception:
            log("[!!]", f"E3002 缺少依赖 {pkg}")
            problems += 1

    try:
        import torch  # noqa: F401
        log("[OK]", "依赖：torch")
    except Exception:
        log("[--]", "依赖：torch 未安装（服务端推理需要，客户端不强制）")

    try:
        import tkinter  # noqa: F401
        log("[OK]", "GUI：tkinter 可用")
    except Exception:
        log("[!!]", "E3002 tkinter 不可用（GUI 起不来，可改用 CLI 模式）")
        problems += 1

    # 3) 项目结构
    for sub in ("client", "server", "scripts"):
        p = root / sub
        log("[OK]" if p.is_dir() else "[!!]",
            f"目录：{sub}" + ("" if p.is_dir() else "  缺失"))
        problems += 0 if p.is_dir() else 1

    # 4) 权重（缺了就是 E3004，不阻塞启动但要提示）
    missing_w = [d for d in WEIGHT_DIRS if not (root / d).is_dir()]
    if missing_w:
        log("[!!]", f"E3004 权重目录缺失 {len(missing_w)}/{len(WEIGHT_DIRS)} 项")
        for d in missing_w[:3]:
            log("     ", f"缺失：{d}")
        problems += 1
    else:
        log("[OK]", f"权重目录齐全（{len(WEIGHT_DIRS)} 项）")

    # 5) 配置
    cfg, note = load_config(root)
    log("[OK]", f"配置：{note}")
    url = cfg.get("server", {}).get("url", "http://127.0.0.1:8000")

    # 6) 服务端连通性（没起不算失败）
    host_port = url.split("//")[-1].split("/")[0]
    host, _, port = host_port.partition(":")
    port = int(port or 8000)
    if server_healthy(url):
        log("[OK]", f"服务端：{url} 已就绪")
    elif port_in_use(host, port):
        log("[--]", f"服务端：{url} 端口在监听但 /healthz 未通过（可能仍在预热）")
    else:
        log("[--]", f"服务端：{url} 未启动（跑 `egomed.bat demo` 会自动拉起）")

    # 7) 结论
    if problems:
        log("[!!]", f"自检完成：发现 {problems} 项问题")
        return RC_ENV
    log("[OK]", "自检完成：全部通过")
    return RC_OK


def cmd_console(args) -> int:
    """打开现场控制台（GUI）——直接复用 client/live_console.py，不重写界面。"""
    root = find_project_root()
    target = root / "client" / "live_console.py"
    if not target.exists():
        log("[!!]", f"找不到控制台：{target}")
        return RC_ENV
    log("[--]", f"启动控制台：{target}")
    return subprocess.call([sys.executable, str(target)] + list(args.rest))


def cmd_calibrate(args) -> int:
    """机位校准（复用 client/calibrate.py）。"""
    root = find_project_root()
    target = root / "client" / "calibrate.py"
    if not target.exists():
        log("[!!]", f"找不到校准工具：{target}")
        return RC_ENV
    return subprocess.call([sys.executable, str(target)] + list(args.rest))


def cmd_diag(args) -> int:
    """生成诊断包（复用 client/make_diag_pack.py）。"""
    root = find_project_root()
    target = root / "client" / "make_diag_pack.py"
    if not target.exists():
        log("[!!]", f"找不到诊断工具：{target}")
        return RC_ENV
    log("[--]", "生成诊断包…")
    rc = subprocess.call([sys.executable, str(target)] + list(args.rest))
    # 增量 4：此处把 VERSION / CONTRACT_VERSION / 环境信息写进诊断包清单
    return rc


def cmd_not_implemented(name: str) -> int:
    log("[--]", f"子命令 `{name}` 尚未实现（骨架阶段）")
    log("     ", "计划见：软件制作/计划_软件化实施.md 的增量 2~4")
    return RC_NOTIMPL


def cmd_demo(args) -> int:
    """一键演示：起服务端（等健康）→ 开 GUI。增量 1 内实装。"""
    return cmd_not_implemented("demo")


def cmd_regress(args) -> int:
    """一键回归：接 acceptance_test / run_regression。增量 2 实装。"""
    return cmd_not_implemented("regress")


def cmd_soak(args) -> int:
    """长稳测试：接 run_soak。增量 2 实装。"""
    return cmd_not_implemented("soak")


# ---------------------------------------------------------------- 入口

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="egomed",
        description=f"EgoMed-Agent 统一启动器 v{VERSION}（契约 {CONTRACT_VERSION}）",
        epilog="返回码：0 成功 | 1 取消 | 2 环境问题 | 3 连接问题 | 4 验收失败 | 5 未实现",
    )
    p.add_argument("-v", "--version", action="version",
                   version=f"egomed {VERSION} (contract {CONTRACT_VERSION})")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("doctor", help="环境自检").set_defaults(func=cmd_doctor)

    for name, fn, help_text in (
        ("console", cmd_console, "打开现场控制台（GUI）"),
        ("calibrate", cmd_calibrate, "机位校准"),
        ("diag", cmd_diag, "生成诊断包"),
        ("demo", cmd_demo, "一键演示（起服务端 + GUI）"),
        ("regress", cmd_regress, "一键回归"),
        ("soak", cmd_soak, "长稳测试"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("rest", nargs=argparse.REMAINDER, help="透传给下游脚本的参数")
        sp.set_defaults(func=fn)

    return p


def main(argv=None) -> int:
    setup_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        # 无子命令 = 打开 GUI（最常用路径）
        args.rest = []
        return cmd_console(args)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log("[--]", "用户中断")
        return RC_CANCELLED


if __name__ == "__main__":
    raise SystemExit(main())
