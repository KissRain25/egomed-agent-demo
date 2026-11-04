#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 launcher 打包成 Windows 可执行程序（2026-11-04 新增，N53 交付项）。

为什么要有这个：`egomed.bat` 只是"找 python.exe + 转发"的入口壳，
目标机器没装 Python / 没建 egomed 环境就**连界面都打不开**。
打成 exe 后，交付方**双击就能打开控制台界面、跑自检、生成诊断包**。

    python scripts/build_exe.py                # 默认：onedir + GUI（双击即用）
    python scripts/build_exe.py --onefile      # 单文件 exe（启动慢 2~5s，解压到临时目录）
    python scripts/build_exe.py --console      # 带控制台窗口（给 doctor/check 看输出）
    python scripts/build_exe.py --clean        # 先清 build/dist 再打

产物：
    dist/EgoMed/EgoMed.exe            （--onedir，默认，推荐）
    dist/EgoMed.zip                   （分发用压缩包，可选）

**边界说明（别误解）**：exe 里**不含** torch / ultralytics / SAM2 / faster-whisper
（合计 >2.5 GB，塞不进单文件，也不该塞）—— 这些是**服务端推理依赖**，
仍需目标机器有 egomed 环境。exe 的作用是"**入口与体验层零依赖**"：
界面、自检、诊断包、配置、错误码全在 exe 里；真跑分割仍走服务端环境。
exe 内置的 doctor 会明确告诉你缺什么、怎么补。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NAME = "EgoMed"

# 这些重依赖**故意不打包**：服务端推理用，体积巨大且依赖 CUDA DLL。
# 不显式排除的话，PyInstaller 的静态分析会把它们拖进来（打包体积涨到 2GB+）。
EXCLUDE = [
    "torch", "torchvision", "ultralytics", "sam2", "faster_whisper", "whisper",
    "scipy", "pandas", "matplotlib", "IPython", "notebook", "pytest",
    "tensorboard", "onnx", "triton", "nvidia",
]

# 需要显式声明的隐藏导入（动态 import，静态分析扫不到）
HIDDEN = [
    "tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.filedialog",
    "cv2", "PIL", "PIL.Image", "PIL.ImageTk",
    "egomed_config", "egomed_errors", "mock_client", "live_client", "live_console",
]


def build(args) -> int:
    name = args.name
    pyinstaller = [sys.executable, "-m", "PyInstaller"]
    if not args.no_clean:
        for d in (PROJECT_ROOT / "build" / "egomed_exe",):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        if args.clean and (PROJECT_ROOT / "dist" / args.name).exists():
            shutil.rmtree(PROJECT_ROOT / "dist" / args.name, ignore_errors=True)

    cmd = pyinstaller + [
        "--noconfirm",
        "--name", name,
        "--distpath", str(PROJECT_ROOT / "dist"),
        "--workpath", str(PROJECT_ROOT / "build" / "egomed_exe"),
        "--specpath", str(PROJECT_ROOT / "build"),
        # 源码根与 client/ 都要在搜索路径里（launcher 用 import 拉 client/*）
        "--paths", str(PROJECT_ROOT),
        "--paths", str(PROJECT_ROOT / "client"),
    ]
    if args.onedir:
        cmd.append("--onedir")
    else:
        cmd.append("--onefile")   # 默认单文件：能单独拷走、双击即用
    if not args.console:
        cmd.append("--windowed")
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]
    for mod in HIDDEN:
        cmd += ["--hidden-import", mod]
    # 数据文件：版本号、配置、文档（界面与诊断包要读）
    for src, dst in (("VERSION", "."), ("configs", "configs"), ("docs", "docs")):
        p = PROJECT_ROOT / src
        if p.exists():
            cmd += ["--add-data", f"{p}{';' if sys.platform == 'win32' else ':'}{dst}"]

    cmd.append(str(PROJECT_ROOT / "launcher.py"))

    # ---- Tcl/Tk 运行时（GUI 的命门）------------------------------------
    # 2026-11-04 实测：conda 环境的 Tcl/Tk 在 `Library/lib/tcl8.6`、`Library/lib/tk8.6`，
    # 而 PyInstaller 的 hook 是按**标准布局**（`<prefix>/tcl/tcl8.6`）找的 → 找不到就**静默漏掉**
    # → 打包出来 exe 里 `import tkinter` 直接失败（doctor 报"tkinter 不可用"）。
    # 所以这里按 conda 布局手动补进 `_tcl_data/`（PyInstaller 认这个目录名）。
    sep = ";" if sys.platform == "win32" else ":"
    prefix = Path(sys.prefix)
    tcl_roots = [
        (prefix / "Library" / "lib" / "tcl8.6", "_tcl_data/tcl8.6"),
        (prefix / "Library" / "lib" / "tk8.6", "_tcl_data/tk8.6"),
        (prefix / "tcl" / "tcl8.6", "_tcl_data/tcl8.6"),
        (prefix / "tcl" / "tk8.6", "_tcl_data/tk8.6"),
    ]
    for src, dst in tcl_roots:
        if src.is_dir():
            cmd += ["--add-data", f"{src}{sep}{dst}"]
            print(f"[build] Tcl/Tk 数据：{src} -> {dst}")
    cmd += ["--collect-all", "tkinter"]

    # ---- conda 的 DLL 搜索路径（另一个命门）------------------------------
    # 2026-11-04 实测：PyInstaller 分析 `_tkinter.pyd` 的依赖时只扫 conda 的 `DLLs`，
    # 而 `tcl86t.dll`/`tk86t.dll` 实际在 `Library\bin` → 报
    # "Library not found: could not resolve 'tcl86t.dll'" → exe 里 `import tkinter` 直接失败。
    # 修法：调 PyInstaller **之前**把 `Library\bin` 临时加进 PATH（它的依赖分析用 PATH 找 DLL），
    # 再显式 --add-binary 双保险。
    lib_bin = Path(sys.prefix) / "Library" / "bin"
    if lib_bin.is_dir():
        os.environ["PATH"] = str(lib_bin) + os.pathsep + os.environ.get("PATH", "")
        for dll in ("tcl86t.dll", "tk86t.dll"):
            src = lib_bin / dll
            if src.is_file():
                cmd += ["--add-binary", f"{src}{sep}."]
                print(f"[build] DLL：{src.name} -> .")

    print("[build] 开始打包（首次约 1~3 分钟，cv2/PIL 较大）…")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    print(f"[build] PyInstaller 退出码 {proc.returncode}，耗时 {time.time() - t0:.0f}s")
    if proc.returncode != 0:
        return proc.returncode

    out = (PROJECT_ROOT / "dist" / name / f"{name}.exe") if args.onedir \
        else (PROJECT_ROOT / "dist" / f"{name}.exe")
    if not out.exists():
        print(f"[build][!!] 没找到产物：{out}")
        return 2

    size_mb = out.stat().st_size / 1024 / 1024
    total = sum(f.stat().st_size for f in out.parent.rglob("*") if f.is_file())
    print(f"[build][OK] {out}")
    print(f"        exe 本体 {size_mb:.1f} MB ｜ 整目录 {total / 1024 / 1024:.1f} MB")

    if args.zip:
        import zipfile
        zp = PROJECT_ROOT / "dist" / f"{name}_win_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted((PROJECT_ROOT / "dist" / name).rglob("*")):
                if f.is_file():
                    z.write(f, Path(name) / f.relative_to(PROJECT_ROOT / "dist" / name))
        print(f"[build][OK] 分发包：{zp}（{zp.stat().st_size / 1024 / 1024:.1f} MB）")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="打包 egomed.exe（PyInstaller）")
    ap.add_argument("--onefile", action="store_true",
                    help="打成单个 exe（默认行为：单文件，双击即用；启动需解压 2~5s）")
    ap.add_argument("--onedir", action="store_true",
                    help="打成目录（启动快，但要分发整个目录）")
    ap.add_argument("--console", action="store_true", help="带控制台窗口（默认 GUI 模式）")
    ap.add_argument("--zip", action="store_true", help="额外打一个分发用 zip")
    ap.add_argument("--clean", action="store_true", help="打包前先删掉旧的 dist/EgoMed")
    ap.add_argument("--no-clean", action="store_true", help="保留 PyInstaller 缓存（增量打包更快）")
    ap.add_argument("--name", default=NAME, help="产物名（默认 EgoMed；console 调试版可用 EgoMedCLI）")
    args = ap.parse_args(argv)
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[build][!!] 没装 PyInstaller。先执行：")
        print("        python -m pip install pyinstaller -i https://pypi.org/simple")
        return 2
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
