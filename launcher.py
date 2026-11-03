#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EgoMed-Agent 统一启动器 v0.2（已落在仓库根目录）

把原来散在 11 个 .bat 里的"解释器解析 + 参数拼装"收敛到一处：
根目录 `egomed.bat` 只做解释器兜底 + chcp + 退出码透传，全部子命令在本文件实现。
旧 bat（启动模拟眼镜 / 现场控制台 / 环境自检）已降级为**转发壳**，行为不变。

已实装：doctor / console / demo / sim / check / calibrate / diag / regress / soak / warmup
版本号唯一来源：根目录 `VERSION` 文件（本文件的 VERSION 常量由它读出，缺文件才回退）
计划见 `软件制作/计划_软件化实施.md`。

用法
----
    python launcher.py                 # 打开 GUI（默认）
    python launcher.py doctor          # 环境自检（返回码 0 通过 / 2 有问题）
    python launcher.py demo            # 一键演示：起服务端 → 等预热 → 开 GUI
    python launcher.py demo --dry-run  # 只打印流程与服务端状态，不起进程
    python launcher.py console         # 直接开 GUI
    python launcher.py calibrate       # 机位校准
    python launcher.py diag            # 生成诊断包
    python launcher.py sim             # 兼容旧「启动模拟眼镜」：服务端 + 命令行客户端
    python launcher.py check           # 兼容旧「环境自检」：scripts/check_env.py 详细检查
    python launcher.py regress         # 一键回归（→ client/run_regression.py）
    python launcher.py soak -m 60      # 长稳测试（→ client/run_soak.py）
    python launcher.py warmup          # 演示前预热：吃掉 ASR 冷启动（--keep-server 保留服务端）

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
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- 常量

FALLBACK_VERSION = "0.2.3"      # VERSION 文件缺失时的兜底，正常不该走到
# 2026-11-03：随 v1.0.6（决议 #16：candidates 语义统一为「候选模态名」）升到 v1.0.6。
# 此前变更记录已写 v1.0.4、但常量没跟，导致诊断包 versions.json 报旧契约版本 —— 这条要盯住。
CONTRACT_VERSION = "v1.0.6"


def read_version() -> str:
    """软件版本的**唯一来源**是根目录 `VERSION` 文件。

    这里为什么不用常量硬编码：GUI「关于」、`--version`、诊断包清单三处都要显示版本，
    硬编码必然漂移。读文件失败就回退，绝不因版本号读不到而启动失败。

    **用 utf-8-sig 读**：Windows 的记事本 / PowerShell `Set-Content -Encoding UTF8`
    会给文件加 BOM（实测 2026-11-04：用 PowerShell 改版本号后 `launcher.py --version`
    输出成 `egomed ﻿0.2.3`，版本号里混进一个 U+FEFF，连便携包文件名都带上了它）。
    `utf-8-sig` 兼容两种（有 BOM 无 BOM都能读），顺带把 BOM 剥掉。
    """
    here = Path(__file__).resolve().parent      # 不用 STAGE_DIR：它在下面才定义
    for cand in (here / "VERSION", here.parent / "VERSION"):
        try:
            if cand.is_file():
                first = cand.read_text(encoding="utf-8-sig").strip().splitlines()
                if first and first[0].strip():
                    return first[0].strip().lstrip("\ufeff").strip()
        except Exception:  # noqa: BLE001
            continue
    return FALLBACK_VERSION


VERSION = read_version()

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
    """读配置：命令行 > 环境变量(EGOMED_*) > `config.toml` > 内置默认值。

    真正的实现在 `egomed_config.py` —— GUI 也要读同一份配置并回写 `[ui]`，
    两处各写一套解析/合并必然漂移，所以这里只做委托。
    文件缺失或损坏都不抛异常（坏了会备份成 `.bak-<时间戳>` 再用默认值）。
    """
    try:
        import egomed_config
        cfg, notes = egomed_config.load(root)
        return cfg, "；".join(notes)
    except Exception as exc:  # noqa: BLE001
        return default_config(), f"配置模块不可用（{exc}），使用内置默认值"


def default_config() -> dict:
    """兜底默认值：只在 import 不到 egomed_config 时用（正常不该走到）。"""
    try:
        import egomed_config
        return egomed_config.default_config()
    except Exception:  # noqa: BLE001
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

    # 5) 配置（读不到/坏了会用默认值兜底，属于"提示"而不是"失败"）
    cfg, note = load_config(root)
    tag = "[--]" if any(k in note for k in ("失败", "未找到", "忽略", "不可用")) else "[OK]"
    log(tag, f"配置：{note}")
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


def forward_script(rel_path: str, args, cwd: Path | None = None) -> int:
    """把子命令原样转发给既有脚本：只接管"用哪个解释器、在哪个目录跑"。

    兼容壳专用——旧 bat 的行为必须一字不改，所以不做任何参数改写。
    """
    root = find_project_root()
    target = root / rel_path
    if not target.exists():
        log("[!!]", f"E3002 找不到脚本：{target}")
        return RC_ENV
    return subprocess.call([sys.executable, str(target)] + list(args.rest),
                           cwd=str(cwd or root))


def cmd_sim(args) -> int:
    """兼容旧「启动模拟眼镜.bat」：起服务端 + 命令行客户端（不开 GUI）。"""
    return forward_script("client/launch_sim_glasses.py", args)


def cmd_check(args) -> int:
    """兼容旧「环境自检.bat」：scripts/check_env.py 的十余项详细检查。"""
    return forward_script("scripts/check_env.py", args)


def load_launch_helper(root: Path):
    """复用 client/launch_sim_glasses.py 的启停与预热等待，不重复实现。

    该模块的 main() 在 __main__ 保护里，import 无副作用。
    """
    client_dir = str(root / "client")
    if client_dir not in sys.path:
        sys.path.insert(0, client_dir)
    import launch_sim_glasses as lsg  # noqa: E402
    return lsg


def cmd_demo(args) -> int:
    """一键演示：起服务端 → 等预热就绪 → 开现场控制台（GUI）。

    就绪判据（与 launch_sim_glasses 一致，非另创）：
      ① 服务端日志出现 "真引擎启动" / "Uvicorn running"
      ② 或 /healthz 返回 200 且 real_engine=真
    上限 240s（含 ASR medium 加载，实测约 78s）。

    --dry-run：只打印将要执行的步骤与服务端当前状态，不启动任何进程。
    """
    root = find_project_root()
    cfg, _ = load_config(root)
    url = cfg.get("server", {}).get("url", "http://127.0.0.1:8000")
    dry = "--dry-run" in args.rest

    console = root / "client" / "live_console.py"
    helper = root / "client" / "launch_sim_glasses.py"
    if not console.exists() or not helper.exists():
        log("[!!]", f"E3002 缺少演示所需脚本（console={console.exists()}, helper={helper.exists()}）")
        return RC_ENV

    if dry:
        log("[--]", "演示流程预演（不启动任何进程）：")
        log("     ", "1) 环境检查（同 doctor）")
        log("     ", "2) 起服务端：server/api.py --real")
        log("     ", "3) 等就绪：日志标记 + /healthz.real_engine，上限 240s")
        log("     ", "4) 打开现场控制台（GUI）")
        log("     ", "5) 控制台退出后关闭服务端，清理进程")
        log("[--]", f"服务端当前状态：{'已在运行' if server_healthy(url) else '未启动'}")
        return RC_OK

    if server_healthy(url):
        log("[--]", f"服务端已在运行（{url}），跳过启动，直接开控制台")
        return subprocess.call([sys.executable, str(console)], cwd=str(root))

    try:
        lsg = load_launch_helper(root)
    except Exception as exc:
        log("[!!]", f"E3002 载入一键演示脚本失败：{exc}")
        return RC_ENV

    proc = None
    try:
        proc = lsg.start_server()
        if not lsg.wait_warmup(proc):
            # 增量 4：区分"没人听(E1002)"与"有人在听但没就绪(E3003)"，两者的处置完全不同
            try:
                import egomed_errors
                host_port = url.split("//")[-1].split("/")[0]
                host, _, port = host_port.partition(":")
                code = egomed_errors.pick_server_code(host, int(port or 8000), healthy=False)
                log("[!!]", egomed_errors.explain(code, "服务端未能就绪（启动失败或预热超时）"))
            except Exception:  # noqa: BLE001
                log("[!!]", "E1002 服务端未能就绪（启动失败或预热超时）")
            log("     ", "服务端日志：runs/launch/")
            return RC_CONN
        log("[OK]", "服务端已就绪，打开现场控制台（关闭窗口即结束并关闭服务端）")
        return subprocess.call([sys.executable, str(console)], cwd=str(root))
    except KeyboardInterrupt:
        log("[--]", "用户中断")
        return RC_CANCELLED
    finally:
        if proc is not None:
            lsg.stop_server(proc)


def cmd_regress(args) -> int:
    """一键回归（N22）：转发 `client/run_regression.py`，由它自己起/关服务端。

    返回码沿用该脚本的既有语义（0 通过 / 非 0 未通过），**不重映射** ——
    旧「回归.bat」也在看这个码，改了会破坏现场习惯。
    """
    return forward_script("client/run_regression.py", args)


def cmd_soak(args) -> int:
    """长稳测试（N24）：转发 `client/run_soak.py`（`--minutes` 缺省 60）。"""
    return forward_script("client/run_soak.py", args)


def cmd_warmup(args) -> int:
    """演示前一键预热：把 **ASR 冷启动** 提前吃掉。

    为什么需要它
    ------------
    服务端 `--real` 的"预热"只把 ASR 权重读进显存（`asr_medical.warmup()` 里只有
    `_get_model()`），**一次推理都不跑**。所以机器上第一次真实语音请求才触发
    CUDA kernel / cuDNN 初始化，2026-11-02 实测：#1 >180s（被客户端 15s 掐断）
    → #2 75s → #3 才 2s。而现场第一句话就会撞上这个 —— 用户看到的是
    "连不上服务端"，不是"正在预热"。

    这条命令做的事：起服务端 → 等就绪 → 发一次样例语音（长超时）→ 报耗时。
    默认跑完关掉服务端；`--keep-server` 留着直接演示。

    用法
    ----
        egomed.bat warmup                     # 预热完关服务端
        egomed.bat warmup --keep-server       # 预热完留着，接着 demo/console
        egomed.bat warmup --timeout 420       # 首次语音等更久（老机器/首次）
    """
    root = find_project_root()
    cfg, _ = load_config(root)
    url = cfg.get("server", {}).get("url", "http://127.0.0.1:8000")

    rest = list(args.rest)
    if "-h" in rest or "--help" in rest:
        # 本子命令是透传壳（--help 会落到 rest），这里自己答，别真去起服务端
        log("[--]", "用法：egomed.bat warmup [--keep-server] [--timeout 秒]")
        log("     ", "  起服务端 → 等就绪 → 发一次样例语音吃掉 ASR 冷启动 → 关闭（或保留）")
        log("     ", "  --keep-server  预热完不关服务端，接着 console / demo 用")
        log("     ", "  --timeout N    首次语音最长等待（默认 300s；老机器可给 420）")
        return RC_OK
    keep = "--keep-server" in rest
    timeout = 300.0
    if "--timeout" in rest:
        i = rest.index("--timeout")
        if i + 1 < len(rest):
            try:
                timeout = float(rest[i + 1])
            except ValueError:
                log("[!!]", f"--timeout 需要数字，收到 {rest[i + 1]!r}，改用 {timeout:.0f}s")

    client_dir = str(root / "client")
    if client_dir not in sys.path:
        sys.path.insert(0, client_dir)
    try:
        import mock_client as mc
    except Exception as exc:  # noqa: BLE001
        log("[!!]", f"E3002 载入客户端工具失败：{exc}")
        return RC_ENV

    proc = None
    started_by_us = False
    try:
        if server_healthy(url):
            log("[--]", f"服务端已在运行（{url}），直接预热语音通道")
        else:
            try:
                lsg = load_launch_helper(root)
            except Exception as exc:  # noqa: BLE001
                log("[!!]", f"E3002 载入一键演示脚本失败：{exc}")
                return RC_ENV
            log("[--]", "启动服务端并等待就绪（真引擎预热约 60~90s）…")
            proc = lsg.start_server()
            started_by_us = True
            if not lsg.wait_warmup(proc):
                log("[!!]", "E1002 服务端未能就绪，预热中止（日志见 runs/launch/）")
                return RC_CONN
            log("[OK]", "服务端已就绪")

        sid = mc.make_session_id("warmup")
        log("[--]", f"发送样例语音预热 ASR（首次最长等 {timeout:.0f}s，这一步就是全部意义）…")
        t0 = time.time()
        # wait_ready_s 跟 timeout 对齐：服务端在 ASR 未热好时会回 503 asr_warming，
        # 客户端按 10s 重试；给 120s 默认值会在 ASR 初始化较慢时提前放弃，
        # 而 first_timeout 又帮不上忙（请求根本没发出去，是被立刻 503 挡回来的）。
        target, used_wav, _payload, terr = mc.set_target(
            url, sid, None, 15.0, verbose=True,
            first_timeout=timeout, wait_ready_s=timeout)
        spent = time.time() - t0
        if target:
            log("[OK]", f"ASR 通道已预热：{target!r}（用 {used_wav}，耗时 {spent:.1f}s）")
            log("     ", "之后现场每句话都是 1~2 秒量级，不会再撞冷启动")
        else:
            log("[!!]", f"E1003 预热未拿到目标（err={terr}，耗时 {spent:.1f}s）")
            log("     ", "服务端可用但语音通道没热起来，演示前建议重试一次")
            return RC_CONN

        if keep:
            log("[OK]", f"服务端保留运行（{url}），可直接 `egomed.bat console` 或 `egomed.bat demo`")
            started_by_us = False      # 交给用户，别在 finally 里关
        return RC_OK
    except KeyboardInterrupt:
        log("[--]", "用户中断")
        return RC_CANCELLED
    finally:
        if proc is not None and started_by_us:
            try:
                load_launch_helper(root).stop_server(proc)
                log("[--]", "服务端已关闭（加 --keep-server 可保留）")
            except Exception:  # noqa: BLE001
                pass


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

    # 第 4 列为 True 的是"兼容壳"子命令：连 -h 也要原样透传给下游脚本，
    # 否则旧 bat 的 --help 行为会变（现场排障时容易踩）。
    for name, fn, help_text, passthru_all in (
        ("doctor", cmd_doctor, "环境自检（精简版）", False),
        ("console", cmd_console, "打开现场控制台（GUI，默认）", False),
        ("demo", cmd_demo, "一键演示（起服务端 + GUI）", False),
        ("sim", cmd_sim, "兼容壳：服务端 + 命令行客户端（旧「启动模拟眼镜」）", True),
        ("check", cmd_check, "兼容壳：详细环境检查（旧「环境自检」）", True),
        ("calibrate", cmd_calibrate, "机位校准", False),
        ("diag", cmd_diag, "生成诊断包", False),
        ("regress", cmd_regress, "一键回归（mock + real 两套验收）", True),
        ("soak", cmd_soak, "长稳测试（默认 60 分钟）", True),
        ("warmup", cmd_warmup, "演示前预热（吃掉 ASR 冷启动，可选 --keep-server）", True),
    ):
        sp = sub.add_parser(name, help=help_text, add_help=not passthru_all)
        sp.add_argument("rest", nargs=argparse.REMAINDER, help="透传给下游脚本的参数")
        sp.set_defaults(func=fn)

    return p


def _split_subcommand(raw: list[str]) -> tuple[str | None, list[str]]:
    """切出 (子命令, 子命令之后的**原始**参数)，保持原顺序不动。

    为什么不用 argparse 的 `nargs=REMAINDER` 取透传参数：子解析器不认识 `--suite`
    这类下游选项时，argparse 会把它丢进 unknown，同时把 `mock` 吃进 REMAINDER，
    两者拼回去顺序就变成 `mock --suite`，下游直接报 "--suite: expected one argument"。
    （2026-11-02 实测踩到；本文件只接受 -v/--version 两个前置选项，故此处可安全切分）
    """
    for i, tok in enumerate(raw):
        if tok.startswith("-"):
            continue
        return tok, raw[i + 1:]
    return None, []


def main(argv=None) -> int:
    setup_stdout()
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # 用 parse_known_args 只为拿到 args.cmd / args.func；透传参数一律取原始 argv 尾部
    args, _extras = parser.parse_known_args(raw)
    sub, tail = _split_subcommand(raw)
    args.rest = tail if sub else []

    if not getattr(args, "func", None):
        # 无子命令 = 打开 GUI（最常用路径）
        return cmd_console(args)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log("[--]", "用户中断")
        return RC_CANCELLED


if __name__ == "__main__":
    raise SystemExit(main())
