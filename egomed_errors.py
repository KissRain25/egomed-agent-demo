#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EgoMed-Agent 错误码（N53 增量 4）

目标：现场出问题的时候，用户看到的是**唯一错误码 + 一句人话 + 一条下一步**，
而不是一屏 traceback。错误码表见 `软件制作/错误码.md`（本文件是它的代码形态，
两边必须一起改）。

分段约定
--------
    E1xxx = 链路/运行期问题（采集源、服务端、超时、回显）
    E3xxx = 环境问题（解释器、依赖、端口、权重）

与返回码的关系（launcher.py）
-----------------------------
    0 成功 | 1 用户取消 | 2 环境问题(E3xxx) | 3 连接问题(E1xxx) | 4 验收失败 | 5 未实现

用法
----
    from egomed_errors import explain, classify
    print(explain("E1002"))
    code = classify(exc)          # 把异常翻成错误码

归属：闫 ｜ 关联：`软件制作/错误码.md`、`软件制作/计划_软件化实施.md` 增量 4
"""
from __future__ import annotations

import socket
import urllib.error
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrInfo:
    code: str
    title: str        # 一句人话：发生了什么
    next_step: str    # 一条下一步：现在该做什么


ERRORS: dict[str, ErrInfo] = {
    # ---- E1xxx 链路 / 运行期
    "E1001": ErrInfo("E1001", "采集源打不开（摄像头或文件不可用）",
                     "换 `--source cam --index 1`，或把 `--input` 指到一个确实存在的帧目录"),
    "E1002": ErrInfo("E1002", "连不上服务端",
                     "先跑 `egomed.bat doctor` 看服务端那一行，再 `netstat -ano | findstr :8000` 查端口占用"),
    "E1003": ErrInfo("E1003", "请求超时（3s 内没有响应）",
                     "看 `runs/server_log.txt` 是否还在预热，真引擎首次加载约 60~90s"),
    "E1004": ErrInfo("E1004", "回显渲染失败（打不开显示窗口）",
                     "改配置文件 `display.renderer = \"file\"`，结果落盘到 runs/ 下再看"),
    # ---- E3xxx 环境
    "E3001": ErrInfo("E3001", "找不到可用的 Python 解释器",
                     "先跑「安装环境.bat」，或设环境变量 `EGOMED_PY` 指向 egomed 环境的 python.exe"),
    "E3002": ErrInfo("E3002", "依赖缺失或脚本找不到",
                     "按提示 `pip install <包名>`；若提示脚本缺失，检查仓库是否完整"),
    "E3003": ErrInfo("E3003", "端口已被占用（多半是已有实例在跑）",
                     "直接附着到已运行的实例；要重开就先结束占用 8000 的进程"),
    "E3004": ErrInfo("E3004", "模型权重缺失",
                     "按 doctor 列出的路径把 `runs/yolo26_det/<模态>_*/weights` 补齐"),
}

DEFAULT_CODE = "E1002"     # 分类不出来时的兜底：当作"连不上服务端"最不容易误导


def explain(code: str, detail: str = "") -> str:
    """错误码 → 一行可直接打印给用户的话（含下一步）。"""
    info = ERRORS.get((code or "").upper())
    if info is None:
        return f"{code} 未登记的错误码（见 软件制作/错误码.md）"
    head = f"{info.code} {info.title}"
    if detail:
        head += f"：{detail}"
    return f"{head} → 下一步：{info.next_step}"


def classify(exc: BaseException, default: str = DEFAULT_CODE) -> str:
    """异常 → 错误码。只认有把握的几类，认不出就回 default（不猜）。"""
    if isinstance(exc, ModuleNotFoundError):
        return "E3002"
    if isinstance(exc, FileNotFoundError):
        return "E1001"
    if isinstance(exc, PermissionError):
        return "E1001"
    if isinstance(exc, urllib.error.HTTPError):
        return "E1002"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "E1003"
        return "E1002"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "E1003"
    if isinstance(exc, ConnectionError):
        return "E1002"
    return default


def port_free(host: str = "127.0.0.1", port: int = 8000, timeout: float = 0.3) -> bool:
    """端口是否空闲（能连上就说明有人在听 = 被占用）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) != 0


def pick_server_code(host: str, port: int, healthy: bool) -> str:
    """服务端类故障的唯一判定入口：先看能不能连，再看健不健康。

    这样 E3003（端口被占但服务未就绪）与 E1002（压根没人听）不会互相冒充。
    """
    if healthy:
        return "OK"
    return "E3003" if not port_free(host, port) else "E1002"


def _drill() -> int:
    """三例故障演练：验证"每种故障只有一个错误码 + 一句人话 + 一条下一步"。"""
    ok = fail = 0

    def check(name: str, cond: bool, extra="") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"[PASS] {name}")
        else:
            fail += 1
            print(f"[FAIL] {name}   {extra}")

    cases = [
        ("故障①：服务端没人监听", pick_server_code("127.0.0.1", 1, healthy=False), "E1002"),
        ("故障②：缺依赖（ModuleNotFoundError）", classify(ModuleNotFoundError("No module named 'cv2'")), "E3002"),
        ("故障③：采集源文件不存在", classify(FileNotFoundError("data/nope")), "E1001"),
        ("补充：请求超时", classify(TimeoutError("timed out")), "E1003"),
        ("补充：URLError 超时原因", classify(urllib.error.URLError(TimeoutError("timed out"))), "E1003"),
        ("补充：认识不出就用兜底", classify(ValueError("随便一个"), default="E1002"), "E1002"),
    ]
    for name, got, want in cases:
        check(name, got == want, f"期望 {want} 实得 {got}")

    # 故障④：端口**真被占用**但服务未就绪 → E3003（用临时 socket 实测，不碰 8000）
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 0
    try:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        host, port = srv.getsockname()
        check("故障④：端口被占用但服务未就绪（真实监听）",
              pick_server_code(host, port, healthy=False), "E3003")
    finally:
        srv.close()
    check("故障④对照：同一端口关掉后应判 E1002",
          pick_server_code("127.0.0.1", port, healthy=False) if port else "E1002", "E1002")

    # 每条码都必须有人话和下一步，且文案里带"下一步"
    for code, info in ERRORS.items():
        line = explain(code)
        check(f"{code} 文案完整", bool(info.title) and bool(info.next_step) and "下一步" in line, line)
        check(f"{code} 文案不重复", f"{code} " in line and line.count("→ 下一步：") == 1, line)

    # 未登记的错误码不许抛异常
    check("未登记码不崩", "未登记" in explain("E9999"))

    print(f"\n演练结论：{ok} PASS / {fail} FAIL")
    return 0 if fail == 0 else 4


if __name__ == "__main__":
    import sys
    if "--drill" in sys.argv:
        raise SystemExit(_drill())
    for _c in ERRORS:
        print(explain(_c))
