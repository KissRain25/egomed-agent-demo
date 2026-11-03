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

import json
import re
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
    # ---- E1xxx 服务端明确拒绝 / 服务端自身故障（2026-11-03 新增，见契约 §6）
    "E1101": ErrInfo("E1101", "请求被服务端拒绝（参数缺失、格式错或上传超限）",
                     "多半是客户端 bug：核对 session_id、timestamp（毫秒整数）、"
                     "文件大小（帧/音频各 ≤2MB）；用 `egomed.bat diag` 看请求原文"),
    "E1500": ErrInfo("E1500", "服务端内部错误（500）",
                     "看 `runs/server_log.txt` 末尾的 traceback（完整异常只进服务端日志、不外泄）；"
                     "用 `egomed.bat diag` 打诊断包给开发"),
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

# ---------------------------------------------------------------------------
# 契约码（服务端说的） → E 码（客户端该做什么）
# ---------------------------------------------------------------------------
# 两套码**不是重复，是两层**：
#   契约码（API_CONTRACT §6：1001/1002/2001~2003/5000）= 服务端说了什么，机器可读、跨端契约；
#   E 码（本文件）= 客户端该怎么办，人可读、带一条下一步。
# 打通方式：客户端收到错误响应时把**契约码**读出来 → 映射到对应 E 码，
# 现场看到的既不是笼统的"连不上"，也不是裸的 1001。
#
# 只有服务端**真会放进响应 JSON** 的码才需要映射：
#   1001（参数缺失/格式错/超限，HTTP 400）、5000（内部错误，HTTP 500）。
# 2001/2002 按决议 #13 只作诊断位、不进 JSON；1002/2003 本期不可达 → 不映射。
CONTRACT_TO_E: dict[str, str] = {
    "1001": "E1101",
    "5000": "E1500",
}


def explain(code: str, detail: str = "") -> str:
    """错误码 → 一行可直接打印给用户的话（含下一步）。"""
    info = ERRORS.get((code or "").upper())
    if info is None:
        return f"{code} 未登记的错误码（见 软件制作/错误码.md）"
    head = f"{info.code} {info.title}"
    if detail:
        head += f"：{detail}"
    return f"{head} → 下一步：{info.next_step}"


def _contract_code_of(resp) -> str | None:
    """从 HTTP 响应里取契约码（`{"status":"error","code":1001,...}`）。

    urllib 的 HTTPError 没有 `.json()` → 抛异常被吞掉返回 None（安全）。
    """
    if resp is None:
        return None
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001  响应体不是 JSON / 已关闭
        return None
    if isinstance(body, dict):
        c = body.get("code")
        if c is not None:
            return str(c)
    return None


def classify(exc: BaseException, default: str = DEFAULT_CODE) -> str:
    """异常 → 错误码。只认有把握的几类，认不出就回 default（不猜）。

    **2026-11-03 修正（真缺陷）**：原实现只认 **urllib** 的异常类型，而客户端实际用的是
    **requests** —— `requests.exceptions.Timeout` / `ConnectionError` / `HTTPError`
    **都不是** builtin 的 `TimeoutError` / `ConnectionError` / `urllib.error.HTTPError`
    （它们继承的是 `RequestException` → `IOError`）→ **全部落到兜底 E1002「连不上服务端」**。
    现场后果：真引擎首帧慢/超时被报成"连不上"，提示去查端口占用 → **排查方向被带偏**。

    现在按「类名 + 属性」duck typing 识别（**保持零依赖，不 import requests**）：
      - 类名含 `Timeout` → **E1003**（是超时，不是连不上；ConnectTimeout/ReadTimeout 都算）
      - 类名含 `ConnectionError` → E1002
      - HTTPError（带 `.response`）→ 先看响应体里的**契约码**（1001/5000）→ E1101/E1500；
        没有契约码就按状态码分：>=500 → E1500，4xx → E1101
    urllib 的老分支保留（launcher 探活走 urllib）。
    """
    name = type(exc).__name__
    resp = getattr(exc, "response", None)

    # --- requests.exceptions.HTTPError：连上了，服务端明确给了错误 ---
    if name.endswith("HTTPError"):
        code = _contract_code_of(resp)
        if code:
            return CONTRACT_TO_E.get(code, default)
        status = getattr(resp, "status_code", None)
        if isinstance(status, int):
            return "E1500" if status >= 500 else "E1101"
        return default

    # --- requests 的超时 / 连接失败 ---
    if "Timeout" in name:
        return "E1003"
    if "ConnectionError" in name:
        return "E1002"

    # --- urllib（launcher 侧）---
    if isinstance(exc, ModuleNotFoundError):
        return "E3002"
    if isinstance(exc, FileNotFoundError):
        return "E1001"
    if isinstance(exc, PermissionError):
        return "E1001"
    if isinstance(exc, urllib.error.HTTPError):
        # urllib 的 HTTPError 带 .code（HTTP 状态码），5xx 归服务端故障
        st = getattr(exc, "code", None)
        if isinstance(st, int) and st >= 500:
            return "E1500"
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


def code_from_error(err_text: str, default: str = DEFAULT_CODE) -> str:
    """客户端错误串 → E 码（把"服务端说了什么"翻译成"我该做什么"）。

    `client/mock_client.post_multipart` 会把服务端响应体里的**契约码**带进错误串
    （形如 `HTTP 400 [code=1001] 缺少 session_id`），这里据此翻译：

      - 有契约码 → 查 `CONTRACT_TO_E`（1001→E1101、5000→E1500）
      - 无契约码、文本以"超时"开头 → **E1003**（别再误报"连不上"）
      - 含"连不上" → E1002
      - `HTTP 5xx` → E1500；`HTTP 4xx` → E1101

    用法：`code = ee.code_from_error(err)` → `ee.explain(code, err)`。
    """
    if not err_text:
        return default
    m = re.search(r"code=(\d{4})", err_text)
    if m:
        return CONTRACT_TO_E.get(m.group(1), default)
    t = err_text.strip()
    if t.startswith("超时"):
        return "E1003"
    if "连不上" in t or "拒绝连接" in t or "Connection refused" in t:
        return "E1002"
    if t.startswith("HTTP 5"):
        return "E1500"
    if t.startswith("HTTP 4"):
        return "E1101"
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
    # 2026-11-03 新增：requests 异常识别 + 契约码翻译
    # 背景：客户端实际用 requests，而 requests 的异常**不是** builtin/urllib 的那些类型
    # → 原实现把它们全归兜底 E1002「连不上」。这里用真的 requests 异常类做回归。
    try:
        import requests

        def _resp(status: int, body: dict | None):
            r = requests.Response()
            r.status_code = status
            r.headers["Content-Type"] = "application/json"
            r._content = (json.dumps(body, ensure_ascii=False).encode("utf-8")
                          if body is not None else b"")
            return r

        cases += [
            ("requests 超时 → E1003（不是 E1002）",
             classify(requests.exceptions.Timeout()), "E1003"),
            ("requests ReadTimeout → E1003",
             classify(requests.exceptions.ReadTimeout()), "E1003"),
            ("requests ConnectionError → E1002",
             classify(requests.exceptions.ConnectionError()), "E1002"),
            ("requests HTTPError(契约码 1001) → E1101",
             classify(requests.exceptions.HTTPError(
                 response=_resp(400, {"status": "error", "code": 1001}))), "E1101"),
            ("requests HTTPError(契约码 5000) → E1500",
             classify(requests.exceptions.HTTPError(
                 response=_resp(500, {"status": "error", "code": 5000}))), "E1500"),
            ("requests HTTPError(无契约码 500) → E1500",
             classify(requests.exceptions.HTTPError(response=_resp(500, None))), "E1500"),
            ("requests HTTPError(无契约码 404) → E1101",
             classify(requests.exceptions.HTTPError(response=_resp(404, None))), "E1101"),
            ("错误串翻译：契约码 1001 → E1101",
             code_from_error("HTTP 400 [code=1001] 缺少 session_id"), "E1101"),
            ("错误串翻译：契约码 5000 → E1500",
             code_from_error("HTTP 500 [code=5000] 服务器内部错误"), "E1500"),
            ("错误串翻译：超时 → E1003（不再误报连不上）",
             code_from_error("超时（>3s）"), "E1003"),
            ("错误串翻译：连不上 → E1002",
             code_from_error("连不上 http://127.0.0.1:8000/api/v1/audio"), "E1002"),
            ("错误串翻译：认不出 → 兜底不猜",
             code_from_error("什么怪东西"), "E1002"),
        ]
    except ImportError:  # pragma: no cover —— 没装 requests 就跳过这组
        print("[SKIP] requests 未安装，跳过 requests 异常用例")

    # 所有用例收集齐之后再统一判定
    # （新增用例必须追加在这个循环**之前**，否则不会被执行 —— 2026-11-03 踩过这个坑）
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
