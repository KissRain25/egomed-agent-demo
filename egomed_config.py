#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EgoMed-Agent 配置读写（N53 增量 3）

四层优先级（高 → 低）
--------------------
    命令行参数  >  环境变量（EGOMED_*）  >  config.toml  >  内置默认值

为什么单独抽一个模块
--------------------
`launcher.py` 要读（服务端地址、fps），GUI 要**读+写**（记住上次选择）。
两边各写一份"解析 + 合并 + 回写"必然漂移，所以收敛到这里，谁都不许自己再解析一遍。

设计取舍
--------
- **注释用 SCHEMA 生成**：`config.toml` 里的中文注释是给人看的，保存时必须还在。
  与其做"改一行保住其它注释"的复杂补丁，不如把"默认值 + 注释"写成 SCHEMA，
  保存时整份重新生成 —— 结果可预期，也不会被手工编辑搞乱。
- **不用 tomli_w**：写 TOML 只用到 str/int/float/bool 与一层 table，
  自己拼 30 行就够，少一个依赖就少一个装不上的理由。
- **配置坏了不许拦住启动**：备份改名成 `.bak-<时间戳>` + 用默认值继续 + 留一条 note。

归属：闫 ｜ 关联：`软件制作/计划_软件化实施.md` 增量 3
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

CONFIG_NAME = "config.toml"

# 每段的每个键：(默认值, 注释)。顺序即写文件时的顺序。
SCHEMA: dict[str, dict[str, tuple]] = {
    "server": {
        "url": ("http://127.0.0.1:8000", "服务端地址（A1/A2 端点均挂在此前缀下）"),
        "timeout_s": (3.0, "单次请求超时（秒）；实测单帧 200~500ms，留 3s 余量"),
    },
    "capture": {
        "source": ("video", "采集源：video（帧目录/视频）| cam（本机摄像头）| url（手机 IP Webcam）| glasses（Rokid）"),
        "fps": (5.0, "发给服务端的帧率（不是源帧率）"),
        "max_side": (1280, "压缩规格（契约）：长边上限"),
        "jpeg_quality": (80, "压缩规格（契约）：JPEG 质量"),
        "max_bytes": (307200, "压缩规格（契约）：单帧字节上限（300KB）"),
    },
    "display": {
        "renderer": ("window", "回显方式：window | file | glasses-mono | null"),
    },
    "logging": {
        "level": ("info", "日志级别：debug | info | warn | error"),
        "dir": ("runs/logs", "日志目录（相对项目根）"),
    },
    "ui": {
        "source": ("video", "上次界面选择的采集源（关掉重开自动恢复）"),
        "input": ("data/samples/acdc", "上次界面填的路径 / URL"),
        "submit_fps": (2.0, "上次界面填的提交帧率"),
        "display": ("result", "上次界面选的显示方式：result | blend"),
        "mic": (False, "上次界面是否开启麦克风"),
    },
}

# 环境变量 → (段, 键, 类型)。只收这几个常用的，不做通用映射（避免和环境里
# 无关的 EGOMED 变量撞车，也避免出现"名字怎么写"的争论）。
ENV_MAP = {
    "EGOMED_SERVER_URL": ("server", "url", str),
    "EGOMED_FPS": ("ui", "submit_fps", float),
    "EGOMED_RENDERER": ("display", "renderer", str),
    "EGOMED_LOG_LEVEL": ("logging", "level", str),
}

PRELUDE = """# EgoMed-Agent 配置（软件级）
# 优先级：命令行参数 > 环境变量(EGOMED_*) > 本文件 > 内置默认值
# 说明：本文件不存任何密钥；由 egomed_config.py 读写，手工编辑也可以（坏了下一次启动会自动备份 .bak）
"""


# ---------------------------------------------------------------- 默认值

def default_config() -> dict:
    """按 SCHEMA 生成内置默认配置（深拷贝，调用方可随意改）。"""
    return {sec: {k: v[0] for k, v in body.items()} for sec, body in SCHEMA.items()}


def _deep_merge(base: dict, over: dict) -> dict:
    """把 over 叠到 base 上：只认 SCHEMA 里存在的段与键，未知键原样保留但不参与合并。"""
    out = {sec: dict(body) for sec, body in base.items()}
    for sec, body in (over or {}).items():
        if not isinstance(body, dict):
            continue
        out.setdefault(sec, {})
        for k, v in body.items():
            out[sec][k] = v
    return out


def config_path(root: Path, override: str = "") -> Path:
    """配置文件路径：环境变量 EGOMED_CONFIG / 显式 override 优先，否则项目根。"""
    cand = override or os.environ.get("EGOMED_CONFIG", "")
    if cand:
        p = Path(cand)
        return p if p.is_absolute() else Path(root) / p
    return Path(root) / CONFIG_NAME


# ---------------------------------------------------------------- 读

def load_file(path: Path) -> tuple[dict, list[str], bool]:
    """只读"文件这一层"。返回 (配置, notes, 是否发生过损坏备份)。"""
    notes: list[str] = []
    path = Path(path)
    if not path.is_file():
        notes.append(f"未找到 {path}，使用内置默认值")
        return {}, notes, False
    try:
        import tomllib
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        if not isinstance(data, dict):
            raise ValueError("TOML 顶层必须是 table")
    except Exception as exc:  # noqa: BLE001
        bak = _quarantine(path)
        notes.append(f"{path.name} 解析失败（{exc}），已备份为 {bak.name} 并用默认值启动")
        return {}, notes, True
    return data, notes, False


def _quarantine(path: Path) -> Path:
    """坏配置改名留档（不删、不覆盖），下一次保存会重建一份干净的。"""
    bak = path.with_name(f"{path.stem}.bak-{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    try:
        shutil.move(str(path), str(bak))
    except Exception:  # noqa: BLE001
        pass
    return bak


def load(root: Path, override: str = "") -> tuple[dict, list[str]]:
    """四层里的"默认 + 文件 + 环境变量"三层（命令行层由调用方自己优先）。

    返回 (配置, notes)。notes 是给人和日志看的一句话说明，永不抛异常。
    """
    root = Path(root)
    path = config_path(root, override)
    file_cfg, notes, _broken = load_file(path)
    cfg = _deep_merge(default_config(), file_cfg)

    for env_name, (sec, key, conv) in ENV_MAP.items():
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        try:
            cfg.setdefault(sec, {})[key] = conv(raw)
            notes.append(f"环境变量 {env_name} 覆盖 {sec}.{key}")
        except Exception:  # noqa: BLE001
            notes.append(f"环境变量 {env_name}={raw!r} 类型不对，已忽略")
    if not notes:
        notes.append(f"已加载 {path}")
    return cfg, notes


# ---------------------------------------------------------------- 写

def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    s = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def dumps(cfg: dict) -> str:
    """按 SCHEMA 顺序生成带注释的 TOML；SCHEMA 外的段/键追加在后面（不丢数据）。"""
    lines = [PRELUDE.rstrip("\n"), ""]
    seen = set()
    order = list(SCHEMA.keys()) + [s for s in cfg.keys() if s not in SCHEMA]
    for sec in order:
        body = cfg.get(sec)
        if not isinstance(body, dict):
            continue
        seen.add(sec)
        lines.append(f"[{sec}]")
        keys = list(SCHEMA.get(sec, {}).keys()) + [k for k in body if k not in SCHEMA.get(sec, {})]
        for k in keys:
            if k not in body or isinstance(body[k], dict):
                continue
            comment = SCHEMA.get(sec, {}).get(k, (None, ""))[1]
            tail = f"  # {comment}" if comment else ""
            lines.append(f"{k} = {_fmt(body[k])}{tail}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def save(root: Path, cfg: dict, override: str = "") -> Path:
    """整份回写（含注释）。写失败抛异常 —— 调用方要提示用户，别静默丢设置。"""
    path = config_path(root, override)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(cfg), encoding="utf-8")
    tmp.replace(path)          # 原子替换：写一半断电也不会留下半个配置
    return path


def save_sections(root: Path, sections: dict, override: str = "") -> Path:
    """只更新指定段/键（设置窗口用），同样只基于文件层回写，不写入环境变量值。"""
    path = config_path(root, override)
    file_cfg, _notes, _broken = load_file(path)
    merged = _deep_merge(default_config(), file_cfg)
    for sec, body in (sections or {}).items():
        if isinstance(body, dict):
            merged.setdefault(sec, {}).update(body)
        else:
            merged[sec] = body
    return save(root, merged, override)


def save_ui(root: Path, ui: dict, override: str = "") -> Path:
    """只更新 `[ui]` 段（上次界面选择）。

    只读**文件层**再改 ui，绝不用合并了环境变量的结果回写 ——
    否则 `EGOMED_FPS=9` 跑一次，配置文件就被永久改成 9，环境变量成了"污染源"。
    """
    path = config_path(root, override)
    file_cfg, _notes, _broken = load_file(path)
    merged = _deep_merge(default_config(), file_cfg)
    merged["ui"] = {**merged.get("ui", {}), **(ui or {})}
    return save(root, merged, override)


def _selftest() -> int:
    """无人值守验收（增量 3 的三条 + 两条补充），全程在临时目录里折腾，不碰真配置。"""
    import tempfile

    ok = fail = 0

    def check(name: str, cond: bool, extra="") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"[PASS] {name}")
        else:
            fail += 1
            print(f"[FAIL] {name}   {extra}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = root / CONFIG_NAME

        # ① 删掉（不存在）→ 用默认值，且不凭空造文件
        cfg, notes = load(root)
        check("缺文件用默认值", cfg["server"]["url"] == "http://127.0.0.1:8000" and not path.exists())
        check("缺文件有说明", any("未找到" in n for n in notes), notes)

        # ② 保存 [ui] → 重新读回（关掉重开记住选择）
        p = save_ui(root, {"source": "cam", "input": "0", "submit_fps": 3.5,
                           "display": "blend", "mic": True})
        cfg2, _ = load(root)
        check("配置文件已生成", p.is_file())
        check("ui 往返一致",
              cfg2["ui"]["source"] == "cam" and cfg2["ui"]["submit_fps"] == 3.5
              and cfg2["ui"]["mic"] is True and cfg2["ui"]["display"] == "blend",
              cfg2["ui"])
        check("注释随文件保留", "# 服务端地址" in p.read_text(encoding="utf-8"))

        # ③ 手工写坏 → 备份 .bak + 默认值 + 不抛异常
        path.write_text('[server\nurl = "x"\n', encoding="utf-8")
        cfg3, notes3 = load(root)
        baks = list(root.glob("config.bak-*.toml"))
        check("坏配置回默认值", cfg3["server"]["url"] == "http://127.0.0.1:8000")
        check("坏配置已备份为 .bak", len(baks) == 1, [b.name for b in baks])
        check("坏配置有说明", any("解析失败" in n for n in notes3), notes3)

        # ④ 环境变量覆盖生效，但 save 不许把环境变量的值写进文件
        save_ui(root, {"source": "video"})
        os.environ["EGOMED_SERVER_URL"] = "http://10.0.0.9:9999"
        try:
            cfg4, _ = load(root)
            check("环境变量覆盖生效", cfg4["server"]["url"] == "http://10.0.0.9:9999")
            save_ui(root, {"source": "cam"})
            fc, _n, _b = load_file(path)
            check("环境变量未被回写进文件",
                  fc.get("server", {}).get("url") == "http://127.0.0.1:8000",
                  fc.get("server"))
        finally:
            os.environ.pop("EGOMED_SERVER_URL", None)

        # ⑤ 类型保真（int 不能变 float，bool 不能变字符串）
        save_sections(root, {"capture": {"jpeg_quality": 92}, "server": {"timeout_s": 4.5}})
        cfg5, _ = load(root)
        check("int/float/bool 类型保真",
              cfg5["capture"]["jpeg_quality"] == 92
              and isinstance(cfg5["capture"]["jpeg_quality"], int)
              and abs(cfg5["server"]["timeout_s"] - 4.5) < 1e-9
              and cfg5["ui"]["mic"] in (True, False)
              and isinstance(cfg5["ui"]["mic"], bool))

    print(f"\n自测结论：{ok} PASS / {fail} FAIL")
    return 0 if fail == 0 else 4


if __name__ == "__main__":       # 手工看一眼当前生效配置 / 跑自测
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    _root = Path(__file__).resolve().parent
    _args = [a for a in sys.argv[1:] if not a.startswith("-")]
    _cfg, _notes = load(_root, _args[0] if _args else "")
    for _n in _notes:
        print(f"[--] {_n}")
    print(dumps(_cfg))
