#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""便携包打包（N53 交付项）

给谁用
------
给"另一台机器要跑起来"的场景：把**能带走的部分**打成一个 zip，
对方解压 + 装环境 + 放权重就能跑。

为什么默认不含环境与权重（重要，别误会成"漏了"）
----------------------------------------------
- `envs/egomed` 实测 **9.56 GB**（torch + CUDA），塞进 zip 既慢又没意义，
  对方机器显卡驱动版本不一定一致；正确做法是对方跑 `安装环境.bat` 按
  `requirements-lock.txt` 自己装。
- `checkpoints/` + `models/` 实测 **370 MB**（sam2 308MB + 两个分类器），
  权重属于"数据/资产"不属于"软件"，默认不打；真要一起带走用 `--with-weights`。

用法
----
    python scripts/make_portable.py                  # 只打代码+文档（约 20MB）
    python scripts/make_portable.py --with-weights   # 连权重一起（约 390MB）
    python scripts/make_portable.py --out D:/tmp     # 指定输出目录

产物
----
    runs/portable/egomed_portable_v<版本>_<时间戳>.zip
    包内额外有 `PORTABLE_README.md`（对方第一眼要看的说明）

归属：闫 ｜ 关联：`软件制作/计划_软件化实施.md`（长稳 + 便携包）
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "runs" / "portable"

# 永远不进包的目录（环境、产物、数据、权重、缓存）
EXCLUDE_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__",
    "runs", "data", "checkpoints", "models",
    ".mypy_cache", ".pytest_cache", ".pytest_tmp",
}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".log", ".tmp", ".bak"}
# 打包时额外带上（放最前面，便于对方先看到）
PRIORITY_FILES = ["VERSION", "config.toml", "egomed.bat", "requirements.txt",
                  "requirements-lock.txt", "README.md"]

PORTABLE_README = """# EgoMed-Agent 便携包（v{ver}）

> 打包时间：{when}　｜　契约版本：{contract}

## 这个包里有什么

- 全部源码：`client/`（客户端、控制台、验收/回归/长稳脚本）、`server/`（服务端）、`scripts/`（工具）
- 统一入口：`egomed.bat` + `launcher.py`（子命令见下），旧 bat 保留为转发壳
- 配置：`config.toml`（软件级）、`configs/live.yaml`（客户端运行参数）
- 文档：`docs/`（工作日志、交付流程、总计划）、`软件制作/`（实施计划、错误码表）

## 这个包里**没有**什么（必须自己补）

| 缺的东西 | 体积 | 怎么补 |
|---|---|---|
| Python 环境 | 约 9.6 GB（torch + CUDA） | 跑 `安装环境.bat`，或按 `requirements-lock.txt` 自建 conda 环境 `egomed`（Python 3.10+） |
| 模型权重 | 约 370 MB | 把 `checkpoints/sam2.1_hiera_base_plus.pt` 与 `models/*.pt` 拷到同名目录 |
| 数据 | 数 GB | 帧目录放到 `data/<模态>/img/<病例>/` 下 |
{weights_line}
## 三步跑起来

```bat
1) 安装环境.bat                  :: 建 egomed 环境（已有环境可跳过）
2) egomed.bat doctor             :: 环境自检，全绿再往下
3) egomed.bat demo               :: 起服务端（等预热）→ 打开现场控制台
```

## 常用子命令

```bat
egomed.bat                 打开 GUI（默认）
egomed.bat doctor          环境自检（精简，返回码 0 通过 / 2 有问题）
egomed.bat check           详细环境检查（14 项）
egomed.bat demo            一键演示：起服务端 → 预热 → 开控制台
egomed.bat sim             服务端 + 命令行客户端
egomed.bat diag            生成诊断包（含 versions.json 与配置快照）
egomed.bat regress         一键回归（mock + real）
egomed.bat soak -m 60      长稳测试
```

## 出问题先看

1. `egomed.bat doctor` —— 环境问题（E3xxx）会直接报出来；
2. `软件制作/错误码.md` —— 每个码都写了「一句人话 + 一条下一步」；
3. `egomed.bat diag` 生成的诊断包 —— 里面有 `versions.json`（版本/契约/Python/git commit）、
   配置快照、服务端日志尾部和最近报告。
"""


def read_version() -> str:
    for cand in (PROJECT_ROOT / "VERSION",):
        try:
            if cand.is_file():
                txt = cand.read_text(encoding="utf-8").strip().splitlines()
                if txt and txt[0].strip():
                    return txt[0].strip()
        except Exception:  # noqa: BLE001
            pass
    return "unknown"


def read_contract() -> str:
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import launcher
        return str(getattr(launcher, "CONTRACT_VERSION", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def iter_files(with_weights: bool = False):
    skip = set(EXCLUDE_DIRS)
    if with_weights:
        skip -= {"checkpoints", "models"}
    for p in sorted(PROJECT_ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(PROJECT_ROOT)
        if any(part in skip for part in rel.parts[:-1]):
            continue
        if p.suffix.lower() in EXCLUDE_SUFFIX:
            continue
        yield p, rel


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="EgoMed-Agent 便携包打包")
    ap.add_argument("--out", default=str(OUT_ROOT), help="输出目录（默认 runs/portable）")
    ap.add_argument("--with-weights", action="store_true",
                    help="连 checkpoints/ 与 models/ 一起打包（约 +370MB）")
    args = ap.parse_args(argv)

    ver, contract = read_version(), read_contract()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = out_dir / f"egomed_portable_v{ver}_{stamp}.zip"

    readme = PORTABLE_README.format(
        ver=ver, contract=contract,
        when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        weights_line=("" if not args.with_weights else
                      "> 注：本包**已包含** `checkpoints/` 与 `models/` 权重。\n"),
    )

    files = list(iter_files(args.with_weights))
    total = sum(p.stat().st_size for p, _ in files)
    print(f"[--] 软件版本 v{ver}（契约 {contract}）")
    print(f"[--] 待打包 {len(files)} 个文件，原始 {total / 1048576:.1f} MB")

    priority = [rel for rel in PRIORITY_FILES
                if (PROJECT_ROOT / rel).is_file() and rel not in {str(r) for _, r in files}]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("PORTABLE_README.md", readme)
        for rel in priority:
            zf.write(PROJECT_ROOT / rel, rel)
        for p, rel in files:
            zf.write(p, rel.as_posix())

    size_mb = zip_path.stat().st_size / 1048576
    print(f"[OK] 便携包：{zip_path}")
    print(f"     压缩后 {size_mb:.1f} MB（{len(files) + 1 + len(priority)} 个条目）")
    if not args.with_weights:
        print("[--] 不含 Python 环境与权重 —— 对方需跑「安装环境.bat」并放入权重，详见包内 PORTABLE_README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
