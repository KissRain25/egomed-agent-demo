"""为缺评测元数据的数据集生成 `split_cases.txt` / `classes.txt` / `<DS>.xlsx`。

背景（2026-09-29 复现补跑）：
    评测脚本 `scripts/egomed_agent/egomed-agent-iou06.py` 需要三个随仓库/数据集提供的元数据：
      1. `data/yolo_det/<DS>/split_cases.txt` —— `[train]/[val]/[test]` 分节病例列表
      2. `data/yolo_det/<DS>/classes.txt`    —— 类别清单（`id: name`）
      3. `data/<DS>/<DS>.xlsx`               —— 含 `灰度标签像素含义` 列（`灰度:类别名`）
    本机只有 ACDC / Amos 有这些文件（且 Amos 的 `[test]` 只列了当时有标注的 2 个病例）。

    远端数据集仓库的 `splits/<DS>_splits.tsv` 只覆盖 ACDC / Montgomery / PolypGen，
    且 **Montgomery 的 split 全部为 `unspecified`**（无官方划分）；因此对这类数据集，
    按**评测计划表**（`data/text_prompt_eval/<DS>_test_prompt_schedule.csv`，
    仓库随发布提供、定义 438 个无歧义评测键）反推 test 病例与类别灰度映射。

用法：
    python scripts/build_eval_metadata.py --datasets Montgomery-County-CXR-Set
    python scripts/build_eval_metadata.py            # 默认：所有缺元数据的数据集
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SCHEDULE = DATA_ROOT / "text_prompt_eval" / "egomed5_test_prompt_schedule.csv"

ALL_DATASETS = [
    "Amos",
    "CAMUS",
    "ACDC",
    "Montgomery-County-CXR-Set",
    "PolypGen2021_MultiCenterData_v3",
]

XLSX_COLUMNS = [
    "数据集", "原数据集序号", "起始帧", "地点", "序号",
    "动作幅度(左右lr，前后fb，静止s)", "检查1", "检查2", "灰度标签像素含义",
]


def load_schedule() -> dict[str, dict]:
    """dataset -> {target_class: gray_value, cases: set, rows: [...]}"""
    out: dict[str, dict] = {}
    with SCHEDULE.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ds = row["dataset"]
            info = out.setdefault(ds, {"mapping": {}, "cases": set(), "all_cases": set()})
            info["all_cases"].add(str(row["case_id"]))
            if row["prompt_type"] == "exact":
                info["mapping"][row["target_class"]] = int(float(row["target_gray_value"]))
                info["cases"].add(str(row["case_id"]))
    return out


def backup_if_exists(dst: pathlib.Path) -> pathlib.Path | None:
    """覆盖已有元数据前先备份（`data/` 不在 git 里，覆盖不可逆）。"""
    if not dst.exists():
        return None
    stamp = "20260929"
    bak = dst.with_suffix(dst.suffix + f".bak-{stamp}")
    if not bak.exists():
        bak.write_bytes(dst.read_bytes())
    return bak


def write_classes_txt(dataset: str, mapping: dict[str, int]) -> pathlib.Path:
    dst = DATA_ROOT / "yolo_det" / dataset / "classes.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(dst)
    items = sorted(mapping.items(), key=lambda kv: kv[1])
    lines = [f"{i}: {name}" for i, (name, _gray) in enumerate(items)]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dst


def write_split_cases(dataset: str, test_cases: set[str], extra_cases: set[str]) -> pathlib.Path:
    dst = DATA_ROOT / "yolo_det" / dataset / "split_cases.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(dst)

    def sort_key(x: str):
        return (0, int(x)) if x.isdigit() else (1, x)

    test = sorted(test_cases, key=sort_key)
    rest = sorted(extra_cases - test_cases, key=sort_key)
    lines = ["[train]"] + rest + ["", "[val]", "", "[test]"] + test + [""]
    dst.write_text("\n".join(lines), encoding="utf-8")
    return dst


def detector_class_names(dataset: str) -> dict[int, str]:
    """读该数据集 YOLO 检测器的类名（灰度值约定 = 类序号 + 1）。"""
    from ultralytics import YOLO

    weight = REPO_ROOT / "runs" / "yolo26_det" / f"{dataset}_yolo26m_imgsz1024" / "weights" / "best.pt"
    if not weight.exists():
        return {}
    names = YOLO(str(weight)).names
    return {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))


def write_xlsx(dataset: str, mapping: dict[str, int], test_cases: set[str],
               names: dict[int, str] | None = None) -> pathlib.Path:
    """把灰度映射写在 `灰度标签像素含义` 列（解析逻辑只读该列）。

    映射取**检测器全部类名**（灰色值 = 类序号 + 1），而不是只用计划表里出现的类：
    这与数据集原始 `灰度标签像素含义`（如 Amos 的 `4:gall bladder`、`6:liver`）一致，
    也让元数据完整（评测脚本对 Amos 用 FOCUS_TARGETS 只会用到其中 5 类）。
    """
    dst = DATA_ROOT / dataset / f"{dataset}.xlsx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(dst)
    pairs: list[tuple[str, int]] = []
    if names:
        pairs = [(nm, idx + 1) for idx, nm in sorted(names.items())]
        # 与计划表交叉校验：计划表出现的 (类名 -> 灰度) 必须一致
        for name, gray in mapping.items():
            hit = [g for nm, g in pairs if nm == name]
            if hit and hit[0] != gray:
                print(f"    [警告] {dataset} 计划表 {name} 灰度 {gray} 与检测器序号+1={hit[0]} 不一致")
    else:
        pairs = sorted(mapping.items(), key=lambda kv: kv[1])

    rows: list[dict] = []
    rows.append({c: None for c in XLSX_COLUMNS} | {"数据集": dataset, "灰度标签像素含义": "0:background"})
    for name, gray in pairs:
        rows.append({c: None for c in XLSX_COLUMNS} | {"数据集": dataset, "灰度标签像素含义": f"{gray}:{name}"})
    for case in sorted(test_cases, key=lambda x: int(x) if x.isdigit() else 0):
        rows.append({c: None for c in XLSX_COLUMNS} | {"数据集": dataset, "原数据集序号": case, "序号": case})
    pd.DataFrame(rows, columns=XLSX_COLUMNS).to_excel(dst, index=False)
    return dst


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="生成评测元数据（划分/类别/灰度映射）")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    sched = load_schedule()
    targets = args.datasets or [d for d in ALL_DATASETS if d in sched]
    print("%-38s %6s %6s %s" % ("数据集", "测试病例", "类别数", "灰度映射"))
    for ds in targets:
        info = sched.get(ds)
        if not info or not info["mapping"]:
            print("%-38s 计划表无该数据集，跳过" % ds)
            continue
        mapping = info["mapping"]
        img_root = DATA_ROOT / ds / "img"
        local_cases = {p.name for p in img_root.iterdir() if p.is_dir()} if img_root.exists() else set()
        test_cases = {c for c in info["cases"] if c in local_cases} or set(info["cases"])
        c1 = write_classes_txt(ds, mapping)
        c2 = write_split_cases(ds, test_cases, local_cases)
        c3 = write_xlsx(ds, mapping, test_cases, names=detector_class_names(ds))
        print("%-38s %6d %6d %s" % (ds, len(test_cases), len(mapping),
                                    ", ".join(f"{g}:{n}" for n, g in sorted(mapping.items(), key=lambda kv: kv[1]))))
        for p in (c1, c2, c3):
            print("      写出:", p.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
