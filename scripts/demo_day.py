#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""当众演示 + 指标当众跑（N43，2026-11-04 新增）。

一条命令把"系统能跑"和"指标是多少"当场摆出来，并出可留存的报告。

    python scripts/demo_day.py                # 全套：预检 → 起真引擎 → 验收 → 延迟 → 精度 → N48 演示 → 面板
    python scripts/demo_day.py --quick        # 跳过精度评测（省约 2 分钟）
    python scripts/demo_day.py --keep-server  # 跑完不关服务端（现场追问时可直接发帧）
    python scripts/demo_day.py --port 8021    # 换端口（默认 8020）

原则：**评测全部复用现有脚本的 CLI**，本脚本只做编排 + 面板 + 报告，不改它们的行为；
**失败不崩**（任一环节挂了继续走，最后统一给结论），避免现场演示黑屏。
产物：`runs/demo_day/<时间戳>/report.md`。
前提：测延迟/精度前**必须腾空 GPU**（外部占用会让语音推理从 0.6s 拖到 88~144s，那是环境问题）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT / "client"
RUNS = ROOT / "runs"
PY = sys.executable


def log(tag: str, msg: str) -> None:
    print(f"{tag} {msg}", flush=True)


def ms(v) -> str:
    try:
        return f"{float(v):.0f}ms"
    except Exception:  # noqa: BLE001
        return "-"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) != 0


def http_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def latest_dir(base: Path, prefix: str = "") -> Path | None:
    if not base.is_dir():
        return None
    cands = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)],
                   key=lambda d: d.name)
    return cands[-1] if cands else None


def preflight(port: int) -> list[str]:
    """环境体检。返回问题清单（空 = 一切正常）。"""
    problems: list[str] = []
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            out = subprocess.run(
                [nvidia, "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            used, total, util = [int(x.strip()) for x in out.split(",")[:3]]
            log("[--]", f"GPU 占用 {used}/{total} MiB（{util}%）")
            if used > 3000:
                problems.append(f"GPU 已被占 {used} MiB（{util}%）：测延迟/语音会假失败，先关掉占用程序")
        except Exception as exc:  # noqa: BLE001
            log("[--]", f"GPU 状态读取失败（不阻塞）：{exc}")
    need = ["runs/yolo26_det/Amos_yolo26m_imgsz1024/weights",
            "runs/yolo26_det/CAMUS_yolo26m_imgsz1024/weights",
            "runs/yolo26_det/Montgomery-County-CXR-Set_yolo26m_imgsz1024/weights",
            "runs/yolo26_det/PolypGen2021_MultiCenterData_v3_yolo26m_imgsz1024/weights",
            "runs/yolo26_det/ACDC_yolo26m_imgsz1024/weights",
            "models/imaging_modality_classifier.pt"]
    missing = [d for d in need if not (ROOT / d).exists()]
    if missing:
        problems.append(f"缺 {len(missing)} 项权重/模型：{missing[:3]}")
    else:
        log("[OK]", f"权重与分类器齐全（{len(need)} 项）")
    if not (ROOT / "data" / "ACDC" / "img").is_dir():
        problems.append("缺 data/ACDC/img（精度评测要用；可 --quick 跳过）")
    if not port_free(port):
        problems.append(f"端口 {port} 已被占用（换 --port，或先停掉旧服务端）")
    else:
        log("[OK]", f"端口 {port} 空闲")
    return problems


def start_server(port: int, out_dir: Path) -> subprocess.Popen:
    log("[..]", f"起真引擎服务端（--real --port {port}）…")
    out = (out_dir / "server_out.txt").open("w", encoding="utf-8")
    err = (out_dir / "server_err.txt").open("w", encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.Popen(
        [PY, str(ROOT / "server" / "api.py"), "--port", str(port), "--real"],
        cwd=str(ROOT), stdout=out, stderr=err, env=env)


def wait_ready(port: int, timeout: float = 300.0) -> bool:
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/healthz"
    last = ""
    while time.time() - t0 < timeout:
        try:
            h = http_json(url, timeout=3)
            state = h.get("state")
            last = f"state={state} real_engine={h.get('real_engine')}"
            if state == "ready" and h.get("real_engine"):
                log("[OK]", f"服务端就绪（{time.time() - t0:.0f}s，{last}）")
                return True
            if state == "failed":
                log("[!!]", "服务端预热失败（state=failed），看 server_err.txt")
                return False
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:80]
        time.sleep(2.0)
    log("[!!]", f"服务端等待超时（{timeout:.0f}s，最后状态：{last}）")
    return False


def prime_asr(port: int) -> None:
    """先发一条语音吃掉 ASR 冷启动（否则第一条语音可能 >30s，客户端会超时）。"""
    wav = ROOT / "data" / "samples" / "audio" / "cmd_03.wav"
    if not wav.is_file():
        return
    try:
        sys.path.insert(0, str(CLIENT))
        import mock_client as mc
        t0 = time.time()
        _, t_ms, err = mc.post_audio(f"http://127.0.0.1:{port}", "demo_prime",
                                     wav.read_bytes(), wav.name, 120.0)
        log("[OK]", f"ASR 预热完成（{time.time() - t0:.1f}s，服务端 {ms(t_ms)}"
                    f"{'，' + err if err else ''}）")
    except Exception as exc:  # noqa: BLE001
        log("[--]", f"ASR 预热跳过：{exc}")


def run_step(name: str, cmd: list[str], out_dir: Path, timeout: float = 1800.0) -> int:
    log("[..]", f"{name} …")
    t0 = time.time()
    with (out_dir / f"{name}.log").open("w", encoding="utf-8") as fh:
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            log("[!!]", f"{name} 超时（{timeout:.0f}s）")
            return 124
    log("[OK]" if rc == 0 else "[!!]", f"{name} 退出码 {rc}，耗时 {time.time() - t0:.0f}s")
    return rc


def n48_demo(url: str, frames: int = 4) -> dict:
    """未设目标连发几帧，看服务端能不能自动识别模态（N48 现场演示）。"""
    try:
        sys.path.insert(0, str(CLIENT))
        import mock_client as mc
        src = ROOT / "data" / "samples" / "acdc"
        imgs = sorted(src.glob("*.jpg"))[:frames] or sorted(src.glob("*.png"))[:frames]
        if not imgs:
            return {"ok": False, "error": f"{src} 下没有样例图"}
        sid = mc.make_session_id("demo48")
        got = []
        for i, fp in enumerate(imgs):
            payload, _t, err = mc.post_frame(url, sid, mc.compress_image(fp),
                                             f"n48_{i:03d}.jpg", 30.0)
            if err:
                return {"ok": False, "error": err}
            got.append((payload.get("modality") or "", payload.get("message") or ""))
        stable_at = next((i + 1 for i, (m, _) in enumerate(got) if m), None)
        mod = next((m for m, _ in got if m), "")
        msg = next((t for m, t in got if m), "")
        return {"ok": bool(mod), "frames": len(got), "stable_at": stable_at,
                "modality": mod, "message": msg}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def build_panel(acc, lat, seg, n48, meta) -> list[str]:
    """把四组数字拼成一张**当众能看懂**的面板。"""
    L: list[str] = []
    bar = "=" * 78
    L.append(bar)
    L.append(f" EgoMed-Agent 当众指标　{meta['ts']}　（版本 {meta['version']} · 契约 {meta['contract']}）")
    L.append(bar)

    L.append("")
    L.append(" 【一】端到端延迟 · A1 客户端往返（真引擎，已预热，GPU 空闲）")
    if lat and lat.get("cases"):
        for c in lat["cases"]:
            if c.get("error"):
                L.append(f"   {c.get('modality', '?'):22s} X {c['error']}")
                continue
            L.append(f"   {c['modality']:22s} 检出 {c['hits']}/{c['frames']} 帧　"
                     f"均值 {ms(c['mean']):>7s}　P50 {ms(c['p50']):>7s}　P95 {ms(c['p95']):>7s}")
        ov = lat.get("overall") or {}
        if ov:
            L.append(f"   {'-- 全部合并':22s} {'':16s}均值 {ms(ov.get('mean')):>7s}　"
                     f"P50 {ms(ov.get('p50')):>7s}　P95 {ms(ov.get('p95')):>7s}　"
                     f"超时 {ov.get('timeouts', 0)} 次")
        segs = lat.get("segments") or {}
        if segs:
            tot = sum(segs.values()) or 1.0
            top = sorted(segs.items(), key=lambda kv: -kv[1])[:3]
            L.append("   耗时构成前三：" + "、".join(f"{k} {v / tot * 100:.0f}%" for k, v in top))
    else:
        L.append("   （本次没拿到数据）")

    L.append("")
    L.append(" 【二】分割精度（ACDC 4 病例 x 8 帧）")
    if seg:
        on = seg.get("online") or {}
        off = seg.get("offline") or {}
        if on:
            L.append(f"   在线 Dice 均值 **{on.get('dice_mean', 0):.4f}**"
                     f"（检出 {on.get('hits')}/{on.get('frames')}）　"
                     f"延迟均值 {ms(on.get('latency_mean_ms'))}　P95 {ms(on.get('latency_p95_ms'))}")
        if off:
            L.append(f"   离线 Dice 均值 {off.get('dice_mean', 0):.4f}"
                     f"（检出 {off.get('hits')}/{off.get('frames')}）")
        base = meta.get("baseline_online_dice")
        if on and base:
            d = on.get("dice_mean", 0) - base
            L.append(f"   与基线对比：在线 {base:.4f} -> {on.get('dice_mean', 0):.4f}"
                     f"（差值 {d:+.4f}{'，零退化' if abs(d) < 1e-9 else ''}）")
    else:
        L.append("   （本次没拿到数据，或用了 --quick 跳过）")

    L.append("")
    L.append(" 【三】验收套件（真引擎）")
    if acc:
        L.append(f"   通过 **{acc.get('passed', 0)}/{acc.get('total', 0)}**"
                 + ("　全部通过" if acc.get("passed") == acc.get("total") else "　有失败项，见 report.json"))
        for c in [x for x in acc.get("cases", []) if not x.get("passed")][:5]:
            L.append(f"   X {c.get('case') or c.get('name')}：{str(c.get('detail'))[:70]}")
    else:
        L.append("   （本次没拿到数据）")

    L.append("")
    L.append(" 【四】新能力现场演示：还没说话时，自动认出这是哪张片子（N48）")
    if n48 and n48.get("ok"):
        L.append(f"   连续 {n48['frames']} 帧未设目标 -> 第 {n48.get('stable_at')} 帧起给出："
                 f"**{n48.get('modality')}**")
        L.append(f"   提示语：{n48.get('message')}")
    else:
        L.append(f"   （未复现：{(n48 or {}).get('error', '未执行')}）")

    L.append("")
    L.append(bar)
    good = bool(acc and acc.get("total") and acc.get("passed") == acc.get("total")
                and lat and lat.get("overall"))
    L.append(f" 结论：{'可当众演示' if good else '有环节未过（见上方与 report.md）'}")
    L.append(bar)
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="当众演示 + 指标当众跑（N43）")
    ap.add_argument("--port", type=int, default=8020, help="服务端端口（默认 8020）")
    ap.add_argument("--quick", action="store_true", help="跳过精度评测（省约 2 分钟）")
    ap.add_argument("--keep-server", action="store_true", help="跑完不关服务端")
    ap.add_argument("--no-panel", action="store_true", help="只跑不打印面板")
    ap.add_argument("--frames", type=int, default=12, help="每模态发帧数（延迟评测）")
    ap.add_argument("--cases", type=int, default=4, help="精度评测病例数")
    args = ap.parse_args(argv)

    now = datetime.now()
    out_dir = RUNS / "demo_day" / now.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    vfile = ROOT / "VERSION"
    version = vfile.read_text(encoding="utf-8-sig").strip() if vfile.is_file() else "?"
    meta = {"ts": now.strftime("%Y-%m-%d %H:%M:%S"), "version": version,
            "contract": "v1.0.7", "baseline_online_dice": 0.403}

    print("=" * 78)
    print(f" EgoMed-Agent 当众演示　{meta['ts']}　版本 {version}")
    print("=" * 78)

    for p in preflight(args.port):
        log("[!!]", p)

    proc = None
    acc = lat = seg = n48 = None
    try:
        proc = start_server(args.port, out_dir)
        if not wait_ready(args.port):
            log("[!!]", "服务端没就绪，后面的跑不了（面板会如实标注）")
        else:
            prime_asr(args.port)

            run_step("acceptance", [PY, str(CLIENT / "acceptance_test.py"),
                                     "--suite", "real", "--url", url], out_dir)
            acc_dir = latest_dir(RUNS / "acceptance")
            if acc_dir:
                rep = read_json(acc_dir / "report.json") or {}
                cases = rep.get("cases") or []
                acc = {"passed": sum(1 for c in cases if c.get("passed")),
                       "total": len(cases), "cases": cases, "dir": str(acc_dir)}

            run_step("latency", [PY, str(CLIENT / "eval_latency.py"),
                                 "--frames", str(args.frames), "--url", url,
                                 "--detail"], out_dir)
            lat_dir = latest_dir(RUNS / "metrics", "latency_")
            if lat_dir:
                lat = read_json(lat_dir / "result.json")

            if not args.quick:
                run_step("segmentation", [PY, str(CLIENT / "eval_segmentation.py"),
                                          "--modality", "ACDC (MRI 心脏)",
                                          "--case", str(args.cases), "--frames", "8",
                                          "--mode", "both", "--url", url], out_dir)
                seg_dir = latest_dir(RUNS / "metrics")
                if seg_dir:
                    for cand in sorted(seg_dir.iterdir(), key=lambda d: d.name, reverse=True):
                        j = read_json(cand / "result.json")
                        if j and ("online" in j or "offline" in j):
                            seg = j
                            break

            n48 = n48_demo(url)
    finally:
        if proc is not None:
            if args.keep_server:
                log("[--]", "服务端保留运行中")
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except Exception:  # noqa: BLE001
                    proc.kill()
                log("[--]", "服务端已关闭")

    panel = build_panel(acc, lat, seg, n48, meta)
    if not args.no_panel:
        print()
        print("\n".join(panel))

    lines = [f"# 当众演示报告　{meta['ts']}", "",
             f"- 版本 {version}｜契约 {meta['contract']}｜端口 {args.port}", ""]
    lines += ["## 当众面板", "", "```"] + panel + ["```", ""]
    lines += ["## 子报告", ""]
    for label, path in (("验收", (acc or {}).get("dir")),
                        ("延迟", str(latest_dir(RUNS / "metrics", "latency_") or "")),
                        ("本次日志目录", str(out_dir))):
        if path:
            lines.append(f"- {label}：`{path}`")
    lines += ["", "## 各步日志", ""]
    for f in sorted(out_dir.glob("*.log")):
        lines.append(f"- `{f.name}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告：{(out_dir / 'report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
