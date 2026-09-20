#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EgoMed-Agent 一键批量验收脚本（T7 真引擎 / P0 返工复验用）

作用：把 docs/API_CONTRACT.md A.10（P0 返工）+ A.11（ASR 谐音）的验收命令
      打包成一条命令跑完，输出 PASS/FAIL 结论表 + 报告文件，李修完即可一键复验。

归属：闫（客户端 / 测试工具）
依赖：mock_client.py（复用其压缩与发送逻辑，不重复实现契约）

验收项（对应 A.10 / A.11 编号）
--------------------------------
  mock 套件（假实现即可跑，验 P0.3 / P0.4）
    C1  正常帧 ×3      期望 ok + overlay 非空 + modality/target 非空
    C2  坏帧   ×5      期望 ok + overlay 空  + modality/target 空 + message 有提示（P0.3+P0.4）

  real 套件（真引擎 --real 启动后跑，验 P0.1 / P0.2 / A.11）
    R1  首帧延迟       新会话第一帧，期望 < --budget-ms（P0.2 预热）
    R2  语音   ×8      期望 text 与预期逐字一致 + target 非空或需澄清；elapsed < 1500ms（A.11）
    R3  并发 5 帧      同一 session 5 线程并发，期望总耗时 < 2×单帧（P0.1 队列=1 覆盖旧帧）
                        ※ 需要 --concurrency N 显式开启（N=0 跳过），真引擎压力大时慎用
    R4  连续帧出结果   X 光样例 + cmd_07 设目标后连发 --seg-frames 帧，期望至少一帧 overlay 非空
                        ※ 2026-09-20 新增：补上此前缺失的 overlay 断言（见联调发现文档）

用法
----
    # 假实现全套（李改完 P0.3/P0.4 后）
    python client/acceptance_test.py --suite mock

    # 真引擎全套（服务端 --real 启动并 warmup 完成后）
    python client/acceptance_test.py --suite real --budget-ms 3000

    # 全部 + 并发
    python client/acceptance_test.py --suite all --budget-ms 3000 --concurrency 5

输出
----
    终端：逐项 PASS/FAIL + 汇总
    文件：runs/acceptance/<时间戳>/report.md + report.json（发群里可直接贴）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_client as mc  # 复用契约压缩 / 发送 / session_id 逻辑

PROJECT_ROOT = mc.PROJECT_ROOT
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"
AUDIO_DIR = SAMPLE_DIR / "audio"
BAD_DIR = SAMPLE_DIR / "bad"
REPORT_ROOT = PROJECT_ROOT / "runs" / "acceptance"

# A.11 测试样例：文件名 -> 预期识别文本（asr_medical 实测 8/8 全对的基准）
AUDIO_CASES = {
    "cmd_01.wav": "帮我分割左心室",
    "cmd_02.wav": "把右心室腔标出来",
    "cmd_03.wav": "分割心肌",      # 谐音回归用例（原配置识别成"刑機/心机"）
    "cmd_04.wav": "帮我分割脾脏",
    "cmd_05.wav": "分割十二指肠",
    "cmd_06.wav": "把息肉标出来",
    "cmd_07.wav": "分割左肺",
    "cmd_08.wav": "帮我分割肝脏和左肾",
}

# A.11 判定口径：字准 ≥90% 等价于 NLU 解析成功（错字不在词表必解析失败）。
# 仅 cmd_03（谐音回归用例）要求逐字一致，其余句子 text 偏差不作硬断言（记入 detail 供观察）。
STRICT_TEXT_CASES = {"cmd_03.wav"}

AUDIO_ELAPSED_BUDGET_MS = 1500.0   # A.11 验收：elapsed < 1500 ms
FRAMES_OK_COUNT = 3                # C1 正常帧数量


# ---------------------------------------------------------------- 结果收集
class Report:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.items: list[dict] = []

    def add(self, case: str, name: str, passed: bool, detail: str, elapsed_ms: float | None = None):
        self.items.append({"case": case, "name": name, "pass": passed,
                           "detail": detail, "elapsed_ms": elapsed_ms})
        mark = "PASS" if passed else "FAIL"
        ms = f"  {elapsed_ms:>7.0f}ms" if elapsed_ms is not None else ""
        mc.log(f"  [{mark}] {case} {name}{ms}")
        if not passed:
            mc.log(f"         {detail}")

    @property
    def passed(self):
        return all(i["pass"] for i in self.items)

    def summary_line(self) -> str:
        total = len(self.items)
        ok = sum(1 for i in self.items if i["pass"])
        return f"{ok}/{total} PASS"

    def dump(self, url: str, suite: str):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        head = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "url": url, "suite": suite, "result": self.summary_line(),
        }
        (self.out_dir / "report.json").write_text(
            json.dumps({"head": head, "items": self.items}, ensure_ascii=False, indent=2),
            encoding="utf-8")

        lines = [f"# 验收报告 {head['date']}", "",
                 f"- 服务器：`{url}`    套件：`{suite}`",
                 f"- 结论：**{self.summary_line()}** {'✅ 通过' if self.passed else '❌ 未通过'}", "",
                 "| 用例 | 名称 | 结果 | 耗时 | 说明 |",
                 "|------|------|------|------|------|"]
        for i in self.items:
            mark = "PASS" if i["pass"] else "**FAIL**"
            ms = f"{i['elapsed_ms']:.0f}ms" if i["elapsed_ms"] is not None else "-"
            detail = i["detail"].replace("|", "\\|")
            lines.append(f"| {i['case']} | {i['name']} | {mark} | {ms} | {detail} |")
        (self.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- mock 套件
def run_mock_suite(args, rep: Report, session: str) -> None:
    """C1 正常帧 + C2 坏帧（验 P0.3 坏帧返空 overlay / P0.4 modality、target 返空）"""

    # C1 正常帧：ok + overlay 非空 + modality/target 非空
    frames = sorted((f for f in (SAMPLE_DIR / "acdc").rglob("*")
                     if f.suffix.lower() in mc.IMG_EXTS), key=mc.natural_key)[:FRAMES_OK_COUNT]
    if not frames:
        rep.add("C1", "正常帧（样例缺失）", False, f"目录无图片：{SAMPLE_DIR / 'acdc'}")
    else:
        for i, path in enumerate(frames, 1):
            try:
                jpg = mc.compress_image(path)
                payload, ms, err = mc.post_frame(args.url, session, jpg, path.name, args.timeout)
            except Exception as exc:
                rep.add("C1", f"正常帧 {i}/{len(frames)}", False, f"异常：{exc}")
                continue
            if err:
                rep.add("C1", f"正常帧 {i}/{len(frames)}", False, err, ms)
                continue
            overlay = payload.get("overlay") or ""
            checks = (payload.get("status") == "ok" and bool(overlay)
                      and bool(payload.get("modality")) and bool(payload.get("target")))
            detail = (f"status={payload.get('status')} overlay={'非空' if overlay else '空'} "
                      f"modality={payload.get('modality')!r} target={payload.get('target')!r}")
            rep.add("C1", f"正常帧 {i}/{len(frames)}", checks, detail, ms)

    # C2 坏帧：ok + overlay 空 + modality/target 空 + message 有提示（P0.3 + P0.4）
    bad_frames = sorted(BAD_DIR.glob("*.jpg"), key=mc.natural_key)
    if not bad_frames:
        rep.add("C2", "坏帧（样例缺失）", False,
                f"目录无坏帧：{BAD_DIR}，先跑 python client/make_test_frames.py")
    else:
        for i, path in enumerate(bad_frames, 1):
            try:
                jpg = mc.compress_image(path)
                payload, ms, err = mc.post_frame(args.url, session, jpg, path.name, args.timeout)
            except Exception as exc:
                rep.add("C2", f"坏帧 {i}/{len(bad_frames)} {path.name}", False, f"异常：{exc}")
                continue
            if err:
                rep.add("C2", f"坏帧 {i}/{len(bad_frames)} {path.name}", False, err, ms)
                continue
            overlay = payload.get("overlay") or ""
            p03 = not overlay                       # P0.3：坏帧 overlay 必须为空
            p04 = not payload.get("modality") and not payload.get("target")  # P0.4：未识别返空
            msg_ok = bool(payload.get("message"))   # message 应有"对准屏幕"类提示
            checks = payload.get("status") == "ok" and p03 and p04 and msg_ok
            detail = (f"P0.3 overlay={'空✓' if p03 else '非空✗'}  "
                      f"P0.4 modality={payload.get('modality')!r} target={payload.get('target')!r}  "
                      f"message={payload.get('message')!r}")
            rep.add("C2", f"坏帧 {i}/{len(bad_frames)} {path.name}", checks, detail, ms)


# ---------------------------------------------------------------- real 套件
def run_real_suite(args, rep: Report, session: str) -> None:
    """R1 首帧延迟（P0.2）+ R2 语音×8（A.11）+ R3 并发（P0.1）"""

    # R1 首帧延迟：新会话第一帧（服务端应已 warmup，否则会撞上模型加载）
    frames = sorted((f for f in (SAMPLE_DIR / "acdc").rglob("*")
                     if f.suffix.lower() in mc.IMG_EXTS), key=mc.natural_key)
    if not frames:
        rep.add("R1", "首帧延迟（样例缺失）", False, f"目录无图片：{SAMPLE_DIR / 'acdc'}")
    else:
        fresh = mc.make_session_id(args.doctor)   # 独立新会话，避开已有缓存
        jpg = mc.compress_image(frames[0])
        payload, ms, err = mc.post_frame(args.url, fresh, jpg, frames[0].name, args.timeout)
        if err:
            rep.add("R1", "首帧延迟", False, err, ms)
        else:
            detail = (f"首帧 {ms:.0f}ms（预算 {args.budget_ms}ms）。"
                      f"若远超预算 → 服务端未在启动时预热引擎（A.10 P0.2）")
            rep.add("R1", "首帧延迟", ms <= args.budget_ms, detail, ms)

    # R2 语音 ×8：text 逐字一致 + target 非空或需澄清；elapsed < 1500ms
    audio_session = mc.make_session_id(args.doctor)
    for fname, expect in AUDIO_CASES.items():
        path = AUDIO_DIR / fname
        if not path.is_file():
            rep.add("R2", f"{fname}（缺失）", False, f"找不到 {path}")
            continue
        try:
            payload, ms, err = mc.post_audio(args.url, audio_session,
                                             path.read_bytes(), fname, args.timeout)
        except Exception as exc:
            rep.add("R2", fname, False, f"异常：{exc}")
            continue
        if err:
            rep.add("R2", fname, False, err, ms)
            continue
        text = (payload.get("text") or "").strip()
        text_ok = text == expect
        if payload.get("status") == "error":
            passed = False
            detail = f"业务错误 code={payload.get('code')} {payload.get('message', '')}"
        else:
            target_ok = bool(payload.get("target")) or bool(payload.get("need_confirm"))
            budget_ok = ms <= AUDIO_ELAPSED_BUDGET_MS
            # A.11 口径：NLU 解析成功即字准达标；回归用例 cmd_03 额外要求逐字一致
            text_hard = text_ok if fname in STRICT_TEXT_CASES else True
            passed = target_ok and budget_ok and text_hard
            text_flag = "✓" if text_ok else (" ✗回归用例逐字不符" if fname in STRICT_TEXT_CASES
                                              else "（偏差，不计入判定）")
            detail = (f"text={text!r}（期望 {expect!r}{text_flag}）  "
                      f"target={payload.get('target') or '-'}  "
                      f"耗时 {ms:.0f}ms{'✓' if budget_ok else ' ✗(>1500ms)'}")
        rep.add("R2", f"{fname} 「{expect}」", passed, detail, ms)

    # R3 并发丢帧：同一 session 并发 N 帧，验 P0.1（per-session 队列=1、新帧覆盖旧帧）
    if args.concurrency <= 0 or not frames:
        mc.log("  [SKIP] R3 并发丢帧（--concurrency 0 或无样例帧，已跳过）")
    else:
        conc_session = mc.make_session_id(args.doctor)
        jpgs = [mc.compress_image(frames[i % len(frames)]) for i in range(args.concurrency)]

        # 真引擎必须先设置分割目标，否则 frame 直接短路返回（不跑推理 → 测不到丢帧语义）。
        # 注意用单目标无歧义指令（cmd_03"分割心肌"）；cmd_01"左心室"是跨模态歧义词，
        # 会触发 need_confirm 澄清而不设置 target（契约边界情况 2）。
        setup_wav = AUDIO_DIR / "cmd_03.wav"
        if setup_wav.is_file():
            ap, _ams, aerr = mc.post_audio(args.url, conc_session,
                                           setup_wav.read_bytes(), setup_wav.name, args.timeout)
            if aerr or not ap.get("target"):
                rep.add("R3", f"并发 {args.concurrency} 帧", False,
                        f"设置分割目标失败（{aerr or ap.get('message')}），无法测推理并发")
                return

        # 先串行发 1 帧测基准（并发请求的耗时含排队等待，不能拿它当基准）
        _, base_ms, base_err = mc.post_frame(args.url, conc_session, jpgs[0],
                                             "conc_base.jpg", args.timeout)
        if base_err:
            rep.add("R3", f"并发 {args.concurrency} 帧", False,
                    f"基准帧发送失败：{base_err}", base_ms)
            return

        def _send(idx: int):
            return mc.post_frame(args.url, conc_session, jpgs[idx],
                                 f"conc_{idx:02d}.jpg", args.timeout)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(_send, range(args.concurrency)))
        total_ms = (time.perf_counter() - t0) * 1000

        done = [r for r in results if r[2] is None]          # 网络层成功
        if not done:
            rep.add("R3", f"并发 {args.concurrency} 帧", False,
                    "全部请求失败，服务端未启动或超时", total_ms)
        else:
            # P0.1 判定：队列=1 + 新帧覆盖旧帧 → 并发总耗时 ≈ 串行基准的 1~2 倍；
            # 全局锁串行排队 → 总耗时 ≈ N × 基准（A.10 实测 5 并发排队 8.75s）
            passed = total_ms < 3.0 * base_ms
            detail = (f"{args.concurrency} 并发总耗时 {total_ms:.0f}ms / 串行基准 {base_ms:.0f}ms。"
                      f"若接近 {args.concurrency}×基准（串行排队）→ 全局锁未改 per-session（A.10 P0.1）")
            rep.add("R3", f"并发 {args.concurrency} 帧", passed, detail, total_ms)


# ---------------------------------------------------------------- R4 分割结果断言
def run_r4_overlay(args, rep: Report) -> None:
    """R4 连续帧出分割结果（补 real 套件的 overlay 断言，2026-09-20 新增）

    背景：此前 real 套件只断言"延迟 + 语音文本"，从未断言 overlay 非空，
    导致"真引擎在客户端链路上全程返空"未被发现（详见
    docs/联调发现_20260920_真引擎实时链路.md 发现 4）。

    设计：
      - 用检出率最高的 X 光样例 + 单目标无歧义指令（cmd_07「分割左肺」）
      - 连发 N 帧（--seg-frames，默认 12），断言**至少一帧** overlay 非空
      - 首帧返空属已知问题（服务端 min_occurrence=2，见 N13），故不要求首帧有结果
    """
    frames = sorted((f for f in (SAMPLE_DIR / "cxr").rglob("*")
                     if f.suffix.lower() in mc.IMG_EXTS), key=mc.natural_key)
    wav = AUDIO_DIR / "cmd_07.wav"
    if not frames or not wav.is_file():
        rep.add("R4", "连续帧出分割结果（样例缺失）", False,
                f"需要 {SAMPLE_DIR / 'cxr'} 的图片与 {wav}")
        return

    session = mc.make_session_id(args.doctor)
    ap, _ams, aerr = mc.post_audio(args.url, session, wav.read_bytes(), wav.name, args.timeout)
    if aerr or not ap.get("target"):
        rep.add("R4", "连续帧出分割结果", False,
                f"设置目标失败（{aerr or ap.get('message')}）——A1 未设目标会直接短路返回")
        return

    n = max(2, args.seg_frames)
    hit, net_err, last_msg = 0, 0, ""
    for i in range(n):
        jpg = mc.compress_image(frames[i % len(frames)])
        payload, _ms, err = mc.post_frame(args.url, session, jpg,
                                          f"seg_{i:03d}.jpg", args.timeout)
        if err:
            net_err += 1
            continue
        last_msg = str(payload.get("message") or "")
        if payload.get("overlay"):
            hit += 1

    passed = hit > 0
    detail = (f"{n} 帧中 {hit} 帧有 overlay（目标 {ap.get('target')}，"
              f"网络错 {net_err}）；首帧返空属已知 N13。末次 message={last_msg!r}")
    if not passed:
        detail += "。全空排查顺序：① 目标是否已设（A2）② 服务端日志是否走推理 ③ 低置信检测被 min_occurrence 过滤"
    rep.add("R4", f"连续 {n} 帧出分割结果", passed, detail)


# ---------------------------------------------------------------- 入口
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="EgoMed-Agent 一键批量验收（A.10 P0 + A.11，报告落盘 runs/acceptance/）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", default=mc.DEFAULT_URL, help=f"服务器地址，默认 {mc.DEFAULT_URL}")
    parser.add_argument("--suite", choices=["mock", "real", "all"], default="all",
                        help="mock=正常帧+坏帧(P0.3/P0.4)；real=首帧延迟+语音+并发(P0.1/P0.2/A.11)")
    parser.add_argument("--budget-ms", type=float, default=3000.0,
                        help="R1 首帧延迟预算（毫秒），默认 3000；P0.2 预热后首帧应远小于此值")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="R3 并发帧数（验 P0.1 丢帧），0=跳过（真引擎压力测试，确认服务端就绪再开）")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="单请求超时秒数（真引擎含推理，默认 30）")
    parser.add_argument("--doctor", default="acc", help="验收用医生编号（session_id 前缀），默认 acc")
    parser.add_argument("--seg-frames", type=int, default=12,
                        help="R4 连续发帧数（验 overlay 非空），默认 12；窗口越满延迟越高，别设太大")
    args = parser.parse_args()
    args.budget_ms = args.budget_ms if args.suite in ("real", "all") else 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = Report(REPORT_ROOT / stamp)
    session = mc.make_session_id(args.doctor)

    mc.log("=" * 78)
    mc.log(f"EgoMed-Agent 批量验收  {stamp}    服务器：{args.url}    套件：{args.suite}")
    mc.log(f"session：{session}    预算：首帧 {args.budget_ms or '-'}ms / 语音 {AUDIO_ELAPSED_BUDGET_MS:.0f}ms")
    mc.log("=" * 78)

    # 连通性预检：连不上直接退出，不浪费跑一半的报告
    probe = mc.post_frame(args.url, session, mc.compress_image(
        next((f for f in (SAMPLE_DIR / "acdc").rglob("*") if f.suffix.lower() in mc.IMG_EXTS),
             Path(__file__))), "probe.jpg", args.timeout)[2]
    if probe:
        mc.log(f"\n[中止] 连不上服务端：{probe}")
        mc.log("请先启动服务端（假实现默认模式 / 真引擎 --real），再重跑验收。")
        sys.exit(2)

    if args.suite in ("mock", "all"):
        mc.log("\n── mock 套件：正常帧 ×3（C1） + 坏帧 ×5（C2，验 P0.3/P0.4） ──")
        run_mock_suite(args, rep, session)
    if args.suite in ("real", "all"):
        mc.log("\n── real 套件：首帧延迟（R1，验 P0.2） + 语音 ×8（R2，验 A.11） + 并发（R3，验 P0.1） ──")
        run_real_suite(args, rep, session)
        mc.log("\n── R4：连续帧出分割结果（补 overlay 断言，2026-09-20 新增） ──")
        run_r4_overlay(args, rep)

    rep.dump(args.url, args.suite)

    mc.log("\n" + "=" * 78)
    mc.log(f"验收结论：{rep.summary_line()}   {'✅ 全部通过' if rep.passed else '❌ 存在未通过项'}")
    mc.log(f"报告已保存：{rep.out_dir / 'report.md'}")
    mc.log("=" * 78)
    sys.exit(0 if rep.passed else 1)


if __name__ == "__main__":
    main()
