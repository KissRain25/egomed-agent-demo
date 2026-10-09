#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VAD 语音门槛回归测试（N37）—— 「噪声不误触发、指令仍 8/8」的自动化证据

做什么
------
把 `data/samples/audio/` 下的录音**逐条离线回放**给 `MicListener`（与真机同一套代码，
只是把麦克风回调换成文件分块喂入），统计：

  · 切出几段语音（`segments`）
  · 拒掉几段环境噪声（`rejected_noise`，N37 新增的两道门槛）

再按"期望"判 PASS/FAIL：

| 文件 | 期望 | 理由 |
|------|------|------|
| `cmd_01~08.wav` | **≥1 段** | 8 条真实指令必须照样能触发（不能为了防噪声把指令也拒了） |
| `voice_far.wav` | **≥1 段** | -20 dB 远场人声也必须能触发（门槛不能过严） |
| `noise/stationary.wav` | **0 段** | 稳态环境噪声（空调/风扇）不得触发 |
| `noise/clicks.wav` | **0 段** | 瞬时敲击（键盘/关门）不得触发 ← **N37 的关键回归** |
| `noise/crowd.wav` | 不判定 | 类语音的远处喧哗：能量 VAD 拦不住是正常的，该由服务端业务层拒识 |

`--legacy` 会**关掉 N37 的两道门槛**（等价改动前逻辑）跑同一批样例，
用来给出"改前 / 改后"对照 —— 这就是验收要的数字。

用法
----
    python scripts/make_noise_samples.py      # 先生成噪声样例（一次即可）
    python scripts/vad_regression.py          # 回归（默认自适应阈值，与真机一致）
    python scripts/vad_regression.py --legacy # 关掉 N37 门槛，看改动前的行为
    python scripts/vad_regression.py --threshold 0.005

退出码：0=全部符合期望，1=有不符合项

归属：闫（测试）｜ 关联：docs/总计划.md 阶段五 N37
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "client"))
import live_client as lc  # noqa: E402

AUDIO = PROJECT_ROOT / "data" / "samples" / "audio"


def read_wav16k(path: Path) -> np.ndarray:
    """读 wav → 16 kHz 单声道 float32（与 vad_selftest 同口径）。"""
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        n = int(len(a) * 16000 / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("float32")
    return a


def replay(path: Path, threshold, legacy: bool) -> tuple[int, dict, float]:
    """离线回放一条 wav，返回 (切出段数, mic 统计, 时长)。"""
    audio = read_wav16k(path)
    # 前置 0.6 s **该样例自身电平的房间底噪**（取自身块级 RMS 的 p20 作估计）：
    #   真机上麦克风是常开的，自适应噪声底是在前 0.6 s 的环境音里估出来的；
    #   而样例 wav 一开口就是人声/噪声 —— 直接回放会让噪声底估计失真
    #   （实测 cmd_07 被估到 0.0833、数字静音则让底噪估成 0，两种都不真实）。
    blocks = [audio[i:i + 1600] for i in range(0, len(audio) - 1599, 1600)]
    block_rms = np.array([float(np.sqrt(np.mean(b ** 2))) for b in blocks]) if blocks else np.array([1e-4])
    bg = max(float(np.percentile(block_rms, 20)), 2e-4)
    rng = np.random.default_rng(1234)
    pre = (rng.standard_normal(int(0.6 * 16000)) * bg).astype("float32")
    audio = np.concatenate([pre, audio])
    segs: list[bytes] = []
    kwargs = dict(threshold=threshold)
    if legacy:                    # 关掉 N37 的四道门槛 = 近似"改动前"的判定逻辑
        kwargs.update(min_active_ratio=0.0, margin=0.0,
                      contrast_min=0.0, min_active_sec=0.0)
    mic = lc.MicListener(lambda w, d: segs.append(w), **kwargs)
    block = 1600
    for i in range(0, len(audio), block):
        mic._on_block(audio[i:i + block])
    silence = np.zeros(int(1.2 * 16000), dtype="float32")     # 补静音让最后一句闭合
    for i in range(0, len(silence), block):
        mic._on_block(silence[i:i + block])
    mic.flush()
    return len(segs), dict(mic.stats), len(audio) / 16000


def collect_cases() -> list[tuple[Path, str, int | None]]:
    """(文件, 类别, 期望段数)；期望 None = 不判定。"""
    cases: list[tuple[Path, str, int | None]] = []
    for p in sorted(AUDIO.glob("cmd_*.wav")):
        cases.append((p, "指令", 1))
    far = AUDIO / "voice_far.wav"
    if far.exists():
        cases.append((far, "远场指令", 1))
    noise_dir = AUDIO / "noise"
    for p in sorted(noise_dir.glob("*.wav")):
        if p.stem == "crowd":
            cases.append((p, "喧哗（不判定）", None))
        else:
            cases.append((p, "噪声", 0))
    return cases


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="VAD 语音门槛回归（N37）")
    ap.add_argument("--threshold", type=float, default=None,
                    help="判定阈值（默认自适应 = 噪声底×4，与真机一致）")
    ap.add_argument("--legacy", action="store_true",
                    help="关掉 N37 的四道门槛 = 近似改动前的判定逻辑，用于对照")
    ap.add_argument("--no-gen", action="store_true",
                    help="噪声样例缺失时不要自动生成（默认会调 make_noise_samples.py 现生成）")
    args = ap.parse_args(argv)

    # 噪声/远场样例是**生成物**（`data/*` 被 .gitignore 忽略，不入库）→ 缺了就现生成
    noise_dir = AUDIO / "noise"
    need_gen = (not noise_dir.exists()) or (not list(noise_dir.glob("*.wav")))
    if need_gen and not args.no_gen:
        print(f"[准备] 噪声样例缺失 → 现生成（{noise_dir.relative_to(PROJECT_ROOT)}）")
        try:
            import make_noise_samples as mns     # 与 scripts/ 同目录
            mns.main(["--out", str(AUDIO)])
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 自动生成失败：{exc}（可手动跑 python scripts/make_noise_samples.py）")

    cases = collect_cases()
    if not cases:
        print(f"[错误] {AUDIO} 下没有样例；先跑 python scripts/make_noise_samples.py")
        return 2

    print("=" * 84)
    print(f"VAD 语音门槛回归（N37）　{'【legacy：已关 N37 门槛】' if args.legacy else '【默认门槛】'}"
          f"　阈值 {'自适应' if args.threshold is None else args.threshold}")
    print("=" * 84)
    print(f"{'文件':<28}{'类别':<16}{'期望':<8}{'段数':<6}{'拒识':<6}{'结果':<6}")
    print("-" * 84)

    n_fail = 0
    for path, kind, expect in cases:
        segs, st, dur = replay(path, args.threshold, args.legacy)
        if expect is None:
            verdict = "—（不判定）"
        elif expect == 0:
            ok = segs == 0
            verdict = "PASS" if ok else "FAIL"
            n_fail += 0 if ok else 1
        else:
            ok = segs >= expect
            verdict = "PASS" if ok else "FAIL"
            n_fail += 0 if ok else 1
        name = str(path.relative_to(AUDIO)).replace("\\", "/")
        exp_txt = "—" if expect is None else ("0 段" if expect == 0 else f"≥{expect} 段")
        print(f"{name:<28}{kind:<16}{exp_txt:<8}{segs:<6}"
              f"{st.get('rejected_noise', 0):<6}{verdict:<6}")

    print("-" * 84)
    n_judged = sum(1 for _, _, e in cases if e is not None)
    print(f"判定 {n_judged} 例 → 通过 {n_judged - n_fail} / 失败 {n_fail}"
          f"{'　（legacy 模式仅用于对照，不代表现状）' if args.legacy else ''}")
    print("=" * 84)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
