#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成"噪声与远场"语音样例（N37 用）—— 可复现，不依赖外部素材

为什么需要合成
--------------
N37 的验收标准是「**噪声样例不误触发、指令仍 8/8**」。手头只有 8 条干净的指令录音
（`data/samples/audio/cmd_01~08.wav`），**没有噪声素材**，也没法把实验室环境录下来
放进仓库。所以这里用确定性算法合成三类典型干扰 + 一条"远场指令"：

  noise/stationary.wav  稳态环境噪声（空调/风扇：粉噪 + 50 Hz 哼声）
  noise/clicks.wav      瞬态敲击（键盘/关门：低幅背景 + 每 0.4s 一次 20ms 脉冲）
                         —— 这条是**关键回归用例**：旧逻辑（只看"有没有块超阈值"）
                            会把它当成一句话发出去，N37 的"活跃占比"门槛应当拒掉它
  noise/crowd.wav       远处喧哗（幅度调制噪声）—— 类语音，**允许**被切成段，
                        真正该拦住它的是服务端识别层（业务拒识），故期望标为"不判定"
  voice_far.wav         cmd_03 衰减 -20 dB（模拟 3 米外说话）—— 期望**仍能**触发，
                        用来防止门槛调得过严而把远场人声也拒了

用法
----
    python scripts/make_noise_samples.py                 # 写入 data/samples/audio/
    python scripts/make_noise_samples.py --out runs/x    # 写到别处
    python scripts/make_noise_samples.py --seed 7        # 换一组随机数

输出：`data/samples/audio/noise/*.wav` + `data/samples/audio/voice_far.wav`（16 kHz 单声道 PCM16）

归属：闫（测试素材）｜ 关联：docs/总计划.md 阶段五 N37、scripts/vad_regression.py
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SR = 16000


def _write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"  {path.relative_to(PROJECT_ROOT)}  {len(audio)/SR:.2f}s  "
          f"RMS {rms:.4f}  {path.stat().st_size/1024:.0f} KB")


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """一阶低通白噪 ≈ 粉噪（环境噪声的主体成分）。"""
    white = rng.standard_normal(n)
    out = np.empty(n, dtype="float64")
    acc = 0.0
    for i, x in enumerate(white):     # 简单 IIR，n 只有几万，够快
        acc = 0.97 * acc + 0.03 * x
        out[i] = acc
    peak = np.max(np.abs(out)) or 1.0
    return out / peak


def make_stationary(dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    base = _pink(n, rng) * 0.010                      # 环境底噪
    hum = 0.002 * np.sin(2 * np.pi * 50 * np.arange(n) / SR)   # 50 Hz 电源哼声
    return (base + hum).astype("float32")


def make_clicks(dur: float, rng: np.random.Generator, every: float = 0.4) -> np.ndarray:
    """低幅背景 + 周期性 20ms 脉冲（敲键盘/关门）。"""
    n = int(dur * SR)
    audio = _pink(n, rng) * 0.003
    pulse_len = int(0.02 * SR)
    t = every
    while t < dur - 0.1:
        i = int(t * SR)
        env = np.hanning(pulse_len)
        audio[i:i + pulse_len] += 0.45 * env * rng.standard_normal(pulse_len) / 3.0
        t += every
    return audio.astype("float32")


def make_crowd(dur: float, rng: np.random.Generator) -> np.ndarray:
    """远处喧哗：粉噪 + 3 Hz 幅度调制（类语音起伏）。"""
    n = int(dur * SR)
    base = _pink(n, rng)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * np.arange(n) / SR)
    return (base * env * 0.020).astype("float32")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
        ch = w.getnchannels()
    a = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    return a.reshape(-1, ch).mean(axis=1) if ch > 1 else a


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="生成噪声/远场语音样例（N37）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "samples" / "audio"),
                    help="输出目录（默认 data/samples/audio）")
    ap.add_argument("--seed", type=int, default=20261009, help="随机种子（默认固定，保证可复现）")
    ap.add_argument("--cmd", default="cmd_03.wav", help="用来派生远场样例的指令录音")
    ap.add_argument("--far-gain-db", type=float, default=-20.0, help="远场衰减（dB，默认 -20）")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    noise_dir = out / "noise"
    rng = np.random.default_rng(args.seed)

    print("=" * 74)
    print(f"生成噪声/远场样例 → {out.relative_to(PROJECT_ROOT)}（seed={args.seed}）")
    print("=" * 74)
    print("[噪声]")
    _write(noise_dir / "stationary.wav", make_stationary(3.0, rng))
    _write(noise_dir / "clicks.wav", make_clicks(2.4, rng))
    _write(noise_dir / "crowd.wav", make_crowd(3.0, rng))

    print("[远场指令]")
    src = out / args.cmd
    if not src.exists():
        print(f"  [跳过] 找不到 {src}（--cmd 指定现有录音）")
        return 1
    a = read_wav(src) * (10 ** (args.far_gain_db / 20.0))
    _write(out / "voice_far.wav", a)
    print("=" * 74)
    print("完成。下一步：python scripts/vad_regression.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
