#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoMed-Agent 医学语音识别模块（ASR 优化版，契约附录 A.11）
==========================================================
解决 asr_engine 的同音错字问题：中文医学词汇（左心室 / 心肌 / 息肉 / 脾脏 /
左肺 …）在同音字上频繁识别错，导致 NLU 解析失败（实测 8 句仅 1/8 可解析）。

优化点（vs asr_engine）：
  - 模型：small -> medium（中文医学词识别更准）
  - language 强制 "zh"（短音频语言误判是错字根源之一）
  - initial_prompt：器官词表提示词，把解码偏向医学用语
  - hotwords：热词偏置，进一步压住同音错字
  - beam_size=5：beam 搜索更稳
  - vad_filter：跳过静音段，减少噪声干扰

接口与 asr_engine.transcribe 完全一致（server/api.py 只换一行 import）：
    from asr_medical import transcribe, warmup

依赖：faster-whisper（本地离线，不联网）。模型：medium，GPU float16 约 1.5GB
显存（RTX 5060 无压力）。稳态转写 400~500ms/句。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 延迟导入 faster-whisper：没装它时 import 本模块不报错（server 启动不崩），
# 首次 transcribe/warmup 才真正加载。
_MODEL = None

# 与 asr_engine 同名常量，保证 `from ... import` 的签名兼容（size 参数忽略，
# 本模块固定 medium）。
DEFAULT_MODEL = "medium"

# 器官词表（来自 nl_parser.SUPPORTED_TARGETS 的中文别名，医用高频词）。
# 同时用于 initial_prompt 与 hotwords，把解码偏向医学用语、压住同音错字。
_ORGAN_VOCAB = (
    "左心室 右心室 心肌 左心房 左室 右室 左心腔 右心腔 "
    "脾脏 肝脏 胆囊 胰腺 胃 食管 食道 十二指肠 膀胱 主动脉 下腔静脉 "
    "左肾 右肾 肾脏 左肾上腺 右肾上腺 前列腺 子宫 "
    "左肺 右肺 息肉 肠息肉 "
)

# initial_prompt：给解码器的上下文，提示这是医生语音分割指令
_INITIAL_PROMPT = (
    "以下是医生在影像/手术场景下的语音分割指令，请逐字准确转写中文医学术语："
    + _ORGAN_VOCAB
)

# hotwords：热词偏置（faster-whisper 会用这些词做解码偏置）
_HOTWORDS = _ORGAN_VOCAB


def _get_model():
    """延迟加载 medium 模型，全局只加载一次（约 30s）。"""
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[ASR-medical] 加载 Whisper(medium) on {device} ({compute_type}) ...",
              flush=True)
        _MODEL = WhisperModel(DEFAULT_MODEL, device=device, compute_type=compute_type)
    return _MODEL


def transcribe(audio, language=None, size=DEFAULT_MODEL):
    """把音频转成文字。签名与 asr_engine.transcribe 完全一致。

    固定中文 + 器官词表提示词 + 热词 + beam5 + VAD。
    返回识别文本（字符串）；失败返回空串（与 asr_engine 行为一致）。
    """
    if audio is None:
        return ""
    try:
        model = _get_model()
        segments, _info = model.transcribe(
            audio,
            language="zh",              # 强制中文，避免短音频语言误判
            beam_size=5,                # beam 搜索
            initial_prompt=_INITIAL_PROMPT,
            hotwords=_HOTWORDS,
            vad_filter=True,            # 跳过静音，减少噪声干扰
        )
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return "".join(parts)
    except Exception as exc:  # noqa: BLE001 —— 失败返回空串，由上层做优雅降级
        print(f"[ASR-medical] 识别失败: {exc}", flush=True)
        return ""


def warmup():
    """预热：把 medium 模型加载进显存（--real 启动时调用一次）。

    不预热的话，首个 A2 请求会把 30s 的模型加载算进请求耗时，必超契约
    3s 超时（与 P0.2 同理）。
    """
    _get_model()


# ---------------------------------------------------------------------------
# 自测（通知第五节验收方式 #1，期望 8/8 PASS）
# ---------------------------------------------------------------------------
# 8 句医学指令，cmd_03（"分割心肌"）是谐音回归用例（原配置识别成"刑機"）。
# 判定标准：转写非空 + nl_parser 能解析出目标（谐音错字"皮杖/西肉/心机"等
# 都不在词表里，parse 必然失败，因此"解析成功"即等价于"字准达标"）。
_SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "samples" / "audio"
_CMD_FILES = [f"cmd_{i:02d}.wav" for i in range(1, 9)]


def _self_test():
    sys.stdout.reconfigure(encoding="utf-8")
    from nl_parser import parse_target

    if not _SAMPLES_DIR.is_dir():
        print(f"[ASR-medical] 未找到测试音频目录：{_SAMPLES_DIR}")
        print("[ASR-medical] 跳过自测（不影响 server 启动）。")
        return 0

    n_pass = 0
    n_total = 0
    print("=" * 72)
    for fname in _CMD_FILES:
        path = _SAMPLES_DIR / fname
        if not path.exists():
            print(f"[SKIP] {fname} 不存在")
            continue
        n_total += 1
        text = transcribe(str(path))
        r = parse_target(text)
        ok = bool(text) and r.ok
        n_pass += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {fname}")
        print(f"        转写: {text!r}")
        print(f"        解析: ok={r.ok} targets={r.targets}")
        if not ok:
            print(f"        提示: {r.message}")
    print("=" * 72)
    print(f"通过 {n_pass}/{n_total}")
    return 0 if (n_total > 0 and n_pass == n_total) else 1


if __name__ == "__main__":
    sys.exit(_self_test())
