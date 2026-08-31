#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoMed-Agent 语音识别模块（ASR）
================================
把用户的语音转成文字，再复用 nl_parser.parse_target() 解析成分割目标，
实现"开口说话 -> 自动识别器官 -> 分割"。

用法：
    from asr_engine import transcribe, speech_to_target
    text = transcribe(audio_path)          # 语音 -> 文字
    r = speech_to_target(audio_path)       # 语音 -> 目标(parse_target 结果)

依赖：faster-whisper（本地离线，不联网，数据不出本机）
模型：whisper small（中文/英文都支持，速度/精度平衡，GPU 加速）
"""

import sys
import tempfile
from pathlib import Path

# faster-whisper 延迟导入，避免 demo 没装时拖累启动
_MODEL = None
_MODEL_NAME = None

# 默认模型大小：small（中文识别好）。可选 base / small / medium
DEFAULT_MODEL = "small"

# 允许的音频扩展名
_ALLOWED_EXT = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".webm"}


def _get_model(size=DEFAULT_MODEL):
    """延迟加载 faster-whisper 模型，全局只加载一次。"""
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != size:
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[ASR] 加载 Whisper({size}) on {device} ({compute_type}) ...", flush=True)
        _MODEL = WhisperModel(size, device=device, compute_type=compute_type)
        _MODEL_NAME = size
    return _MODEL


def transcribe(audio, language=None, size=DEFAULT_MODEL):
    """把音频转成文字。audio 可以是路径、临时文件对象，或 numpy 波形。

    返回识别文本（字符串）。失败返回空串。
    """
    if audio is None:
        return ""
    try:
        model = _get_model(size)
        # faster-whisper 接受文件路径或 file 对象；Gradio 传的可能是临时文件路径
        segments, info = model.transcribe(audio, language=language)
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        text = "".join(parts)
        return text
    except Exception as e:
        print(f"[ASR] 识别失败: {e}", flush=True)
        return ""


def speech_to_target(audio, size=DEFAULT_MODEL):
    """语音 -> 目标。返回 (text, ParseResult)。

    如果音频输入是 (sr, numpy) 元组（Gradio 录音格式），先转成 wav 文件。
    """
    from nl_parser import parse_target

    # 处理 Gradio 录音元组：<class 'tuple'> (sample_rate, numpy_array)
    path_to_delete = None
    if isinstance(audio, tuple) and len(audio) == 2 and audio[0] is not None:
        import numpy as np
        import scipy.io.wavfile as wavfile
        sr, data = audio
        if isinstance(data, np.ndarray):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            path_to_delete = tmp.name
            wavfile.write(tmp.name, int(sr), data)
            audio = tmp.name

    try:
        text = transcribe(audio, size=size)
        if not text:
            return text, None
        r = parse_target(text)
        return text, r
    finally:
        if path_to_delete and Path(path_to_delete).exists():
            try:
                Path(path_to_delete).unlink()
            except Exception:
                pass


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # 自测：如果传了音频文件路径则识别，否则提示
    if len(sys.argv) > 1:
        t, r = speech_to_target(sys.argv[1])
        print("识别文本:", t)
        if r:
            print("解析:", r.message, "| 目标:", r.targets)
    else:
        print("用法: python asr_engine.py <音频文件.wav>")
