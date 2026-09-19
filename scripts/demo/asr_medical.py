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
  - 幻觉防护（N1，2026-09-19）：hotwords 是压同音错字的功臣，但也让解码器在
    "音频含糊"时陷入复读循环（实测 cmd_02 复读词表 → 8.8~20s + 全乱码）。
    防护见"幻觉防护"小节：解码侧抑制 + 输出侧检测 + 时间兜底 → 降级重试 → 返空串。

接口与 asr_engine.transcribe 完全一致（server/api.py 只换一行 import）：
    from asr_medical import transcribe, warmup

依赖：faster-whisper（本地离线，不联网）。模型：medium，GPU float16 约 1.5GB
显存（RTX 5060 无压力）。稳态转写 400~500ms/句。
"""

from __future__ import annotations

import sys
import time
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


# ---------------------------------------------------------------------------
# 幻觉防护（N1，2026-09-19；契约 A.11 遗留项）
# ---------------------------------------------------------------------------
# 三层刹车：
#   ① 解码侧抑制：condition_on_previous_text=False（切断"上句→下句"的条件链，
#      复读循环的主要成因）+ repetition_penalty + no_repeat_ngram_size
#   ② 输出侧检测：n-gram 重复率 / 连续重复字符 / 文本长度与音频时长严重不符
#   ③ 时间兜底：逐段消费 segment 并计时，超时立即中止（避免 20s 卡住 A2 锁）
# 检出异常 → 降级重试（去掉 hotwords/prompt，复读诱因）→ 仍异常返回空串，
# 由上层走"没听清，请再说一遍"分支（契约 2003 语义）。宁可让医生说第二遍，
# 也不能把编造的目标交给 NLU。

_MAX_DECODE_SECONDS = 4.0          # 首次转写时间上限（正常 400~700ms，留足余量）
_MAX_DECODE_SECONDS_RETRY = 2.5    # 降级重试时间上限
_REPEAT_NGRAM_RATIO = 0.5          # 2~4 gram 重复占比 ≥ 该值判复读
_MAX_CHAR_RUN = 4                  # 连续相同字符数 ≥ 该值判复读（"肝肝肝肝"）
_CHARS_PER_SECOND_LIMIT = 8.0      # 每秒字符数上限（2.85s 音频正常 8~12 字）

# 解码侧防复读参数（当前 faster-whisper 版本不支持时会被自动剔除，见 _safe_transcribe）
_ANTI_REPEAT_KW = {
    "condition_on_previous_text": False,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "temperature": 0.0,
}


def _repeat_ratio(text: str, n: int) -> float:
    """n-gram 重复占比：1 - 去重后占比。文本过短返回 0。"""
    tokens = [text[i:i + n] for i in range(len(text) - n + 1)]
    if len(tokens) < 3:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def _max_char_run(text: str) -> int:
    """最长连续相同字符数。"""
    best = cur = 1
    for i in range(1, len(text)):
        cur = cur + 1 if text[i] == text[i - 1] else 1
        best = max(best, cur)
    return best if text else 0


def _looks_hallucinated(text: str, duration: float | None = None) -> bool:
    """判断转写结果是否是幻觉（复读/编造）。

    正常中文指令（如"把左心室腔标出来"共 8 字）三项指标都远离阈值；
    复读文本（"右心室 右心房 右心室 右心房…肝肝肝肝"）任一指标即触发。
    """
    if not text:
        return False
    for n in (2, 3, 4):
        if _repeat_ratio(text, n) >= _REPEAT_NGRAM_RATIO:
            return True
    if _max_char_run(text) >= _MAX_CHAR_RUN:
        return True
    if duration and duration > 0.5:
        if len(text) > _CHARS_PER_SECOND_LIMIT * duration + 10:
            return True
    return False


def _unexpected_kwarg(exc: TypeError) -> str | None:
    """从 TypeError 里解析出不被支持的参数名（'got an unexpected keyword argument'）。"""
    msg = str(exc)
    marker = "unexpected keyword argument"
    if marker not in msg:
        return None
    tail = msg.split(marker, 1)[1].strip()
    return tail.strip("'\" ") or None


def _safe_transcribe(model, audio, kwargs: dict):
    """调用转写；参数不被当前 faster-whisper 版本支持时自动剔除后重试。

    这样 _ANTI_REPEAT_KW 里的新参数（repetition_penalty 等）在旧版本上不会崩，
    只是少一层防护，其余逻辑照常。
    """
    attempt = dict(kwargs)
    for _ in range(len(attempt) + 1):
        try:
            return model.transcribe(audio, **attempt)
        except TypeError as exc:
            bad = _unexpected_kwarg(exc)
            if bad and bad in attempt:
                print(f"[ASR-medical] 当前 faster-whisper 不支持参数 {bad}，已忽略", flush=True)
                attempt.pop(bad)
                continue
            raise
    raise TypeError("transcribe 参数兼容处理失败")


def _run_once(model, audio, *, hotwords, initial_prompt, beam_size, extra,
              deadline_s: float):
    """跑一次转写，带时间兜底 + 幻觉检测。

    返回 (text, elapsed, reason)：text 为 None 表示判定异常（reason 说明原因）。
    """
    kwargs = {"language": "zh", "beam_size": beam_size, "vad_filter": True}
    if hotwords:
        kwargs["hotwords"] = hotwords
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    kwargs.update(extra)

    t0 = time.perf_counter()
    segments, info = _safe_transcribe(model, audio, kwargs)

    parts: list[str] = []
    timed_out = False
    # 惰性消费 segments：逐段计时，超时立即停止解码（复读循环会产生大量段）
    for seg in segments:
        if time.perf_counter() - t0 > deadline_s:
            timed_out = True
            break
        piece = seg.text.strip()
        if piece:
            parts.append(piece)
    elapsed = time.perf_counter() - t0

    text = "".join(parts)
    duration = getattr(info, "duration", None)
    if timed_out:
        return None, elapsed, f"超时>{deadline_s:.1f}s"
    if _looks_hallucinated(text, duration):
        return None, elapsed, f"复读幻觉(len={len(text)})"
    return text, elapsed, ""


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

    阶梯式策略（N1 幻觉防护）：
      ① 完整配置（medium + zh + 器官词表 prompt + hotwords + beam5 + 防复读参数）
      ② 检出复读/超时 → 降级重试（去 hotwords/prompt，贪心解码，它们不可用时
         由 beam1 兜住）——复读的主要诱因正是"强提示词 + 模糊音频"
      ③ 仍异常 → 返回空串（上层走"没听清，请再说一遍"，绝不给 NLU 编造文本）

    返回识别文本（字符串）；失败返回空串（与 asr_engine 行为一致）。
    """
    if audio is None:
        return ""
    try:
        model = _get_model()

        # ① 完整配置
        text, elapsed, why = _run_once(
            model, audio,
            hotwords=_HOTWORDS, initial_prompt=_INITIAL_PROMPT, beam_size=5,
            extra=_ANTI_REPEAT_KW, deadline_s=_MAX_DECODE_SECONDS,
        )
        if text is not None:
            return text
        print(f"[ASR-medical] 首次转写异常（{why}，{elapsed:.2f}s），降级重试…", flush=True)

        # ② 降级重试：去掉热词与提示词（复读诱因），贪心解码
        text2, elapsed2, why2 = _run_once(
            model, audio,
            hotwords=None, initial_prompt=None, beam_size=1,
            extra={"condition_on_previous_text": False, "temperature": 0.0},
            deadline_s=_MAX_DECODE_SECONDS_RETRY,
        )
        if text2 is not None:
            print(f"[ASR-medical] 降级重试成功（{elapsed2:.2f}s）：{text2!r}", flush=True)
            return text2

        # ③ 兜底：返空串，让医生重说一遍，而不是把幻觉交给 NLU
        print(f"[ASR-medical] 降级重试仍异常（{why2}，{elapsed2:.2f}s），"
              f"返回空串（上层提示重说）", flush=True)
        return ""
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

# 幻觉检测器单元用例（不需要模型，秒出）：(文本, 期望是否判为幻觉)
_DETECTOR_CASES = [
    ("把左心室腔标出来", False),                          # 正常指令
    ("帮我分割一下心肌区域", False),                       # 正常长句
    ("", False),                                        # 空串不是幻觉（走另一分支）
    ("肝肝肝肝肝肝", True),                                # 连续重复字符
    ("右心室 右心房 右心室 右心房 右心室 右心房", True),      # 词表复读循环
    ("右心室腔 右心室腔 右心室腔 右心室腔 右心室腔", True),    # 整词复读
]


def _detector_self_test() -> tuple[int, int]:
    """幻觉检测器单元自测（纯函数，秒出，不需要模型）。"""
    n_pass = 0
    print("-" * 72)
    print("幻觉检测器单元自测：")
    for text, expect in _DETECTOR_CASES:
        got = _looks_hallucinated(text)
        ok = got == expect
        n_pass += ok
        shown = text if len(text) <= 20 else text[:20] + "…"
        print(f"  [{'PASS' if ok else 'FAIL'}] {shown!r:28} 期望幻觉={expect}，实际={got}")
    return n_pass, len(_DETECTOR_CASES)


def _cmd_self_test(files, stress: int = 0) -> tuple[int, int]:
    """逐句转写 + NLU 解析（谐音回归）。stress>0 时对指定文件连跑 stress 次。"""
    from nl_parser import parse_target

    warmup()          # 先加载模型，避免把 30s 加载时间算进单句耗时
    n_pass = n_total = 0
    print("-" * 72)
    for fname in files:
        path = _SAMPLES_DIR / fname
        if not path.exists():
            print(f"[SKIP] {fname} 不存在")
            continue
        n_total += 1
        t0 = time.perf_counter()
        text = transcribe(str(path))
        elapsed = time.perf_counter() - t0
        r = parse_target(text)
        ok = bool(text) and r.ok
        n_pass += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {fname}  ({elapsed:.2f}s)")
        print(f"        转写: {text!r}")
        print(f"        解析: ok={r.ok} targets={r.targets}")
        if not ok:
            print(f"        提示: {r.message or '转写为空（幻觉防护已拦截，需重说）'}")

    if stress and files:
        target = files[0]
        print("-" * 72)
        print(f"压测：{target} 连跑 {stress} 次（验证幻觉不再复现）")
        ok_cnt = 0
        for i in range(stress):
            t0 = time.perf_counter()
            text = transcribe(str(_SAMPLES_DIR / target))
            elapsed = time.perf_counter() - t0
            r = parse_target(text)
            good = bool(text) and r.ok and elapsed < 3.0
            ok_cnt += good
            flag = "PASS" if good else "FAIL"
            print(f"  [{flag}] 第 {i + 1:2d} 次  {elapsed:.2f}s  {text!r}")
        n_pass += ok_cnt
        n_total += stress

    return n_pass, n_total


def _self_test(stress_file: str | None = None, stress: int = 0, only: str | None = None):
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 72)

    # 1) 检测器单元自测（秒出，先跑，便于快速验证防护逻辑本身）
    d_pass, d_total = _detector_self_test()
    if d_pass != d_total:
        print("=" * 72)
        print(f"检测器自测未通过 {d_pass}/{d_total}，先修检测逻辑，跳过模型自测")
        return 1

    if not _SAMPLES_DIR.is_dir():
        print(f"[ASR-medical] 未找到测试音频目录：{_SAMPLES_DIR}")
        print("[ASR-medical] 跳过模型自测（不影响 server 启动）。")
        return 0

    # 2) 逐句转写 + 解析
    files = [f"cmd_{int(only):02d}.wav"] if only else _CMD_FILES
    stress_files = [stress_file] if stress_file else []
    m_pass, m_total = _cmd_self_test(files, stress=0)
    if stress_files and stress:
        s_pass, s_total = _cmd_self_test(stress_files, stress=stress)
        m_pass += s_pass
        m_total += s_total

    print("=" * 72)
    total_pass = d_pass + m_pass
    total = d_total + m_total
    print(f"通过 {total_pass}/{total}（检测器 {d_pass}/{d_total}，模型 {m_pass}/{m_total}）")
    return 0 if total_pass == total else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ASR 医学谐音 + 幻觉防护自测")
    ap.add_argument("--only", help="只测某一句，填编号如 2 跑 cmd_02.wav")
    ap.add_argument("--stress", type=int, default=0,
                    help="对指定文件连跑次数（验证幻觉不再复现）")
    ap.add_argument("--stress-file", default="cmd_02.wav",
                    help="压测目标文件（默认 cmd_02.wav，A.11 已知幻觉样本）")
    a = ap.parse_args()
    sys.exit(_self_test(stress_file=a.stress_file, stress=a.stress, only=a.only))
