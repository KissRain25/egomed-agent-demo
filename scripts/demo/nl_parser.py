#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoMed-Agent 自然语言目标解析模块（NLU）
========================================
把用户的一句话（中/英文）解析成可分割的目标类别，例如：

    "帮我分割左心室"         -> ["LV cavity"]
    "把右心室腔标出来"       -> ["RV cavity"]
    "分割心肌"               -> ["myocardium"]
    "left ventricle please"  -> ["LV cavity"]
    "分割左心室和心肌"       -> ["LV cavity", "myocardium"]  （多目标）

设计原则：
- 零依赖（只用标准库），离线可用，单次解析微秒级
- 规则 + 同义词词典：当前目标类别少（3 个），规则足够准确且可解释（答辩友好）
- 以后接语音识别（ASR）时，只需把语音转成文字，再调用 parse_target() 即可复用

用法：
    from nl_parser import parse_target
    r = parse_target("帮我分割左心室")
    r.ok        # True
    r.targets   # ["LV cavity"]
    r.message   # 给用户看的提示
"""

import difflib
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------

# 所有可分割目标（标准名 = 各模态 YOLO 权重里的类名，必须与 egomed_demo.py 的 name_to_gray 键一致）
SUPPORTED_TARGETS = (
    # ACDC (MRI 心脏)
    "LV cavity", "RV cavity", "myocardium",
    # CAMUS (超声 心脏)
    "Left Ventricular Myocardium", "Left Ventricle", "Left Atrium",
    # Amos (CT 腹部)
    "spleen", "right kidney", "left kidney", "gall bladder", "esophagus",
    "liver", "stomach", "arota", "postcava", "pancreas",
    "right adrenal gland", "left adrenal gland", "duodenum", "bladder",
    "prostate/uterus",
    # Montgomery (X光 胸片)
    "left lung", "right lung",
    # PolypGen (内窥镜)
    "polyp",
)

# 每个目标的中英文同义词/口语说法
# 注意：英文短别名（lv / rv）用词边界匹配，避免误命中（如 review 里的 rv）
TARGET_ALIASES = {
    # ---------- ACDC (MRI 心脏) ----------
    "LV cavity": [
        "左心室腔", "左心室", "左室", "左心室腔体", "左心腔",
        "left ventricle", "left ventricular", "left heart chamber",
        "lv cavity", "lv",
    ],
    "RV cavity": [
        "右心室腔", "右心室", "右室", "右心室腔体", "右心腔",
        "right ventricle", "right ventricular", "right heart chamber",
        "rv cavity", "rv",
    ],
    "myocardium": [
        "心肌", "心肌层", "心肌组织", "心壁", "心脏肌肉",
        "myocardium", "myocardial", "heart muscle",
    ],
    # ---------- CAMUS (超声 心脏) ----------
    "Left Ventricular Myocardium": [
        "左室心肌", "左心室心肌", "左室壁", "左心室壁",
        "left ventricular myocardium", "lvm",
    ],
    "Left Ventricle": [
        "左心室", "左室腔", "左心室腔",
        "left ventricle",
    ],
    "Left Atrium": [
        "左心房", "左房",
        "left atrium",
    ],
    # ---------- Amos (CT 腹部) ----------
    "spleen": ["脾", "脾脏", "spleen"],
    "right kidney": ["右肾", "右侧肾脏", "right kidney"],
    "left kidney": ["左肾", "左侧肾脏", "left kidney"],
    "gall bladder": ["胆囊", "gall bladder", "gallbladder"],
    "esophagus": ["食管", "食道", "esophagus", "oesophagus"],
    "liver": ["肝", "肝脏", "肝部", "liver"],
    "stomach": ["胃", "胃部", "stomach"],
    "arota": ["主动脉", "aorta", "arota"],
    "postcava": ["下腔静脉", "postcava", "inferior vena cava", "ivc"],
    "pancreas": ["胰腺", "胰", "pancreas"],
    "right adrenal gland": ["右肾上腺", "right adrenal gland", "right adrenal"],
    "left adrenal gland": ["左肾上腺", "left adrenal gland", "left adrenal"],
    "duodenum": ["十二指肠", "duodenum"],
    "bladder": ["膀胱", "bladder"],
    "prostate/uterus": ["前列腺", "子宫", "prostate", "uterus"],
    # ---------- Montgomery (X光 胸片) ----------
    "left lung": ["左肺", "left lung"],
    "right lung": ["右肺", "right lung"],
    # ---------- PolypGen (内窥镜) ----------
    "polyp": ["息肉", "肠息肉", "polyp", "polyps"],
}

# 否定词：句子中一旦出现，就把对应的目标从结果中排除。
# 例："不要分割右心室，分割左心室" -> 排除 RV，保留 LV
NEGATION_WORDS = [
    "不要", "别", "除了", "排除", "去掉", "不算", "忽略", "不管", "不分割",
    "except", "without", "excluding", "not",
]

# 多目标连接词（用于判断用户是否想一次分割多个结构）
CONJUNCTIONS = ["和", "与", "及", "还有", "以及", "、", "and", "plus", "both"]

# 常见歧义/太宽泛的说法 -> 引导用户说清楚
AMBIGUOUS_HINTS = {
    "心室": '你说的是左心室还是右心室？可以这样说："分割左心室" 或 "分割右心室"',
    "心腔": '你说的是左心腔还是右心腔？可以这样说："分割左心室" 或 "分割右心室"',
    "心脏": '心脏太宽泛了，请指定具体结构，比如："分割左心室"、"分割心肌"',
}

# ---------------------------------------------------------------------------
# 解析结果
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    targets: list = field(default_factory=list)   # 解析出的目标列表（标准名）
    ok: bool = False                              # 是否解析成功
    message: str = ""                             # 给用户/界面看的提示

# ---------------------------------------------------------------------------
# 底层工具
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """小写、把标点/多余空白归一化为单个空格。"""
    text = text.lower().strip()
    text = re.sub(r"[，。！？!?；;：:\"'、,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_cjk(word: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", word))


def _contains(text: str, keyword: str) -> bool:
    """中文关键字用子串匹配；英文关键字用词边界匹配（避免 love 命中 lv）。"""
    if _is_cjk(keyword):
        return keyword in text
    return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None


def _collect_hits(norm: str) -> set:
    """收集命中的目标，并用"最长别名优先"消歧。

    当一个说法同时命中多个目标（如"左心室"同时是 MRI 的 LV cavity 和
    超声的 Left Ventricle 的别名），只保留"命中别名最长"的那个目标，
    避免多目标冲突。例如：
      - "左心室" -> 命中多个 LV，都按别名"左心室"(3字) 平级，需进一步判断
      - "左心室心肌" -> 命中 "Left Ventricular Myocardium"（别名5字），
        比"左心室"更具体，优先它
    """
    # 先收集所有 (目标, 命中的最长别名长度, 最长别名)
    matches = {}  # target -> (matched_alias_len, matched_alias)
    for target, aliases in TARGET_ALIASES.items():
        best_alias = None
        for a in aliases:
            if _contains(norm, a):
                if best_alias is None or len(a) > len(best_alias):
                    best_alias = a
        if best_alias is not None:
            matches[target] = (len(best_alias), best_alias)

    # 同一别名被多个目标共享时，取匹配别名最长者；若仍并列，取定义靠前者
    # （保持可预测）。这里把"命中别名长度"作为消歧主键。
    if not matches:
        return set()
    max_len = max(l for l, _ in matches.values())
    hits = {t for t, (l, _) in matches.items() if l == max_len}

    # 若并列命中多个（如"左心室"同时命中 LV cavity / Left Ventricle /
    # Left Ventricular Myocardium，且别名长度一样），此时无法用别名区分，
    # 交由上层结合模态判断：这里返回全部候选。
    return hits


def _has_negation(norm: str) -> bool:
    return any(_contains(norm, w) for w in NEGATION_WORDS)


def _apply_negation(norm: str, hits: set) -> set:
    """把"被否定词修饰"的目标从 hits 中排除。

    判定规则：否定词出现在某目标别名之前很近的位置（后面 30 个字符内
    出现该目标的别名），才认为这个目标被否定。
    例："我要分割右心室，不要左心室"
        -> "不要" 之后紧跟 "左心室" -> 排除 LV，保留 RV
    """
    if not hits or not _has_negation(norm):
        return hits
    negated = set()
    for target in hits:
        aliases = TARGET_ALIASES[target]
        for word in NEGATION_WORDS:
            if not _contains(norm, word):
                continue
            for m in re.finditer(re.escape(word), norm):
                tail = norm[m.end():m.end() + 30]
                if any(_contains(tail, a) for a in aliases):
                    negated.add(target)
                    break
            if target in negated:
                break
    return hits - negated


def _fuzzy_match(norm: str):
    """容错匹配：短语级 + 单词级相似度，用于英文拼写错误等。

    返回 (best_target, best_score)。
    """
    best, best_score = None, 0.0
    words = norm.split()
    for target, aliases in TARGET_ALIASES.items():
        for a in aliases:
            # 短语级：整句 vs 整个别名
            s = difflib.SequenceMatcher(None, norm, a).ratio()
            if s > best_score:
                best, best_score = target, s
            # 单词级：输入词 vs 别名单词（容忍拼写错误）
            for w in words:
                for aw in a.split():
                    s = difflib.SequenceMatcher(None, w, aw).ratio()
                    if s > best_score:
                        best, best_score = target, s
    return best, best_score


# ---------------------------------------------------------------------------
# 主解析入口
# ---------------------------------------------------------------------------

def parse_target(text: str) -> ParseResult:
    """把自然语言解析为分割目标。

    返回 ParseResult：
      ok=True, targets=[...]  -> 解析成功（可能多目标）
      ok=False                -> 无法解析（message 里给原因和建议）
    """
    if not text or not text.strip():
        return ParseResult(
            ok=False,
            message='请告诉我你想分割什么，例如："帮我分割左心室"',
        )

    norm = _normalize(text)

    # 1. 收集所有命中的目标
    hits = _collect_hits(norm)

    # 2. 否定词排除（只排除被否定词修饰的目标）
    had_hits = bool(hits)
    hits = _apply_negation(norm, hits)
    if had_hits and not hits:
        return ParseResult(
            ok=False,
            message='你似乎不想要任何目标被分割。请明确告诉我分割哪个结构，例如："分割左心室"',
        )

    # 3. 命中处理
    if len(hits) == 1:
        t = list(hits)[0]
        return ParseResult(
            targets=[t], ok=True,
            message=f"已识别目标：{t}",
        )

    if len(hits) > 1:
        ordered = [t for t in SUPPORTED_TARGETS if t in hits]
        return ParseResult(
            targets=ordered, ok=True,
            message=f"识别到多个目标：{'、'.join(ordered)}（当前一次分割一个，请明确指定一个）",
        )

    # 4. 未命中：先看是否说得很宽泛/有歧义
    for word, hint in AMBIGUOUS_HINTS.items():
        if word in norm:
            return ParseResult(ok=False, message=hint)

    # 5. 模糊匹配（英文拼写错误，如 "left ventrical" -> "LV cavity"）
    best, best_score = _fuzzy_match(norm)
    if best_score >= 0.8:
        return ParseResult(
            targets=[best], ok=True,
            message=f"你的输入接近 「{best}」（相似度 {best_score:.0%}），按此处理",
        )
    if best_score >= 0.6:
        return ParseResult(
            ok=False,
            message=f'你是想说「{best}」吗？可以直接输入 {best}，或换一种说法。',
        )

    # 6. 完全没听懂
    return ParseResult(
        ok=False,
        message=(
            f'抱歉，没听懂 "{text.strip()}"。\n'
            f"目前支持的分割目标：{', '.join(SUPPORTED_TARGETS)}\n"
            "你可以这样说：\n"
            '  · 帮我分割左心室\n'
            '  · 分割右心室腔\n'
            '  · 把心肌标出来\n'
            '  · 或直接输入 LV cavity / RV cavity / myocardium'
        ),
    )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

def _self_test():
    cases = [
        # (输入, 期望目标列表 or None, 说明)
        # 明确带模态区分的说法 -> 唯一命中
        ("把右心室腔标出来", ["RV cavity"], "中文口语-RV"),
        ("左室心肌", ["Left Ventricular Myocardium"], "中文-超声心肌"),
        ("分割左心房", ["Left Atrium"], "中文-左心房"),
        ("帮我分割肝脏", ["liver"], "中文-CT肝"),
        ("把右肾标出来", ["right kidney"], "中文-CT右肾"),
        ("分割脾脏", ["spleen"], "中文-CT脾"),
        ("分割右肺", ["right lung"], "中文-X光右肺"),
        ("帮我把息肉标出来", ["polyp"], "中文-内窥镜"),
        ("segment the right ventricular cavity", ["RV cavity"], "英文-RV"),
        ("segment the liver please", ["liver"], "英文-CT肝"),
        ("right lung", ["right lung"], "英文-X光右肺"),
        ("polyp", ["polyp"], "英文-内窥镜"),
        ("LV", ["LV cavity"], "缩写"),
        ("rv", ["RV cavity"], "缩写"),
        ("我要分割右心室，不要左心室", ["RV cavity"], "否定排除"),
        # 歧义/多候选（无模态上下文时）
        ("帮我分割左心室", ["LV cavity", "Left Ventricle"], "多候选：左心室(2模态)"),
        ("分割心肌", ["myocardium"], "消歧：仅MRI心肌"),
        ("left ventricle please", ["LV cavity", "Left Ventricle"], "多候选：左心室(英文)"),
        ("帮我看看心室", None, "歧义：心室"),
        ("请分割心脏", None, "歧义：心脏"),
        ("请分割肺", None, "歧义：左/右肺"),
        ("left ventrical", ["LV cavity"], "拼写纠错"),
        ("", None, "空输入"),
    ]
    print("=" * 60)
    n_ok = 0
    for text, expect, note in cases:
        r = parse_target(text)
        got = r.targets if r.ok else None
        passed = got == expect
        n_ok += passed
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {note:18s} | 输入: {text!r}")
        print(f"        -> ok={r.ok} targets={got}")
        if not passed:
            print(f"        -> 期望: {expect}")
            print(f"        -> 提示: {r.message}")
    print("=" * 60)
    print(f"通过 {n_ok}/{len(cases)}")
    return n_ok == len(cases)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
