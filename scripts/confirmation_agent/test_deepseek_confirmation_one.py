#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
from openai import OpenAI


def load_dotenv(path):
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


load_dotenv(f"{REPO_ROOT}/.env")

api_key = os.environ.get("DEEPSEEK_API_KEY", "")
base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY is empty in .env")

print("API key loaded:", api_key[:3] + "***")
print("Base URL:", base_url)
print("Model:", model_name)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

instruction = "Segment the kidney"
candidate_targets = ["left kidney", "right kidney"]

prompt = f"""
You are the Confirmation Agent in an interactive egocentric medical image segmentation system.

Your task is to ground a user's instruction to one of the detected candidate medical targets.

Input:
- User instruction: "{instruction}"
- Candidate targets: {candidate_targets}

Return a JSON object with:
- matched_target: the most likely target from the candidate list. It must be exactly one of the candidates.
- score: a confidence score from 0 to 1.
- reason: a short explanation.

Scoring rule:
- Use a high score, 0.80 to 1.00, if the instruction uniquely specifies one candidate, such as "left kidney" or "LV cavity".
- Use a medium score, 0.50 to 0.79, if one candidate is likely but not fully explicit.
- Use a low score, below 0.50, if the instruction is ambiguous among multiple candidates, such as "kidney" when both left kidney and right kidney exist.
- If the instruction is ambiguous, still choose the best matched_target, but give a low score.

Return only valid JSON. Do not include markdown.
""".strip()

resp = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)

text = resp.choices[0].message.content.strip()

print("\nraw response:")
print(text)

data = json.loads(text)

print("\nparsed:")
print(data)
print("\nmatched_target:", data.get("matched_target"))
print("score:", data.get("score"))
print("reason:", data.get("reason"))
