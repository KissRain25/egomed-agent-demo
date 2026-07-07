#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import re
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


INPUT_CSV = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_full.csv")
OUT_DIR = Path(f"{REPO_ROOT}/runs/target_confirmation_eval/full_llm_deepseek")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV = OUT_DIR / "predictions.csv"
SUMMARY_CSV = OUT_DIR / "summary_tau06.csv"
THRESHOLD_CSV = OUT_DIR / "threshold_analysis.csv"

TAU1 = 0.6
MAX_RETRY = 3
SLEEP_BETWEEN_CALLS = 0.2


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


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = x.replace("_", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def split_candidates(candidate_str):
    if pd.isna(candidate_str):
        return []
    return [c.strip() for c in str(candidate_str).split(";") if c.strip()]


def target_match(pred, gt):
    return normalize_text(pred) == normalize_text(gt)


def extract_json(text):
    text = text.strip()

    # remove markdown fence if any
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # fallback: extract first {...}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"Cannot parse JSON from: {text}")


def resolve_with_clarification(clarification_response, candidates):
    clar = normalize_text(clarification_response)
    if not clar:
        return ""

    # direct candidate mention
    for cand in sorted(candidates, key=lambda x: len(normalize_text(x)), reverse=True):
        if normalize_text(cand) in clar:
            return cand

    # side-based
    if "left" in clar:
        for cand in candidates:
            cn = normalize_text(cand)
            if "left" in cn or cn.startswith("lv"):
                return cand

    if "right" in clar:
        for cand in candidates:
            cn = normalize_text(cand)
            if "right" in cn or cn.startswith("rv"):
                return cand

    if clar in ["lv", "left ventricle"]:
        for cand in candidates:
            cn = normalize_text(cand)
            if cn in ["lv cavity", "left ventricle"] or "left ventricle" in cn:
                return cand

    if clar in ["rv", "right ventricle"]:
        for cand in candidates:
            cn = normalize_text(cand)
            if cn in ["rv cavity", "right ventricle"] or "right ventricle" in cn:
                return cand

    # organ names
    for cand in candidates:
        if normalize_text(cand) == clar:
            return cand

    return ""


def build_prompt(instruction, candidates):
    return f"""
You are the Confirmation Agent in an interactive egocentric medical image segmentation system.

Your task is to ground a user's instruction to one of the detected candidate medical targets.

Input:
- User instruction: "{instruction}"
- Candidate targets: {candidates}

Return a JSON object with:
- matched_target: the most likely target from the candidate list. It must be exactly one of the candidates.
- score: a confidence score from 0 to 1.
- reason: a short explanation.

Scoring rule:
- Use a high score, 0.80 to 1.00, if the instruction uniquely specifies one candidate, such as "left kidney", "right lung", "LV cavity", or "polyp".
- Use a medium score, 0.50 to 0.79, if one candidate is likely but the instruction is not fully explicit.
- Use a low score, below 0.50, if the instruction is ambiguous among multiple candidates, such as "kidney" when both left kidney and right kidney exist, "lung" when both left lung and right lung exist, or "chamber" when multiple cardiac chambers exist.
- If the instruction is ambiguous, still choose the best matched_target, but give a low score.
- Do not invent targets outside the candidate list.

Return only valid JSON. Do not include markdown.
""".strip()


def call_confirmation_agent(client, model_name, instruction, candidates):
    prompt = build_prompt(instruction, candidates)

    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            data = extract_json(text)

            matched = str(data.get("matched_target", "")).strip()
            score = float(data.get("score", 0.0))
            reason = str(data.get("reason", "")).strip()

            # force matched target to be one of candidates
            cand_norm = {normalize_text(c): c for c in candidates}
            if normalize_text(matched) in cand_norm:
                matched = cand_norm[normalize_text(matched)]
            else:
                # fallback to first candidate to avoid invalid target
                reason = f"[INVALID_TARGET_FIXED] raw={matched}. " + reason
                matched = candidates[0] if candidates else ""

            score = max(0.0, min(1.0, score))

            return {
                "llm_raw": text,
                "matched_target": matched,
                "score": score,
                "reason": reason,
                "api_error": "",
            }

        except Exception as e:
            last_err = repr(e)
            time.sleep(1.0 * attempt)

    return {
        "llm_raw": "",
        "matched_target": candidates[0] if candidates else "",
        "score": 0.0,
        "reason": "",
        "api_error": last_err or "unknown error",
    }


def compute_metrics(pred_df, tau):
    df = pred_df.copy()

    df["pred_state"] = df["score"].apply(lambda x: "unique" if float(x) >= tau else "ambiguous")
    df["asks_clarification"] = df["pred_state"].eq("ambiguous")

    final_targets = []
    for _, row in df.iterrows():
        candidates = split_candidates(row["candidate_targets"])
        if row["pred_state"] == "unique":
            final_targets.append(row["matched_target"])
        else:
            final_targets.append(resolve_with_clarification(row.get("clarification_response", ""), candidates))

    df["final_target"] = final_targets

    df["gsa_correct"] = df.apply(
        lambda r: normalize_text(r["pred_state"]) == normalize_text(r["gt_state"]),
        axis=1,
    )
    df["tsa_before_correct"] = df.apply(
        lambda r: target_match(r["matched_target"], r["intended_target"]),
        axis=1,
    )
    df["tsa_after_correct"] = df.apply(
        lambda r: target_match(r["final_target"], r["intended_target"]),
        axis=1,
    )

    unique_df = df[df["gt_state"].map(normalize_text) == "unique"]
    amb_df = df[df["gt_state"].map(normalize_text) == "ambiguous"]

    row = {
        "tau1": tau,
        "num_samples": len(df),
        "num_unique": len(unique_df),
        "num_ambiguous": len(amb_df),
        "GSA": float(df["gsa_correct"].mean()) if len(df) else float("nan"),
        "TSA_before_all": float(df["tsa_before_correct"].mean()) if len(df) else float("nan"),
        "TSA_unique": float(unique_df["tsa_after_correct"].mean()) if len(unique_df) else float("nan"),
        "TSA_after_all": float(df["tsa_after_correct"].mean()) if len(df) else float("nan"),
        "clarification_rate_ambiguous": float(amb_df["asks_clarification"].mean()) if len(amb_df) else float("nan"),
        "false_clarification_rate_unique": float(unique_df["asks_clarification"].mean()) if len(unique_df) else float("nan"),
    }

    return df, row


def main():
    load_dotenv(f"{REPO_ROOT}/.env")

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is empty in .env")

    print("=" * 100)
    print("Target Confirmation LLM Evaluation")
    print("=" * 100)
    print(f"Input CSV:  {INPUT_CSV}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Model:      {model_name}")
    print(f"Base URL:   {base_url}")
    print(f"TAU1:       {TAU1}")
    print("=" * 100)

    df = pd.read_csv(INPUT_CSV)
    client = OpenAI(api_key=api_key, base_url=base_url)

    # cache support
    if PRED_CSV.exists():
        pred_df = pd.read_csv(PRED_CSV)
        done_keys = set(pred_df["sample_id"].astype(str))
        rows = pred_df.to_dict("records")
        print(f"[CACHE] Loaded {len(rows)} existing predictions.")
    else:
        done_keys = set()
        rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="DeepSeek confirmation"):
        sample_id = str(idx)
        if sample_id in done_keys:
            continue

        candidates = split_candidates(row["candidate_targets"])
        pred = call_confirmation_agent(
            client=client,
            model_name=model_name,
            instruction=row["instruction"],
            candidates=candidates,
        )

        out_row = row.to_dict()
        out_row.update({
            "sample_id": sample_id,
            "matched_target": pred["matched_target"],
            "score": pred["score"],
            "reason": pred["reason"],
            "llm_raw": pred["llm_raw"],
            "api_error": pred["api_error"],
        })
        rows.append(out_row)

        # save every sample
        pd.DataFrame(rows).to_csv(PRED_CSV, index=False)
        time.sleep(SLEEP_BETWEEN_CALLS)

    pred_df = pd.DataFrame(rows).sort_values("sample_id", key=lambda x: x.astype(int))
    pred_df.to_csv(PRED_CSV, index=False)

    # threshold analysis
    threshold_rows = []
    final_df_tau = None
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        final_df, metric_row = compute_metrics(pred_df, tau)
        threshold_rows.append(metric_row)
        if abs(tau - TAU1) < 1e-9:
            final_df_tau = final_df

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(THRESHOLD_CSV, index=False)

    if final_df_tau is None:
        final_df_tau, metric_row = compute_metrics(pred_df, TAU1)
    else:
        metric_row = threshold_df[threshold_df["tau1"].eq(TAU1)].iloc[0].to_dict()

    final_df_tau.to_csv(OUT_DIR / f"predictions_with_metrics_tau{TAU1:.1f}.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "num_samples", "value": metric_row["num_samples"]},
        {"metric": "num_unique", "value": metric_row["num_unique"]},
        {"metric": "num_ambiguous", "value": metric_row["num_ambiguous"]},
        {"metric": "Grounding State Accuracy (GSA)", "value": metric_row["GSA"]},
        {"metric": "TSA before clarification/all instructions", "value": metric_row["TSA_before_all"]},
        {"metric": "TSA on unique instructions", "value": metric_row["TSA_unique"]},
        {"metric": "TSA after clarification/all instructions", "value": metric_row["TSA_after_all"]},
        {"metric": "Clarification rate on ambiguous instructions", "value": metric_row["clarification_rate_ambiguous"]},
        {"metric": "False clarification rate on unique instructions", "value": metric_row["false_clarification_rate_unique"]},
    ])
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 100)
    print("[SUMMARY @ tau1=0.6]")
    print(summary.to_string(index=False))
    print("\n[THRESHOLD ANALYSIS]")
    print(threshold_df.to_string(index=False))
    print("=" * 100)
    print(f"Predictions:        {PRED_CSV}")
    print(f"Summary:            {SUMMARY_CSV}")
    print(f"Threshold analysis: {THRESHOLD_CSV}")


if __name__ == "__main__":
    main()
