#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Target Confirmation evaluation demo.

This script evaluates the target confirmation stage only.
It does NOT run segmentation.

Input:
  target_confirmation_eval_demo.csv

Output:
  predictions.csv
  summary.csv

Metrics:
  - GSA: Grounding State Accuracy
  - TSA on unique instructions
  - TSA after clarification
  - Clarification rate on ambiguous instructions
  - False clarification rate on unique instructions
"""

from pathlib import Path

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import re
import pandas as pd


INPUT_CSV = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_full.csv")
OUT_DIR = Path(f"{REPO_ROOT}/runs/target_confirmation_eval/full_rule_based")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV = OUT_DIR / "predictions.csv"
SUMMARY_CSV = OUT_DIR / "summary.csv"


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


def find_explicit_candidate(instruction, candidates):
    """
    If the instruction explicitly contains a candidate name, return that candidate.
    Example:
      instruction = "Highlight the left kidney"
      candidates = ["left kidney", "right kidney"]
      -> "left kidney"
    """
    instr = normalize_text(instruction)

    # Prefer longer names first, e.g. "left kidney" before "kidney"
    candidates_sorted = sorted(candidates, key=lambda x: len(normalize_text(x)), reverse=True)

    for cand in candidates_sorted:
        c = normalize_text(cand)
        if c and c in instr:
            return cand

    return None


def ambiguous_family_from_instruction(instruction, candidates):
    """
    Detect whether instruction refers to a broad family with multiple candidates.

    Examples:
      "kidney" -> left kidney / right kidney
      "lung" -> left lung / right lung
      "ventricle" -> LV cavity / RV cavity, Left Ventricle / Left Atrium may be trickier
      "chamber" -> Left Ventricle / Left Atrium / LV cavity / RV cavity
    """
    instr = normalize_text(instruction)
    cands_norm = [(c, normalize_text(c)) for c in candidates]

    families = {
        "kidney": ["kidney"],
        "lung": ["lung"],
        "ventricle": ["ventricle", "lv cavity", "rv cavity", "left ventricle", "right ventricle"],
        "chamber": ["chamber", "atrium", "ventricle", "lv cavity", "rv cavity", "left atrium", "left ventricle"],
        "cavity": ["cavity", "lv cavity", "rv cavity"],
    }

    for trigger, terms in families.items():
        if trigger not in instr:
            continue

        matched = []
        for cand, cand_norm in cands_norm:
            for term in terms:
                if term in cand_norm:
                    matched.append(cand)
                    break

        # If one broad word maps to more than one candidate, treat as ambiguous.
        if len(matched) >= 2:
            return matched

    return []


def resolve_with_clarification(clarification_response, candidates):
    """
    Resolve final target from user clarification.
    Examples:
      clarification_response = "left"
      candidates = ["left kidney", "right kidney"]
      -> "left kidney"
    """
    clar = normalize_text(clarification_response)
    if not clar:
        return ""

    # Direct full candidate match
    direct = find_explicit_candidate(clar, candidates)
    if direct is not None:
        return direct

    # Side-based clarification
    if "left" in clar:
        for cand in candidates:
            if "left" in normalize_text(cand) or normalize_text(cand).startswith("lv"):
                return cand

    if "right" in clar:
        for cand in candidates:
            if "right" in normalize_text(cand) or normalize_text(cand).startswith("rv"):
                return cand

    # Abbreviations
    if clar in ["lv", "left ventricle"]:
        for cand in candidates:
            if normalize_text(cand) in ["lv cavity", "left ventricle"] or "left ventricle" in normalize_text(cand):
                return cand

    if clar in ["rv", "right ventricle"]:
        for cand in candidates:
            if normalize_text(cand) in ["rv cavity", "right ventricle"] or "right ventricle" in normalize_text(cand):
                return cand

    return ""


def confirm_target(instruction, candidate_targets, clarification_response):
    """
    Rule-based mock Confirmation Agent.

    Later, you can replace this function with your DeepSeek/LLM Confirmation Agent.
    """
    candidates = split_candidates(candidate_targets)

    explicit = find_explicit_candidate(instruction, candidates)

    if explicit is not None:
        return {
            "pred_state": "unique",
            "pred_target_before": explicit,
            "asks_clarification": False,
            "clarification_question": "",
            "final_target": explicit,
        }

    ambiguous_candidates = ambiguous_family_from_instruction(instruction, candidates)

    if len(ambiguous_candidates) >= 2:
        final_target = resolve_with_clarification(clarification_response, ambiguous_candidates)
        question = "I found " + " and ".join(ambiguous_candidates) + ". Which one do you want?"
        return {
            "pred_state": "ambiguous",
            "pred_target_before": "",
            "asks_clarification": True,
            "clarification_question": question,
            "final_target": final_target,
        }

    # Fallback: if only one candidate, choose it.
    if len(candidates) == 1:
        return {
            "pred_state": "unique",
            "pred_target_before": candidates[0],
            "asks_clarification": False,
            "clarification_question": "",
            "final_target": candidates[0],
        }

    # If uncertain and multiple candidates exist, ask clarification.
    final_target = resolve_with_clarification(clarification_response, candidates)
    return {
        "pred_state": "ambiguous",
        "pred_target_before": "",
        "asks_clarification": True,
        "clarification_question": "Multiple candidate targets are detected. Which one do you want?",
        "final_target": final_target,
    }


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)
    rows = []

    for idx, row in df.iterrows():
        pred = confirm_target(
            instruction=row["instruction"],
            candidate_targets=row["candidate_targets"],
            clarification_response=row.get("clarification_response", ""),
        )

        gt_state = normalize_text(row["gt_state"])
        intended = row["intended_target"]

        pred_state = normalize_text(pred["pred_state"])
        pred_before = pred["pred_target_before"]
        final_target = pred["final_target"]

        rows.append({
            **row.to_dict(),
            "pred_state": pred["pred_state"],
            "pred_target_before": pred_before,
            "asks_clarification": pred["asks_clarification"],
            "clarification_question": pred["clarification_question"],
            "final_target": final_target,
            "gsa_correct": pred_state == gt_state,
            "tsa_before_correct": target_match(pred_before, intended),
            "tsa_after_correct": target_match(final_target, intended),
        })

    out = pd.DataFrame(rows)
    out.to_csv(PRED_CSV, index=False)

    total = len(out)
    unique_df = out[out["gt_state"].map(normalize_text) == "unique"].copy()
    amb_df = out[out["gt_state"].map(normalize_text) == "ambiguous"].copy()

    gsa = out["gsa_correct"].mean() if total else float("nan")
    tsa_unique = unique_df["tsa_after_correct"].mean() if len(unique_df) else float("nan")
    tsa_after_all = out["tsa_after_correct"].mean() if total else float("nan")
    clar_rate_amb = amb_df["asks_clarification"].mean() if len(amb_df) else float("nan")
    false_clar_unique = unique_df["asks_clarification"].mean() if len(unique_df) else float("nan")

    summary = pd.DataFrame([
        {"metric": "num_samples", "value": total},
        {"metric": "num_unique", "value": len(unique_df)},
        {"metric": "num_ambiguous", "value": len(amb_df)},
        {"metric": "Grounding State Accuracy (GSA)", "value": gsa},
        {"metric": "TSA on unique instructions", "value": tsa_unique},
        {"metric": "TSA after clarification/all instructions", "value": tsa_after_all},
        {"metric": "Clarification rate on ambiguous instructions", "value": clar_rate_amb},
        {"metric": "False clarification rate on unique instructions", "value": false_clar_unique},
    ])

    summary.to_csv(SUMMARY_CSV, index=False)

    print("=" * 100)
    print("Target Confirmation Demo Evaluation Done")
    print("=" * 100)
    print(f"Input:       {INPUT_CSV}")
    print(f"Predictions: {PRED_CSV}")
    print(f"Summary:     {SUMMARY_CSV}")
    print("=" * 100)

    print("\nPredictions:")
    display_cols = [
        "dataset",
        "instruction",
        "intended_target",
        "gt_state",
        "pred_state",
        "pred_target_before",
        "asks_clarification",
        "final_target",
        "gsa_correct",
        "tsa_after_correct",
    ]
    print(out[display_cols].to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nSummary in percent:")
    for _, r in summary.iterrows():
        metric = r["metric"]
        value = r["value"]
        if metric.startswith("num_"):
            print(f"{metric}: {int(value)}")
        else:
            print(f"{metric}: {float(value) * 100:.2f}%")


if __name__ == "__main__":
    main()
