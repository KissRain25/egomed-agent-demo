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


# ============================================================
# Paths
# ============================================================

INPUT_CSV = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_detector_generated_v2_sample150.csv")

OUT_DIR = Path(f"{REPO_ROOT}/runs/target_confirmation_eval/detector_generated_sample150_deepseek")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV = OUT_DIR / "predictions.csv"
THRESHOLD_CSV = OUT_DIR / "threshold_analysis.csv"

# Main threshold. We still perform threshold analysis below.
TAU1 = 0.8

MAX_RETRY = 3
SLEEP_BETWEEN_CALLS = 0.2


# ============================================================
# Basic utilities
# ============================================================

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


def safe_json_loads(x, default=None):
    if default is None:
        default = []

    if pd.isna(x):
        return default

    try:
        return json.loads(str(x))
    except Exception:
        return default


def compact_candidate_objects(candidate_objects):
    """
    Keep only the fields defined in the paper for candidate objects:
    candidate id, category, and bounding box.
    Spatial relations such as left/right/upper/lower/larger/smaller
    should be inferred by the Confirmation Agent from the bbox.
    """
    objs = safe_json_loads(candidate_objects, default=[])

    compact = []
    for obj in objs:
        compact.append({
            "id": str(obj.get("id", "")),
            "category": str(obj.get("category", "")),
            "bbox": obj.get("bbox", []),
        })

    return compact


def extract_json(text):
    text = str(text).strip()

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


def find_object_by_id(candidate_objects, obj_id):
    obj_id = str(obj_id).strip()
    for obj in candidate_objects:
        if str(obj.get("id", "")).strip() == obj_id:
            return obj
    return None


def find_object_by_category(candidate_objects, category):
    cat_norm = normalize_text(category)
    for obj in candidate_objects:
        if normalize_text(obj.get("category", "")) == cat_norm:
            return obj
    return None


def repair_target(initial_or_clarified, candidate_objects, id_key, category_key):
    """
    Ensure target id/category are valid candidate outputs.
    """
    target_id = str(initial_or_clarified.get(id_key, "")).strip()
    target_category = str(initial_or_clarified.get(category_key, "")).strip()

    # Prefer valid id.
    obj = find_object_by_id(candidate_objects, target_id)
    if obj is not None:
        return str(obj["id"]), str(obj["category"])

    # Fallback to category match.
    obj = find_object_by_category(candidate_objects, target_category)
    if obj is not None:
        return str(obj["id"]), str(obj["category"])

    # Invalid output.
    return "", ""


def target_id_match(pred_id, gt_id):
    return str(pred_id).strip() == str(gt_id).strip()


# ============================================================
# Prompts
# ============================================================

def build_initial_confirmation_prompt(instruction, candidate_objects):
    candidate_json = json.dumps(candidate_objects, ensure_ascii=False, indent=2)

    return f"""
You are the Confirmation Agent in EgoMed-Agent, a multi-agent system for Interactive Egocentric Medical Image Segmentation (IEMIS).

Your task is to evaluate whether the user's instruction can be reliably grounded to a unique candidate medical target before segmentation.

You are given:
1. A user instruction.
2. A set of candidate medical targets extracted from the initial frame.
Each candidate contains:
- id: candidate identifier
- category: medical target category
- bbox: bounding box [x1, y1, x2, y2]

User instruction:
"{instruction}"

Candidate targets:
{candidate_json}

You should determine whether the instruction reliably specifies exactly one candidate target.

Grounding state definition:
- "unique": the instruction provides enough semantic or spatial information to identify exactly one candidate target reliably.
- "ambiguous": the instruction does not provide enough information to identify a unique target among the candidates.

Important rules:
1. If the instruction explicitly names a unique category, such as "left kidney", "right lung", "LV cavity", "Left Atrium", or "polyp", the grounding state should be "unique".
2. If the instruction refers to a category shared by multiple candidates, such as "kidney", "lung", "cardiac chamber", "cardiac structure", or "cardiac target", the grounding state should be "ambiguous".
3. If the instruction uses spatial information, such as "the object on the left", "the upper object", "the lower object", "the object with the larger bounding box", or "the object with the smaller bounding box", infer the spatial or size relationship directly from the candidate bounding boxes. For left/right, compare the horizontal center of the boxes. For upper/lower, compare the vertical center of the boxes. For larger/smaller, compare the box areas.
4. If multiple candidates remain plausible, the grounding state should be "ambiguous".
5. Do not invent candidate targets. The target_id must be one of the candidate ids.
6. If the grounding state is "ambiguous", target_id and target_category should be empty strings because the final target should be confirmed after user clarification.
7. The score measures the reliability of grounding between the user instruction and the candidate targets.

Return only a valid JSON object with the following fields:
{{
  "grounding_state": "unique" or "ambiguous",
  "target_id": "candidate id if uniquely grounded, otherwise empty string",
  "target_category": "candidate category if uniquely grounded, otherwise empty string",
  "score": a reliability score from 0 to 1,
  "reason": "brief explanation"
}}

Scoring rule:
- Use 0.80 to 1.00 when the instruction reliably grounds to exactly one candidate.
- Use 0.50 to 0.79 when the instruction is partially informative but still not fully reliable.
- Use below 0.50 when the instruction is ambiguous among multiple candidates.

Return only JSON. Do not include markdown.
""".strip()


def build_clarification_prompt(instruction, candidate_objects, clarification_response):
    candidate_json = json.dumps(candidate_objects, ensure_ascii=False, indent=2)

    return f"""
You are the Confirmation Agent in EgoMed-Agent, a multi-agent system for Interactive Egocentric Medical Image Segmentation (IEMIS).

The previous grounding result was considered unreliable or ambiguous. The system has requested clarification from the user.

Your task is to determine the final user-intended target according to:
1. The original user instruction.
2. The candidate medical targets.
3. The user's clarification response.

Original user instruction:
"{instruction}"

Candidate targets:
{candidate_json}

User clarification response:
"{clarification_response}"

Each candidate contains:
- id: candidate identifier
- category: medical target category
- bbox: bounding box [x1, y1, x2, y2]

Important rules:
1. Use the clarification response to resolve the user's intended target.
2. The confirmed target must be one of the provided candidates.
3. Do not invent candidate targets.
4. If the clarification mentions a category, match it to the corresponding candidate category.
5. If the clarification mentions spatial information, infer the target directly from the candidate bounding boxes. For left/right, compare the horizontal center of the boxes. For upper/lower, compare the vertical center of the boxes. For larger/smaller, compare the box areas.

Return only a valid JSON object with the following fields:
{{
  "confirmed_target_id": "candidate id",
  "confirmed_target_category": "candidate category",
  "reason": "brief explanation"
}}

Return only JSON. Do not include markdown.
""".strip()


# ============================================================
# LLM calls
# ============================================================

def chat_completion(client, model_name, prompt):
    last_err = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip(), ""
        except Exception as e:
            last_err = repr(e)
            time.sleep(1.0 * attempt)

    return "", last_err or "unknown error"


def call_initial_confirmation_agent(client, model_name, instruction, candidate_objects):
    prompt = build_initial_confirmation_prompt(instruction, candidate_objects)
    raw, err = chat_completion(client, model_name, prompt)

    if err:
        return {
            "initial_raw": raw,
            "initial_api_error": err,
            "grounding_state": "ambiguous",
            "target_id": "",
            "target_category": "",
            "score": 0.0,
            "initial_reason": "",
        }

    try:
        data = extract_json(raw)

        grounding_state = normalize_text(data.get("grounding_state", "ambiguous"))
        if grounding_state not in ["unique", "ambiguous"]:
            grounding_state = "ambiguous"

        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))

        target_id, target_category = repair_target(
            data,
            candidate_objects,
            id_key="target_id",
            category_key="target_category",
        )

        # If the model says ambiguous, force empty target.
        if grounding_state == "ambiguous":
            target_id, target_category = "", ""

        return {
            "initial_raw": raw,
            "initial_api_error": "",
            "grounding_state": grounding_state,
            "target_id": target_id,
            "target_category": target_category,
            "score": score,
            "initial_reason": str(data.get("reason", "")).strip(),
        }

    except Exception as e:
        return {
            "initial_raw": raw,
            "initial_api_error": repr(e),
            "grounding_state": "ambiguous",
            "target_id": "",
            "target_category": "",
            "score": 0.0,
            "initial_reason": "",
        }


def call_clarification_confirmation_agent(
    client,
    model_name,
    instruction,
    candidate_objects,
    clarification_response,
):
    prompt = build_clarification_prompt(
        instruction=instruction,
        candidate_objects=candidate_objects,
        clarification_response=clarification_response,
    )

    raw, err = chat_completion(client, model_name, prompt)

    if err:
        return {
            "clarification_raw": raw,
            "clarification_api_error": err,
            "confirmed_target_id": "",
            "confirmed_target_category": "",
            "clarification_reason": "",
        }

    try:
        data = extract_json(raw)

        target_id, target_category = repair_target(
            data,
            candidate_objects,
            id_key="confirmed_target_id",
            category_key="confirmed_target_category",
        )

        return {
            "clarification_raw": raw,
            "clarification_api_error": "",
            "confirmed_target_id": target_id,
            "confirmed_target_category": target_category,
            "clarification_reason": str(data.get("reason", "")).strip(),
        }

    except Exception as e:
        return {
            "clarification_raw": raw,
            "clarification_api_error": repr(e),
            "confirmed_target_id": "",
            "confirmed_target_category": "",
            "clarification_reason": "",
        }


# ============================================================
# Metrics
# ============================================================

def compute_metrics(pred_df, tau):
    df = pred_df.copy()

    # Predicted grounding state for GSA uses thresholded reliability,
    # aligned with the paper formulation.
    df["pred_state_tau"] = df["score"].apply(
        lambda x: "unique" if float(x) >= float(tau) else "ambiguous"
    )

    final_ids = []
    final_categories = []
    asks = []

    for _, row in df.iterrows():
        if row["pred_state_tau"] == "unique":
            final_ids.append(str(row.get("target_id", "")))
            final_categories.append(str(row.get("target_category", "")))
            asks.append(False)
        else:
            final_ids.append(str(row.get("confirmed_target_id", "")))
            final_categories.append(str(row.get("confirmed_target_category", "")))
            asks.append(True)

    df["asks_clarification_tau"] = asks
    df["final_target_id_tau"] = final_ids
    df["final_target_category_tau"] = final_categories

    df["gsa_correct_tau"] = df.apply(
        lambda r: normalize_text(r["pred_state_tau"]) == normalize_text(r["gt_state"]),
        axis=1,
    )

    df["tca_correct_tau"] = df.apply(
        lambda r: target_id_match(r["final_target_id_tau"], r["intended_target_id"]),
        axis=1,
    )

    unique_df = df[df["gt_state"].map(normalize_text).eq("unique")]
    amb_df = df[df["gt_state"].map(normalize_text).eq("ambiguous")]
    semantic_df = df[df["instruction_type"].map(normalize_text).eq("semantic")]
    spatial_df = df[df["instruction_type"].map(normalize_text).eq("spatial")]

    row = {
        "tau1": tau,
        "num_samples": len(df),
        "num_unique": len(unique_df),
        "num_ambiguous": len(amb_df),
        "num_semantic": len(semantic_df),
        "num_spatial": len(spatial_df),

        "GSA": float(df["gsa_correct_tau"].mean()) if len(df) else float("nan"),
        "TCA": float(df["tca_correct_tau"].mean()) if len(df) else float("nan"),

        "GSA_unique": float(unique_df["gsa_correct_tau"].mean()) if len(unique_df) else float("nan"),
        "GSA_ambiguous": float(amb_df["gsa_correct_tau"].mean()) if len(amb_df) else float("nan"),

        "TCA_unique": float(unique_df["tca_correct_tau"].mean()) if len(unique_df) else float("nan"),
        "TCA_ambiguous": float(amb_df["tca_correct_tau"].mean()) if len(amb_df) else float("nan"),
        "TCA_semantic": float(semantic_df["tca_correct_tau"].mean()) if len(semantic_df) else float("nan"),
        "TCA_spatial": float(spatial_df["tca_correct_tau"].mean()) if len(spatial_df) else float("nan"),

        "clarification_rate": float(df["asks_clarification_tau"].mean()) if len(df) else float("nan"),
        "clarification_rate_ambiguous": float(amb_df["asks_clarification_tau"].mean()) if len(amb_df) else float("nan"),
        "false_clarification_rate_unique": float(unique_df["asks_clarification_tau"].mean()) if len(unique_df) else float("nan"),
    }

    return df, row


# ============================================================
# Main
# ============================================================

def main():
    load_dotenv(f"{REPO_ROOT}/.env")

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is empty in .env")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    print("=" * 100)
    print("Target Confirmation Evaluation V2")
    print("=" * 100)
    print(f"Input CSV:  {INPUT_CSV}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Model:      {model_name}")
    print(f"Base URL:   {base_url}")
    print(f"Main TAU1:  {TAU1}")
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="DeepSeek confirmation v2"):
        sample_id = str(row["sample_id"])

        if sample_id in done_keys:
            continue

        candidate_objects = compact_candidate_objects(row["candidate_objects"])
        instruction = str(row["instruction"])

        initial = call_initial_confirmation_agent(
            client=client,
            model_name=model_name,
            instruction=instruction,
            candidate_objects=candidate_objects,
        )

        # Always call clarification for ambiguous GT rows or low-score rows?
        # To support threshold analysis, call clarification whenever a clarification response exists.
        clarification_response = "" if pd.isna(row.get("clarification_response", "")) else str(row.get("clarification_response", ""))

        if clarification_response.strip():
            clarified = call_clarification_confirmation_agent(
                client=client,
                model_name=model_name,
                instruction=instruction,
                candidate_objects=candidate_objects,
                clarification_response=clarification_response,
            )
        else:
            clarified = {
                "clarification_raw": "",
                "clarification_api_error": "",
                "confirmed_target_id": "",
                "confirmed_target_category": "",
                "clarification_reason": "",
            }

        out_row = row.to_dict()
        out_row.update({
            "candidate_objects_compact": json.dumps(candidate_objects, ensure_ascii=False),

            "grounding_state": initial["grounding_state"],
            "target_id": initial["target_id"],
            "target_category": initial["target_category"],
            "score": initial["score"],
            "initial_reason": initial["initial_reason"],
            "initial_raw": initial["initial_raw"],
            "initial_api_error": initial["initial_api_error"],

            "confirmed_target_id": clarified["confirmed_target_id"],
            "confirmed_target_category": clarified["confirmed_target_category"],
            "clarification_reason": clarified["clarification_reason"],
            "clarification_raw": clarified["clarification_raw"],
            "clarification_api_error": clarified["clarification_api_error"],
        })

        rows.append(out_row)

        # save every sample
        pd.DataFrame(rows).to_csv(PRED_CSV, index=False)

        time.sleep(SLEEP_BETWEEN_CALLS)

    pred_df = pd.DataFrame(rows)
    pred_df["sample_id"] = pred_df["sample_id"].astype(str)
    pred_df = pred_df.sort_values("sample_id", key=lambda x: x.astype(int))
    pred_df.to_csv(PRED_CSV, index=False)

    # Threshold analysis
    threshold_rows = []
    final_df_tau = None

    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        final_df, metric_row = compute_metrics(pred_df, tau)
        threshold_rows.append(metric_row)

        if abs(float(tau) - float(TAU1)) < 1e-9:
            final_df_tau = final_df

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(THRESHOLD_CSV, index=False)

    if final_df_tau is None:
        final_df_tau, metric_row = compute_metrics(pred_df, TAU1)
    else:
        metric_row = threshold_df[threshold_df["tau1"].eq(TAU1)].iloc[0].to_dict()

    final_pred_path = OUT_DIR / f"predictions_with_metrics_tau{TAU1:.1f}.csv"
    summary_path = OUT_DIR / f"summary_tau{TAU1:.1f}.csv"

    final_df_tau.to_csv(final_pred_path, index=False)

    summary = pd.DataFrame([
        {"metric": "num_samples", "value": metric_row["num_samples"]},
        {"metric": "num_unique", "value": metric_row["num_unique"]},
        {"metric": "num_ambiguous", "value": metric_row["num_ambiguous"]},
        {"metric": "num_semantic", "value": metric_row["num_semantic"]},
        {"metric": "num_spatial", "value": metric_row["num_spatial"]},

        {"metric": "Grounding State Accuracy (GSA)", "value": metric_row["GSA"]},
        {"metric": "Target Confirmation Accuracy (TCA)", "value": metric_row["TCA"]},

        {"metric": "GSA on unique instructions", "value": metric_row["GSA_unique"]},
        {"metric": "GSA on ambiguous instructions", "value": metric_row["GSA_ambiguous"]},

        {"metric": "TCA on unique instructions", "value": metric_row["TCA_unique"]},
        {"metric": "TCA on ambiguous instructions", "value": metric_row["TCA_ambiguous"]},
        {"metric": "TCA on semantic instructions", "value": metric_row["TCA_semantic"]},
        {"metric": "TCA on spatial instructions", "value": metric_row["TCA_spatial"]},

        {"metric": "Clarification rate", "value": metric_row["clarification_rate"]},
        {"metric": "Clarification rate on ambiguous instructions", "value": metric_row["clarification_rate_ambiguous"]},
        {"metric": "False clarification rate on unique instructions", "value": metric_row["false_clarification_rate_unique"]},
    ])

    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print(f"[SUMMARY @ tau1={TAU1:.1f}]")
    print(summary.to_string(index=False))

    print("\n[THRESHOLD ANALYSIS]")
    print(threshold_df.to_string(index=False))

    print("=" * 100)
    print("Saved:")
    print(f"Predictions:        {PRED_CSV}")
    print(f"Prediction+metrics: {final_pred_path}")
    print(f"Summary:            {summary_path}")
    print(f"Threshold analysis: {THRESHOLD_CSV}")
    print("=" * 100)


if __name__ == "__main__":
    main()
