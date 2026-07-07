#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import random
import pandas as pd

OUT = Path(f"{REPO_ROOT}/data/text_prompt_eval/target_confirmation_eval_full.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)

rows = []

def add_unique(dataset, case_id, frame_idx, target, candidates, n=5):
    templates = [
        "Segment the {target}",
        "Highlight the {target}",
        "Show me the {target}",
        "Find the {target}",
        "Please segment the {target}",
    ]
    for i in range(n):
        inst = templates[i % len(templates)].format(target=target)
        rows.append({
            "dataset": dataset,
            "case_id": case_id,
            "frame_idx": frame_idx,
            "instruction": inst,
            "intended_target": target,
            "gt_state": "unique",
            "candidate_targets": ";".join(candidates),
            "clarification_response": "",
        })

def add_ambiguous(dataset, case_id, frame_idx, instruction, intended_target, candidates, clarification, n=5):
    templates = [
        instruction,
        instruction.replace("Segment", "Highlight"),
        instruction.replace("Segment", "Show me"),
        instruction.replace("Segment", "Find"),
        instruction.replace("Segment", "Please segment"),
    ]
    for i in range(n):
        rows.append({
            "dataset": dataset,
            "case_id": case_id,
            "frame_idx": frame_idx,
            "instruction": templates[i % len(templates)],
            "intended_target": intended_target,
            "gt_state": "ambiguous",
            "candidate_targets": ";".join(candidates),
            "clarification_response": clarification,
        })

# -------------------------
# AMOS
# -------------------------
amos_candidates = ["liver", "spleen", "left kidney", "right kidney", "stomach"]
for target in amos_candidates:
    add_unique("Amos", "207", 124, target, amos_candidates, n=5)

add_ambiguous("Amos", "207", 124, "Segment the kidney", "left kidney", ["left kidney", "right kidney"], "left", n=5)
add_ambiguous("Amos", "207", 124, "Segment the kidney", "right kidney", ["left kidney", "right kidney"], "right", n=5)
add_ambiguous("Amos", "207", 124, "Segment the organ", "liver", amos_candidates, "liver", n=5)
add_ambiguous("Amos", "207", 124, "Segment the organ", "spleen", amos_candidates, "spleen", n=5)
add_ambiguous("Amos", "207", 124, "Segment the organ", "stomach", amos_candidates, "stomach", n=5)

# -------------------------
# ACDC
# -------------------------
acdc_candidates = ["RV cavity", "LV cavity", "myocardium"]
for target in acdc_candidates:
    add_unique("ACDC", "14", 0, target, acdc_candidates, n=5)

add_ambiguous("ACDC", "14", 0, "Segment the ventricle", "LV cavity", ["LV cavity", "RV cavity"], "left ventricle", n=5)
add_ambiguous("ACDC", "14", 0, "Segment the ventricle", "RV cavity", ["LV cavity", "RV cavity"], "right ventricle", n=5)
add_ambiguous("ACDC", "14", 0, "Segment the cardiac cavity", "LV cavity", ["LV cavity", "RV cavity"], "LV cavity", n=5)
add_ambiguous("ACDC", "14", 0, "Segment the cardiac structure", "myocardium", acdc_candidates, "myocardium", n=5)

# -------------------------
# CAMUS
# -------------------------
camus_candidates = ["Left Ventricle", "Left Atrium"]
for target in camus_candidates:
    add_unique("CAMUS", "1", 0, target, camus_candidates, n=5)

add_ambiguous("CAMUS", "1", 0, "Segment the chamber", "Left Ventricle", camus_candidates, "left ventricle", n=5)
add_ambiguous("CAMUS", "1", 0, "Segment the chamber", "Left Atrium", camus_candidates, "left atrium", n=5)
add_ambiguous("CAMUS", "1", 0, "Segment the cardiac chamber", "Left Ventricle", camus_candidates, "left ventricle", n=5)
add_ambiguous("CAMUS", "1", 0, "Segment the cardiac chamber", "Left Atrium", camus_candidates, "left atrium", n=5)

# -------------------------
# MCC
# -------------------------
mcc_candidates = ["left lung", "right lung"]
for target in mcc_candidates:
    add_unique("Montgomery-County-CXR-Set", "1", 0, target, mcc_candidates, n=5)

add_ambiguous("Montgomery-County-CXR-Set", "1", 0, "Segment the lung", "left lung", mcc_candidates, "left", n=5)
add_ambiguous("Montgomery-County-CXR-Set", "1", 0, "Segment the lung", "right lung", mcc_candidates, "right", n=5)

# -------------------------
# PMC
# -------------------------
pmc_candidates = ["polyp"]
add_unique("PolypGen2021_MultiCenterData_v3", "1", 0, "polyp", pmc_candidates, n=10)

# Shuffle
random.shuffle(rows)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"Saved: {OUT}")
print(f"num_samples: {len(df)}")
print(df["gt_state"].value_counts())
print(df.groupby(["dataset", "gt_state"]).size())
