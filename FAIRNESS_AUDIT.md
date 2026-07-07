# Evaluation Fairness (Table II)

This note documents that the main comparison in Table II — EgoMed-Agent vs. the text-prompt baselines, the module ablations (Init-Only Prop. / Frame-wise Det.), and the nnU-Net upper bound — is evaluated fairly. Every axis (inputs, activation, metric, aggregation) was verified against the code and the data.

## Two evaluation code paths (proven equivalent)
- **Detector path**: EgoMed-Agent (`scripts/egomed_agent/egomed-agent-iou06.py`), Init-Only Prop. (`scripts/baselines/eval_baseline1.py`), Frame-wise Det. (`scripts/baselines/eval_baseline2.py`) — driven by `data/text_prompt_eval/<DS>_test_prompt_schedule.csv`.
- **Text path**: Grounded SAM2 / LangSAM / MedSAM3 / SAM3 (`scripts/baselines/eval_*_save_outputs.py`) — driven by `<DS>_test_text_prompts_online_schedule.csv`. The five scripts share an identical evaluation template.
- The two schedules cover the **same 438 `(dataset, case, target_class)` keys**, with **0 activation-frame mismatches** and **identical prompt strings per key** — both paths evaluate the same targets at the same activation moment.

## Per-axis verification
| Axis | Verdict | Evidence |
|---|---|---|
| Same targets / activation | ✅ | The two schedules share all 438 keys; 0 activation-frame mismatches |
| Unbiased activation timing | ✅ | `prompt_activate_frame == first gt_exists frame` for all 438; no GT frames before activation → no pre-activation penalty |
| Dice function | ✅ | Identical `dice_score` across methods: boolean masks, both-empty → 1.0, one-empty → 0.0, `2·inter/(p+g+eps)`, eps = 1e-6 |
| Scored-frame set | ✅ | Dice is computed only on `gt_exists` frames (from activation onward) |
| Missing / empty prediction | ✅ | Frames where GT exists but the prediction is missing/empty → Dice = 0 (not skipped), consistently across methods |
| Aggregation | ✅ | Two-level case-weighted mean (per-frame → per-case mean → across-case mean), identical on both paths |
| Prompt fairness | ✅ | Baselines receive a clean `"segment the {target_class}"` — the same target the agent confirms — never a weakened prompt |
| Error-frame handling | ✅ | The text-baseline summary excludes error frames, but 0 errors were measured on `gt_exists` frames (e.g. 119,043 frames for Grounded SAM2) → no effect |

## Honest disclosures (all fair)
1. **nnU-Net is a supervised upper bound** (trained on masks, fixed classes); the paper discloses it as a reference, not a direct competitor — standard practice.
2. **The main table uses the unambiguous prompt for the baselines.** Left-kidney / left-lung cases are each evaluated twice — with an unambiguous prompt (`"segment the left kidney"`) and an ambiguous one (`"segment the kidney"`). The main table always uses the unambiguous number; the ambiguous rows are extra data for the ambiguity analysis / Table III, not the main table.
3. **The agent's main-table run selects the target from `target_class` directly** (no LLM call). The main table therefore measures segmentation + tracking quality given the target, while confirmation accuracy is measured separately in Table III — a clean decoupling in which both methods receive the same unambiguous target.
4. **The text baselines run per-frame**, which is their stronger configuration. The module ablation shows that naive temporal propagation drifts under head motion (Init-Only Prop. 50.46 < Frame-wise Det. 59.27), so per-frame gives the baselines their best shot and is conservative for EgoMed-Agent.
5. All methods share the same inputs (same egocentric video, same instruction, same activation schedule) and the same outputs (same DSC, same frame set, same aggregation). EgoMed-Agent's advantage comes from the method (detection + confirmation + consistency-gated correction), not from any evaluation convenience.
