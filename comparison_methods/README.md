# comparison_methods/ — external baselines

The external repositories used for the *Comparison Methods* in §IV-A of the paper. They are **not bundled** in this repository. To run the comparison baselines, clone each one under `$EGOMED_EXT_ROOT` (at the commits listed in [`../REPRODUCE.md`](../REPRODUCE.md)), apply the matching patch from [`patches/`](patches), and use the evaluation scripts in `scripts/baselines/` as mapped below.

| Repo | Role | Eval script(s) in this repo |
|---|---|---|
| `sam3` | SAM3 baseline (Table II); `pip install -e .` into the sam2 env; also provides the backbone/checkpoint for MedSAM3 | `scripts/baselines/eval_sam3_image_text_prompt_egomed5*.py`, `eval_sam3_video_text_prompt_egomed5.py` |
| `Grounded-SAM-2` | Grounded SAM2 baseline (Table II) | `scripts/baselines/eval_grounded_sam2_image_text_prompt_egomed5_save_outputs.py` |
| `MedSAM3` | MedSAM3 baseline (Table II); uses the `sam3` checkpoint | `scripts/baselines/eval_medsam3_image_text_prompt_egomed5_save_outputs.py` |
| `lang-segment-anything` | LangSAM baseline (Table II) | `scripts/baselines/eval_langsam_image_text_prompt_egomed5_save_outputs.py` |
| `LISA` | LISA baseline (Table II); **evaluated inside the LISA repo** | none here; results in `LISA/runs/` and `runs/eval_lisa_text_prompt/` |

Notes:
- `clipseg` is a dependency of LangSAM.
- The evaluation scripts locate these repositories through `$EGOMED_EXT_ROOT` (default: the parent directory of this repository). See [`../REPRODUCE.md`](../REPRODUCE.md) step 3 for the commits and patches.
