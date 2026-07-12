# Reproduction Guide (REPRODUCE)

Reproduces the experiments in *“Understanding From Human Perspective: A Multi-agent System for Interactive Egocentric Medical Image Segmentation”*. Read alongside `README.md` (repository layout) and `PAPER_TO_CODE.md` (paper section / table / figure ↔ script).

## What can be reproduced
- **Evaluation (Table II main results / ablations / qualitative)**: given the weights below, inference is deterministic and reproducible.
- **Training**: the scripts do **not** fix a random seed, so re-training gives slightly different weights; we therefore provide the trained weights.
- **Target Confirmation (Table III, GSA/TCA)**: relies on the DeepSeek API, which is **non-deterministic** and whose models change over time, so the numbers may not reproduce bit-for-bit.
- **Dataset**: The Dataset contains egocentric recordings and projected GT and is released on Hugging Face (see the Dataset section of `README.md`). The first-frame-selection / keyframe scripts used to build the GT run on the collectors' local machines and are not part of this repository.

## 0. Prerequisites
- Linux + NVIDIA GPU (the paper used an RTX A6000 48G); CUDA 12.x; conda.

## 1. Environment
```bash
conda env create -n sam2 -f environment.yml      # or: conda create -n sam2 python=3.12 && pip install -r requirements.txt
conda activate sam2
pip install -e .                                  # install this repo's SAM_2 package (editable)
```
> `requirements.txt` pins two git editable installs — this repo's `SAM_2` and `sam3` — that cannot be restored from the freeze lines automatically: `git clone` each at its pinned commit and `pip install -e .` (sam3: see step 3).

## 2. Checkpoints
- **SAM 2**: `bash checkpoints/download_ckpts.sh` (produces `checkpoints/sam2.1_hiera_base_plus.pt`, etc.).
- **Trained detectors / classifier** (used in the paper; reuse to skip re-training):
  - per-modality + all-dataset detectors: `runs/yolo26_det/<DS>_yolo26m_imgsz1024/weights/best.pt`
  - image-type classifier: `runs/tool_selection_cls/tool_selection_yolo26m_cls_imgsz320/weights/best.pt`
  - to re-train, see `scripts/detection_agent/train_yolo26_egomed5_all.py` and `train_tool_selection_yolo26_cls.py`.
- **Baseline checkpoints**: e.g. `sam3.pt` (3.3G, under `<sam3>/checkpoints/`); each baseline repo also needs its own official weights.

## 3. Comparison-method repositories (baselines)
The comparison baselines use external repositories; clone each at the commit below under `$EGOMED_EXT_ROOT` (details in `comparison_methods/README.md`):

| Repo | commit | note |
|---|---|---|
| sam3 | `847e1a3` | `pip install -e .`; provides the backbone/checkpoint for SAM3 and MedSAM3 |
| Grounded-SAM-2 | `b7a9c29` | apply `comparison_methods/patches/Grounded-SAM-2.patch` |
| MedSAM3 | `f79eef3` | apply `comparison_methods/patches/MedSAM3.patch` (points infer_sam.py at the sam3 checkpoint) |
| LISA | `3cb2d43` | apply `comparison_methods/patches/LISA.patch`; LISA is evaluated inside its own repo |
| lang-segment-anything | (no git) | LangSAM; depends on `clipseg` |

Apply a patch: `cd /path/to/<repo> && git apply /path/to/EgoMed-Agent/comparison_methods/patches/<repo>.patch`

## 4. Data
Obtain the Dataset from Hugging Face and lay it out under `data/<DS>/`. Key files:
`data/<DS>/img/<case>/*.jpg` (frames), `data/<DS>/label/<case>/*.png` (GT), and
`data/text_prompt_eval/<DS>_test_prompt_schedule.csv` (evaluation schedule — shipped in this repo).
The dataset split is train:val:test = 5:2:3.

> `egomed5` appears in several legacy script and file names as an internal identifier. It is not the name of the Dataset.

## 5. Paths
The core pipeline (`scripts/egomed_agent`, `detection_agent`, `confirmation_agent`, `baselines`) resolves the repository root automatically and runs from any clone. Override data / output locations with the environment variables `EGOMED_ROOT` / `EGOMED_EXT_ROOT` (see Path configuration below).

## 6. DeepSeek (Confirmation Agent)
```bash
cp .env.example .env        # fill in DEEPSEEK_API_KEY; BASE_URL / MODEL default to deepseek-v4-flash
```

## 7. Running (order follows the paper; see PAPER_TO_CODE.md)
Run from the repository root; first edit `CUDA_VISIBLE_DEVICES` etc. at the top of each script:
```bash
# Table II main result (τ₂=0.6)
python scripts/egomed_agent/egomed-agent-iou06.py
# module ablations
python scripts/baselines/eval_baseline1.py      # Init-Only Prop.
python scripts/baselines/eval_baseline2.py      # Frame-wise Det.
# τ₂ hyper-parameter ablation
python scripts/egomed_agent/tau2_ablation/egomed-agent-iou0.py   # likewise iou03 / iou09
# text-prompt baselines
python scripts/baselines/eval_grounded_sam2_image_text_prompt_egomed5_save_outputs.py
python scripts/baselines/eval_langsam_image_text_prompt_egomed5_save_outputs.py
python scripts/baselines/eval_medsam3_image_text_prompt_egomed5_save_outputs.py
python scripts/baselines/eval_sam3_image_text_prompt_egomed5_save_outputs.py
# upper bound
python scripts/baselines/eval_nnunet_egomed5_test_dice.py
# Target Confirmation (Table III, needs DeepSeek)
python scripts/confirmation_agent/run_target_confirmation_eval_detector_generated_v5_balanced200.py
```
Outputs are written under `runs/`.

## Known reproduction limits
1. Training uses no random seed (mitigated by providing the weights).
2. The DeepSeek API is non-deterministic and models may be retired; the repo defaults to `deepseek-v4-flash` (the model reported in the paper).
3. LISA is evaluated inside its own repo; there is no script for it here.

## Path configuration

All scripts resolve the repository root automatically (the directory that contains
`setup.py`), so running them from a clone works out of the box. Put the dataset under
`<repo>/data/` and the SAM 2 checkpoints under `<repo>/checkpoints/` (see
`checkpoints/download_ckpts.sh`).

To store data or outputs elsewhere, override the roots via environment variables:

```
export EGOMED_ROOT=/path/to/EgoMed-Agent       # repo root: data/, runs/, checkpoints/
export EGOMED_EXT_ROOT=/path/to/external/repos # only needed to run the comparison baselines
```

The comparison baselines additionally require these external tools cloned under
`$EGOMED_EXT_ROOT`: `sam3`, `MedSAM3`, `Grounded-SAM-2`, `lang-segment-anything`.
