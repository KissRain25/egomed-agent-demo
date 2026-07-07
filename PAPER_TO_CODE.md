# Paper → Code Map (PAPER_TO_CODE)

Maps each section / table of the paper *Understanding From Human Perspective: A Multi-agent System for Interactive Egocentric Medical Image Segmentation* to the scripts under `scripts/` and the result paths under `runs/`.

> Main-result script: `scripts/egomed_agent/egomed-agent-iou06.py` (`TRACK_IOU_THRES=0.6`, i.e. τ₂=0.6 in the paper).
> Run everything from the repository root; scripts have no argparse — edit the config block at the top of each file.

## §III Method (EgoMed-Agent)

| Paper | Code |
|---|---|
| §III-B Detection Agent · image-type classifier (YOLO26-cls) | train `scripts/detection_agent/train_tool_selection_yolo26_cls.py`; eval `scripts/detection_agent/eval_tool_selection_first_frame_test.py` |
| §III-B Detection Agent · 5 per-modality detectors (YOLO26-det) | train `scripts/detection_agent/train_yolo26_egomed5_all.py`; eval `scripts/detection_agent/eval_yolo26_det_egomed5_test_full_frames.py` → `runs/yolo26_det/<DS>_yolo26m_imgsz1024/weights/best.pt` |
| §III-B Confirmation Agent (DeepSeek-V4-Flash, τ₁=0.8) | `scripts/confirmation_agent/`: LLM confirmation `run_target_confirmation_eval_*.py`, rule-based `*_rule_based.py`; eval-set construction `build_target_confirmation_eval_*.py` |
| §III-C Localization-Guided Propagation (Propagation Agent + Consistency Evaluation, τ₂=0.6) | **`scripts/egomed_agent/egomed-agent-iou06.py`** (SAM 2.1 b+ propagation + IoU-gated correction) |

## §IV Experiments

| Paper | Code / results |
|---|---|
| **Table II** main comparison (Quantitative Evaluation) | `scripts/egomed_agent/egomed-agent-iou06.py` + `scripts/baselines/*` → `runs/eval_yolo26_original_sam2_video_schedule_reset_retrack_correction/` |
| Text-prompt baselines | `scripts/baselines/`: `eval_grounded_sam2_*` · `eval_langsam_*` · `eval_medsam3_*` · `eval_sam3_*` (LISA numbers are produced in the external repo; no script here) |
| Module ablations | `scripts/baselines/eval_baseline1.py` = Init-Only Prop.; `eval_baseline2.py` = Frame-wise Det. |
| nnU-Net upper bound | `scripts/baselines/eval_nnunet_egomed5_test_dice.py` (train `train_nnunet_egomed5.py`) |
| **Fig. 5** τ₂ hyper-parameter ablation | `scripts/egomed_agent/tau2_ablation/egomed-agent-iou{0,03,09}.py` (τ₂ = 0 / 0.3 / 0.9); main script iou06 = τ₂=0.6 |
| **Table III** Target Confirmation (GSA / TCA, 200 samples / 50 per instruction type) | `scripts/confirmation_agent/run_target_confirmation_eval_*.py` + `build_target_confirmation_eval_*.py`. Final: DeepSeek-V4-Flash GSA 100 / TCA 100; DeepSeek-Chat ablation 80 / 70.5 |

> Metrics DSC / GSA / TCA: DSC is written per case/class to CSV inside each agent/baseline script; GSA / TCA are computed by the Confirmation Agent evaluation scripts.
> The dataset-construction pipeline and the paper's figure-generation scripts are **not in this repository** (internal tooling); the dataset is on Hugging Face and the collection protocol is in the paper's supplementary material.
