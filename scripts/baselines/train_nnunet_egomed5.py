import os

# --- portable paths (was hardcoded); override via EGOMED_ROOT / EGOMED_EXT_ROOT ---
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(_os.environ.get("EGOMED_ROOT") or next((_a for _a in _Path(__file__).resolve().parents if (_a / "setup.py").exists()), _Path(__file__).resolve().parents[1]))
import shutil
import subprocess
from pathlib import Path


# ============================================================
# Config
# ============================================================

DATASET_ID = 501
DATASET_NAME = "Dataset501_EgoMed5"

BASE_DIR = Path(f"{REPO_ROOT}")

NNUNET_RAW = BASE_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = BASE_DIR / "nnUNet_results"

RAW_DATASET_DIR = NNUNET_RAW / DATASET_NAME
PREPROCESSED_DATASET_DIR = NNUNET_PREPROCESSED / DATASET_NAME

# 你的 prepare 脚本生成的 split 文件
SPLITS_FINAL = PREPROCESSED_DATASET_DIR / "splits_final.json"

# 如果你之前额外备份了 split，可以填这里；没有就保持 None
# BACKUP_SPLITS_FINAL = BASE_DIR / "splits_final_egomed5.json"
BACKUP_SPLITS_FINAL = None

CONFIG = "2d"
FOLD = 0
GPU_ID = "0"

VERIFY_DATASET_INTEGRITY = True

# 如果 plan/preprocess 已经跑过，并且你只想重新训练，可以改成 False
RUN_PLAN_AND_PREPROCESS = True

# 如果训练中断后想继续，改成 True，会加 --c
CONTINUE_TRAINING = False


# ============================================================
# Utilities
# ============================================================

def run_cmd(cmd, env=None):
    print("\n" + "=" * 100)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 100)

    subprocess.run(cmd, check=True, env=env)


def main():
    # ------------------------------------------------------------
    # 1. Set environment variables
    # ------------------------------------------------------------
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)

    # 指定物理 GPU 1
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    print("=" * 100)
    print("nnU-Net v2 EgoMed5 Training")
    print("=" * 100)
    print(f"nnUNet_raw:          {env['nnUNet_raw']}")
    print(f"nnUNet_preprocessed: {env['nnUNet_preprocessed']}")
    print(f"nnUNet_results:      {env['nnUNet_results']}")
    print(f"CUDA_VISIBLE_DEVICES:{env['CUDA_VISIBLE_DEVICES']}")
    print(f"Raw dataset dir:     {RAW_DATASET_DIR}")
    print(f"Dataset ID:          {DATASET_ID}")
    print(f"Config:              {CONFIG}")
    print(f"Fold:                {FOLD}")
    print("=" * 100)

    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Raw nnU-Net dataset not found: {RAW_DATASET_DIR}\n"
            f"Please run prepare_nnunet_egomed5.py first."
        )

    dataset_json = RAW_DATASET_DIR / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json}")

    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 2. Plan and preprocess
    # ------------------------------------------------------------
    if RUN_PLAN_AND_PREPROCESS:
        cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(DATASET_ID),
        ]

        if VERIFY_DATASET_INTEGRITY:
            cmd.append("--verify_dataset_integrity")

        run_cmd(cmd, env=env)
    else:
        print("\n[Skip] RUN_PLAN_AND_PREPROCESS=False")

    # ------------------------------------------------------------
    # 3. Ensure splits_final.json exists
    # ------------------------------------------------------------
    if not SPLITS_FINAL.exists():
        if BACKUP_SPLITS_FINAL is not None and Path(BACKUP_SPLITS_FINAL).exists():
            PREPROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BACKUP_SPLITS_FINAL, SPLITS_FINAL)
            print(f"[Copied] {BACKUP_SPLITS_FINAL} -> {SPLITS_FINAL}")
        else:
            raise FileNotFoundError(
                f"splits_final.json not found: {SPLITS_FINAL}\n"
                f"Your prepare_nnunet_egomed5.py should have generated it. "
                f"If plan_and_preprocess removed it, rerun prepare_nnunet_egomed5.py "
                f"or set BACKUP_SPLITS_FINAL."
            )
    else:
        print(f"[OK] Found split file: {SPLITS_FINAL}")

    # ------------------------------------------------------------
    # 4. Train nnU-Net
    # ------------------------------------------------------------
    train_cmd = [
        "nnUNetv2_train",
        str(DATASET_ID),
        CONFIG,
        str(FOLD),
    ]

    if CONTINUE_TRAINING:
        train_cmd.append("--c")

    run_cmd(train_cmd, env=env)

    print("\nDone.")
    print(f"Results should be under: {NNUNET_RESULTS}")


if __name__ == "__main__":
    main()