#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical roots (cluster)
# Use the active Waterbird_Runs checkout by default; RedMeat data still lives under Food101/data.
PROJECT_ROOT_DEFAULT=/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs
GALS_ROOT_DEFAULT="${PROJECT_ROOT_DEFAULT}/GALS"
DATA_ROOT_DEFAULT=/home/ryreu/guided_cnn/Food101/data
DATA_DIR_DEFAULT=food-101-redmeat
LOG_DIR_DEFAULT=/home/ryreu/guided_cnn/logsRedMeat

redmeat_set_defaults() {
  export PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"
  export GALS_ROOT="${GALS_ROOT:-$GALS_ROOT_DEFAULT}"
  export DATA_ROOT="${DATA_ROOT:-$DATA_ROOT_DEFAULT}"
  export DATA_DIR="${DATA_DIR:-$DATA_DIR_DEFAULT}"
  export LOG_DIR="${LOG_DIR:-$LOG_DIR_DEFAULT}"
  mkdir -p "$LOG_DIR"
}

redmeat_activate_env() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV_NAME:-gals_a100}"

  export TF_CPP_MIN_LOG_LEVEL=3
  export TF_ENABLE_ONEDNN_OPTS=0
  export WANDB_DISABLED=true
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export PYTHONNOUSERSITE=1

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
  fi
}

redmeat_prepare_food_layout() {
  # Makes the existing GALS FoodSubset/extract_attention assumptions work
  # while keeping canonical dataset location at ${DATA_ROOT}/${DATA_DIR}.
  local data_root="$1"
  local data_dir="$2"

  local dataset_root="${data_root}/${data_dir}"
  local compat_root="${data_root}/food-101"

  if [[ ! -d "$dataset_root" ]]; then
    echo "[ERROR] Missing dataset root: $dataset_root" >&2
    return 2
  fi

  # GALS hardcodes ${DATA.ROOT}/food-101 in FoodSubset + extract_attention.py.
  if [[ ! -e "$compat_root" ]]; then
    ln -s "$dataset_root" "$compat_root"
  fi

  # GALS looks for food-101/meta/all_images.csv.
  mkdir -p "$dataset_root/meta"
  # Support either metadata layout while avoiding symlink loops:
  # 1) real file at root/all_images.csv
  # 2) real file at meta/all_images.csv
  local root_meta="$dataset_root/all_images.csv"
  local meta_meta="$dataset_root/meta/all_images.csv"
  if [[ -f "$root_meta" && ! -L "$root_meta" ]]; then
    ln -sfn ../all_images.csv "$meta_meta"
  elif [[ -f "$meta_meta" && ! -L "$meta_meta" ]]; then
    ln -sfn meta/all_images.csv "$root_meta"
  elif [[ -L "$root_meta" && -f "$root_meta" && ! -L "$meta_meta" && -f "$meta_meta" ]]; then
    ln -sfn meta/all_images.csv "$root_meta"
  elif [[ -L "$meta_meta" && -f "$meta_meta" && ! -L "$root_meta" && -f "$root_meta" ]]; then
    ln -sfn ../all_images.csv "$meta_meta"
  else
    echo "[WARN] Could not locate a real all_images.csv file in $dataset_root or $dataset_root/meta" >&2
  fi

  # If metadata is still missing (or broken), rebuild it from split_images.
  if [[ ! -f "$root_meta" ]]; then
    if [[ -d "$dataset_root/split_images/train" && -d "$dataset_root/split_images/val" && -d "$dataset_root/split_images/test" ]]; then
      echo "[INFO] Rebuilding $root_meta from split_images/{train,val,test} ..."
      python - "$dataset_root" "$root_meta" <<'PY'
import csv
import os
import sys

dataset_root, out_csv = sys.argv[1], sys.argv[2]
split_root = os.path.join(dataset_root, "split_images")
images_root = os.path.join(dataset_root, "images")

rows = []
for split in ("train", "val", "test"):
    split_dir = os.path.join(split_root, split)
    if not os.path.isdir(split_dir):
        continue
    for label in sorted(os.listdir(split_dir)):
        label_dir = os.path.join(split_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # Prefer canonical images/ path when present.
            canonical = os.path.join(images_root, label, fname)
            if os.path.isfile(canonical):
                rel_path = os.path.join("images", label, fname)
            else:
                rel_path = os.path.relpath(os.path.join(label_dir, fname), dataset_root)
            rows.append(
                {
                    "abs_file_path": rel_path.replace("\\", "/"),
                    "label": str(label),
                    "split": str(split),
                }
            )

if not rows:
    raise SystemExit("No image rows found under split_images; cannot rebuild all_images.csv")

tmp_csv = out_csv + ".tmp"
with open(tmp_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["abs_file_path", "label", "split"])
    w.writeheader()
    w.writerows(rows)
os.replace(tmp_csv, out_csv)
PY
      ln -sfn ../all_images.csv "$meta_meta"
    fi
  fi

  if [[ ! -f "$root_meta" ]]; then
    echo "[ERROR] Missing metadata file: $root_meta" >&2
    return 2
  fi

  # Some scripts initialize ImageFolder(".../train").
  if [[ ! -e "$dataset_root/train" && -d "$dataset_root/images" ]]; then
    ln -s "images" "$dataset_root/train"
  fi

  # Optional convenience links for val/test from split_images if present.
  if [[ -d "$dataset_root/split_images" ]]; then
    for split in train val test; do
      if [[ ! -e "$dataset_root/$split" && -d "$dataset_root/split_images/$split" ]]; then
        ln -s "split_images/$split" "$dataset_root/$split"
      fi
    done
  fi
}

redmeat_require_vit_attention_dir() {
  local p="$1"
  if [[ ! -d "$p" ]]; then
    echo "[ERROR] Missing ViT attentions: $p" >&2
    return 2
  fi
}

redmeat_require_mask_dir() {
  local p="$1"
  p="${p%\"}"; p="${p#\"}"; p="${p%\'}"; p="${p#\'}"
  if [[ ! -d "$p" ]]; then
    echo "[ERROR] Missing mask dir: $p" >&2
    return 2
  fi
  printf '%s\n' "$p"
}
