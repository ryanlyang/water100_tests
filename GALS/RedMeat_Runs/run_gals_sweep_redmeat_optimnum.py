import argparse
import csv
import errno
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Ensure local package imports work no matter the cwd.
GALS_ROOT = Path(__file__).resolve().parent.parent
if str(GALS_ROOT) not in sys.path:
    sys.path.insert(0, str(GALS_ROOT))

from RedMeat_Runs.optimnum_metric_redmeat import compute_main_checkpoint_optimnum


FLOAT_RE = re.compile(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def _safe_write_and_flush(log_file, text, *, retries=3, sleep_s=0.05):
    """
    Best-effort write helper for transient non-blocking filesystem errors.
    Returns True on success and False when retries are exhausted.
    """
    for attempt in range(retries):
        try:
            log_file.write(text)
            log_file.flush()
            return True
        except BlockingIOError:
            if attempt == retries - 1:
                return False
            time.sleep(sleep_s)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                if attempt == retries - 1:
                    return False
                time.sleep(sleep_s)
                continue
            raise
    return False


def _range_around(center, lo_factor=0.1, hi_factor=10.0, fallback=None):
    """
    Build a log-scale range around `center` as [center*lo_factor, center*hi_factor].
    If center is missing/invalid, use fallback=(min, max).
    """
    if center is None:
        return fallback
    try:
        c = float(center)
    except Exception:
        return fallback
    if c <= 0:
        return fallback
    lo = c * float(lo_factor)
    hi = c * float(hi_factor)
    if lo <= 0 or hi <= 0 or lo >= hi:
        return fallback
    return lo, hi


def loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sanitize_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")


def parse_csv_list(text):
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def upsert_override(overrides, key, value):
    key_eq = f"{key}="
    new_override = f"{key}={value}"
    out = []
    replaced = False
    for ov in (overrides or []):
        if ov.startswith(key_eq):
            if not replaced:
                out.append(new_override)
                replaced = True
            continue
        out.append(ov)
    if not replaced:
        out.append(new_override)
    return out


def _maybe_import_omegaconf():
    try:
        from omegaconf import OmegaConf  # type: ignore

        return OmegaConf
    except Exception:
        return None


def _infer_dataset_and_attention_dir(config_path):
    OmegaConf = _maybe_import_omegaconf()
    if OmegaConf is None:
        return None, None
    try:
        cfg = OmegaConf.load(config_path)
    except Exception:
        return None, None

    dataset = None
    attention_dir = None
    try:
        dataset = str(cfg.DATA.DATASET)
    except Exception:
        dataset = None
    try:
        attention_dir = str(cfg.DATA.ATTENTION_DIR)
    except Exception:
        attention_dir = None
    return dataset, attention_dir


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_runtime_summary(tag, rows, num_epochs=None):
    secs = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    if not secs:
        print(f"[TIME] {tag}: no successful trials to summarize.", flush=True)
        return

    arr = np.array(secs, dtype=float)
    med_trial_min = float(np.median(arr) / 60.0)
    total_gpu_hours = float(np.sum(arr) / 3600.0)
    print(
        f"[TIME] {tag}: median min/trial={med_trial_min:.2f} | total tuning GPU-hours={total_gpu_hours:.2f}",
        flush=True,
    )
    if num_epochs is not None and num_epochs > 0:
        med_epoch_min = float(np.median(arr / float(num_epochs)) / 60.0)
        print(
            f"[TIME] {tag}: median min/epoch={med_epoch_min:.4f} (epochs/trial={int(num_epochs)})",
            flush=True,
        )


def _parse_metric_line(line):
    # Matches lines like: "balanced_val_acc: 0.8123" or "test_acc: 0.901"
    parts = line.strip().split(":")
    if len(parts) < 2:
        return None
    key = parts[0].strip()
    m = FLOAT_RE.search(line)
    if not key or m is None:
        return None
    return key, float(m.group(1))


def run_one_trial(
    trial_id,
    *,
    run_name=None,
    method="gals",
    config,
    data_root,
    dataset_dir,
    dataset_name,
    train_seed,
    base_lr,
    classifier_lr,
    grad_weight=None,
    grad_criterion=None,
    cam_weight=None,
    abn_cls_weight=None,
    abn_att_weight=None,
    weight_decay,
    python_exe,
    logs_dir,
    extra_overrides=None,
    optim_beta=1.0,
):
    if run_name is None:
        run_name = f"{method}_trial_{trial_id:03d}"
    data_root = os.path.abspath(os.path.expanduser(str(data_root)))
    dataset_dir = str(dataset_dir)

    cmd = [
        python_exe,
        "-u",
        "main.py",
        "--config",
        config,
        "--name",
        run_name,
        f"DATA.ROOT={data_root}",
        f"DATA.FOOD_SUBSET_DIR={dataset_dir}",
        f"DATA.SUBDIR={dataset_dir}",
        f"SEED={train_seed}",
        f"EXP.BASE.LR={base_lr}",
        f"EXP.CLASSIFIER.LR={classifier_lr}",
        f"EXP.WEIGHT_DECAY={weight_decay}",
        # Validation objective is balanced val accuracy. Keeping aux guidance losses off on val
        # avoids expensive second-order gradient graphs and reduces random cuDNN/OOM failures.
        "EXP.AUX_LOSSES_ON_VAL=False",
    ]
    if method in ("gals", "rrr"):
        if grad_weight is None:
            raise ValueError("grad_weight must be provided when method is gals/rrr")
        if grad_criterion is None:
            grad_criterion = "L1"
        cmd.append(f"EXP.LOSSES.GRADIENT_OUTSIDE.WEIGHT={grad_weight}")
        cmd.append(f"EXP.LOSSES.GRADIENT_OUTSIDE.CRITERION={grad_criterion}")
    elif method == "gradcam":
        if cam_weight is None:
            raise ValueError("cam_weight must be provided when method='gradcam'")
        cmd.append(f"EXP.LOSSES.GRADCAM.WEIGHT={cam_weight}")
    elif method == "abn_cls":
        if abn_cls_weight is None:
            raise ValueError("abn_cls_weight must be provided when method='abn_cls'")
        cmd.append(f"EXP.LOSSES.ABN_CLASSIFICATION.WEIGHT={abn_cls_weight}")
    elif method == "abn_att":
        if abn_att_weight is None:
            raise ValueError("abn_att_weight must be provided when method='abn_att'")
        cmd.append(f"EXP.LOSSES.ABN_SUPERVISION.WEIGHT={abn_att_weight}")
    elif method == "upweight":
        pass
    else:
        raise ValueError(f"Unknown method: {method}")

    if extra_overrides:
        cmd.extend(list(extra_overrides))

    os.makedirs(logs_dir, exist_ok=True)
    trial_log = os.path.join(logs_dir, f"{run_name}.log")

    best_balanced_val = None
    test_metrics = {}
    checkpoint_used = None

    run_dir = os.path.join("trained_weights", dataset_name, run_name)
    start = time.time()
    with open(trial_log, "w") as lf:
        _safe_write_and_flush(lf, f"[CMD] {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        write_warned = False
        for line in proc.stdout:
            if not _safe_write_and_flush(lf, line) and not write_warned:
                print(
                    f"[WARN] Trial {trial_id}: log write temporarily unavailable; continuing run.",
                    flush=True,
                )
                write_warned = True

            if "BALANCED VAL ACC:" in line:
                m = FLOAT_RE.search(line)
                if m:
                    v = float(m.group(1))
                    if best_balanced_val is None or v > best_balanced_val:
                        best_balanced_val = v

            if line.startswith("TEST SET RESULTS FOR CHECKPOINT"):
                # example: TEST SET RESULTS FOR CHECKPOINT path
                checkpoint_used = line.strip().split("CHECKPOINT", 1)[-1].strip()

            parsed = _parse_metric_line(line)
            if parsed:
                k, v = parsed
                if k.endswith("_test_acc") or k in ("test_acc", "balanced_test_acc", "balanced_val_acc"):
                    test_metrics[k] = v

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Trial {trial_id} failed with exit code {rc}. See {trial_log}")

    # Compute per-group and worst-group on test if available
    group_keys = [
        k for k in test_metrics.keys()
        if (k.endswith("_test_acc") and k not in ("balanced_test_acc",))
        or k.startswith("test_acc_")
    ]
    group_vals = [test_metrics[k] for k in group_keys]
    per_group = float(np.mean(group_vals)) if group_vals else None
    worst_group = float(np.min(group_vals)) if group_vals else None

    # Normalize accs to %
    def to_pct(x):
        if x is None:
            return None
        return x * 100.0 if x <= 1.0 else x

    elapsed_s = time.time() - start

    if checkpoint_used is None:
        raise RuntimeError(
            f"Trial {trial_id} finished but no checkpoint path was parsed from logs. "
            f"Cannot compute optim_value. See {trial_log}"
        )

    # Reuse the same dataset-location overrides for post-train optim metric eval.
    # Without these, optimnum_metric_redmeat reloads YAML defaults and may fall back
    # to relative paths like ./data/food-101-redmeat.
    eval_overrides = list(extra_overrides or [])
    eval_overrides = upsert_override(eval_overrides, "DATA.ROOT", data_root)
    eval_overrides = upsert_override(eval_overrides, "DATA.FOOD_SUBSET_DIR", dataset_dir)
    eval_overrides = upsert_override(eval_overrides, "DATA.SUBDIR", dataset_dir)

    try:
        optim_metrics = compute_main_checkpoint_optimnum(
            config_path=config,
            overrides=eval_overrides,
            checkpoint_path=checkpoint_used,
            method=method,
            beta=float(optim_beta),
            data_root=data_root,
            dataset_dir=dataset_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Trial {trial_id} failed while computing optim_value metrics: {exc}. "
            f"See {trial_log}"
        ) from exc

    return {
        "trial": trial_id,
        "name": run_name,
        "method": method,
        "seed": train_seed,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "grad_weight": grad_weight,
        "grad_criterion": grad_criterion,
        "cam_weight": cam_weight,
        "abn_cls_weight": abn_cls_weight,
        "abn_att_weight": abn_att_weight,
        "weight_decay": weight_decay,
        "best_balanced_val_acc": best_balanced_val,
        "val_acc_for_optim": float(optim_metrics["val_acc"]),
        "val_ig_fwd_kl": float(optim_metrics["val_ig_fwd_kl"]),
        "log_optim_num": float(optim_metrics["log_optim_num"]),
        "optim_value": float(optim_metrics["optim_value"]),
        "optim_beta": float(optim_beta),
        "test_acc": to_pct(test_metrics.get("test_acc")),
        "balanced_test_acc": to_pct(test_metrics.get("balanced_test_acc")),
        "per_group": to_pct(per_group),
        "worst_group": to_pct(worst_group),
        "checkpoint": checkpoint_used,
        "log_path": trial_log,
        "seconds": int(elapsed_s),
        "run_dir": run_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["gals", "rrr", "gradcam", "abn_cls", "abn_att", "upweight"],
        default="gals",
        help="Which 'classifier attention method' to sweep. gals/rrr=GRADIENT_OUTSIDE, gradcam=GRADCAM, "
        "abn_cls=ABN_CLASSIFICATION, abn_att=ABN_SUPERVISION, upweight=no attention loss.",
    )
    parser.add_argument("--config", default="RedMeat_Runs/configs/redmeat_gals_vit.yaml")
    parser.add_argument("--data-root", default="/home/ryreu/guided_cnn/Food101/data")
    parser.add_argument("--dataset-dir", default="food-101-redmeat")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling hyperparameters")
    parser.add_argument("--train-seed", type=int, default=0, help="Training seed used for every trial")
    parser.add_argument("--output-csv", default="gals_sweep.csv")
    parser.add_argument("--logs-dir", default="gals_sweep_logs")
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument(
        "--keep",
        choices=["best", "all", "none"],
        default="best",
        help="What to keep under trained_weights/: best=keep best trial only, all=keep all, none=delete everything",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name used for run dirs under trained_weights/ (defaults to DATA.DATASET from config)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Optional wallclock limit; stops launching new trials once exceeded",
    )

    parser.add_argument("--weight-min", type=float, default=None, help="Min for GRADIENT_OUTSIDE weight (gals/rrr)")
    parser.add_argument("--weight-max", type=float, default=None, help="Max for GRADIENT_OUTSIDE weight (gals/rrr)")
    parser.add_argument(
        "--grad-criteria",
        default="L1,L2",
        help="Comma-separated criteria to consider for GRADIENT_OUTSIDE when method is gals/rrr.",
    )
    parser.add_argument("--cam-weight-min", type=float, default=None, help="Min for GRADCAM weight (gradcam)")
    parser.add_argument("--cam-weight-max", type=float, default=None, help="Max for GRADCAM weight (gradcam)")
    parser.add_argument("--abn-cls-weight-min", type=float, default=None, help="Min for ABN_CLASSIFICATION weight (abn_cls)")
    parser.add_argument("--abn-cls-weight-max", type=float, default=None, help="Max for ABN_CLASSIFICATION weight (abn_cls)")
    parser.add_argument("--abn-att-weight-min", type=float, default=None, help="Min for ABN_SUPERVISION weight (abn_att)")
    parser.add_argument("--abn-att-weight-max", type=float, default=None, help="Max for ABN_SUPERVISION weight (abn_att)")
    parser.add_argument("--base-lr-min", type=float, default=None)
    parser.add_argument("--base-lr-max", type=float, default=None)
    parser.add_argument("--cls-lr-min", type=float, default=None)
    parser.add_argument("--cls-lr-max", type=float, default=None)
    parser.add_argument(
        "--tune-weight-decay",
        action="store_true",
        help="If set, also tune EXP.WEIGHT_DECAY (otherwise uses the value from the YAML config).",
    )
    parser.add_argument("--weight-decay-min", type=float, default=None)
    parser.add_argument("--weight-decay-max", type=float, default=None)
    parser.add_argument(
        "--post-seeds",
        type=int,
        default=0,
        help="After the sweep, rerun the best hyperparameters for N different seeds and write a second CSV.",
    )
    parser.add_argument(
        "--post-seed-start",
        type=int,
        default=0,
        help="Starting seed for post-sweep reruns (uses seeds post_seed_start..post_seed_start+post_seeds-1).",
    )
    parser.add_argument(
        "--post-output-csv",
        default=None,
        help="CSV for post-sweep seed reruns (default: <output-csv basename>_best_seeds.csv).",
    )
    parser.add_argument(
        "--post-logs-dir",
        default=None,
        help="Logs dir for post-sweep seed reruns (default: <logs-dir>_best_seeds).",
    )
    parser.add_argument(
        "--post-keep",
        choices=["all", "none"],
        default="all",
        help="What to keep for the post-sweep seed reruns under trained_weights/: all or none.",
    )
    parser.add_argument(
        "--post-segmentation-dirs",
        default="",
        help=(
            "Comma-separated extra segmentation directories for additional post-sweep reruns. "
            "Each phase overrides DATA.SEGMENTATION_DIR to the given path."
        ),
    )
    parser.add_argument(
        "--post-segmentation-labels",
        default="",
        help="Comma-separated labels for --post-segmentation-dirs (must match count if provided).",
    )
    parser.add_argument(
        "--run-name-prefix",
        default=None,
        help=(
            "Prefix for all run names/checkpoint dirs. "
            "Default is auto-generated from method, dataset dir, and SLURM_JOB_ID."
        ),
    )
    parser.add_argument(
        "--optim-beta",
        type=float,
        default=10.0,
        help="Penalty strength in log_optim_num = log(val_acc) - beta * ig_fwd_kl.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra OmegaConf overrides passed through to main.py (e.g. DATA.SEGMENTATION_DIR=/path/to/masks).",
    )
    args = parser.parse_args()

    header = [
        "trial",
        "name",
        "method",
        "seed",
        "base_lr",
        "classifier_lr",
        "grad_weight",
        "grad_criterion",
        "cam_weight",
        "abn_cls_weight",
        "abn_att_weight",
        "weight_decay",
        "best_balanced_val_acc",
        "val_acc_for_optim",
        "val_ig_fwd_kl",
        "log_optim_num",
        "optim_value",
        "optim_beta",
        "test_acc",
        "balanced_test_acc",
        "per_group",
        "worst_group",
        "checkpoint",
        "log_path",
        "seconds",
        "sampler",
        "run_dir",
    ]

    rng = np.random.default_rng(args.seed)
    python_exe = sys.executable

    inferred_dataset, inferred_attention_dir = _infer_dataset_and_attention_dir(args.config)
    dataset_name = args.dataset or inferred_dataset or "food_subset"
    dataset_tag = sanitize_name(os.path.basename(args.dataset_dir) or args.dataset_dir)
    job_tag = sanitize_name(os.environ.get("SLURM_JOB_ID", "nojid"))
    if args.run_name_prefix:
        run_name_prefix = sanitize_name(args.run_name_prefix)
    else:
        run_name_prefix = sanitize_name(f"{args.method}_{dataset_tag}_{job_tag}")
    print(f"[SWEEP] run_name_prefix={run_name_prefix}", flush=True)

    cfg_base_lr = None
    cfg_cls_lr = None
    cfg_weight_decay = None
    cfg_grad_weight = None
    cfg_cam_weight = None
    cfg_abn_cls_weight = None
    cfg_abn_att_weight = None
    cfg_num_epochs = 150
    OmegaConf = _maybe_import_omegaconf()
    if OmegaConf is not None:
        try:
            _cfg = OmegaConf.load(args.config)
            cfg_base_lr = float(_cfg.EXP.BASE.LR)
            cfg_cls_lr = float(_cfg.EXP.CLASSIFIER.LR)
            cfg_weight_decay = float(_cfg.EXP.WEIGHT_DECAY)
            cfg_num_epochs = int(_cfg.EXP.NUM_EPOCHS)
            losses = _cfg.EXP.LOSSES if "LOSSES" in _cfg.EXP else None
            if losses is not None:
                if "GRADIENT_OUTSIDE" in losses and "WEIGHT" in losses.GRADIENT_OUTSIDE:
                    cfg_grad_weight = float(losses.GRADIENT_OUTSIDE.WEIGHT)
                if "GRADCAM" in losses and "WEIGHT" in losses.GRADCAM:
                    cfg_cam_weight = float(losses.GRADCAM.WEIGHT)
                if "ABN_CLASSIFICATION" in losses and "WEIGHT" in losses.ABN_CLASSIFICATION:
                    cfg_abn_cls_weight = float(losses.ABN_CLASSIFICATION.WEIGHT)
                if "ABN_SUPERVISION" in losses and "WEIGHT" in losses.ABN_SUPERVISION:
                    cfg_abn_att_weight = float(losses.ABN_SUPERVISION.WEIGHT)
        except Exception:
            cfg_base_lr = None
            cfg_cls_lr = None
            cfg_weight_decay = None
            cfg_grad_weight = None
            cfg_cam_weight = None
            cfg_abn_cls_weight = None
            cfg_abn_att_weight = None
    if cfg_weight_decay is None:
        cfg_weight_decay = 1e-5

    # Sweep around config/paper defaults unless explicit ranges are provided.
    if args.base_lr_min is None or args.base_lr_max is None:
        lo, hi = _range_around(cfg_base_lr, fallback=(5e-4, 5e-2))
        if args.base_lr_min is None:
            args.base_lr_min = lo
        if args.base_lr_max is None:
            args.base_lr_max = hi
    if args.cls_lr_min is None or args.cls_lr_max is None:
        lo, hi = _range_around(cfg_cls_lr, fallback=(1e-5, 1e-2))
        if args.cls_lr_min is None:
            args.cls_lr_min = lo
        if args.cls_lr_max is None:
            args.cls_lr_max = hi
    if args.weight_decay_min is None or args.weight_decay_max is None:
        lo, hi = _range_around(cfg_weight_decay, fallback=(1e-6, 1e-4))
        if args.weight_decay_min is None:
            args.weight_decay_min = lo
        if args.weight_decay_max is None:
            args.weight_decay_max = hi

    if args.method in ("gals", "rrr"):
        if args.weight_min is None or args.weight_max is None:
            lo, hi = _range_around(cfg_grad_weight, fallback=(1e3, 1e5))
            if args.weight_min is None:
                args.weight_min = lo
            if args.weight_max is None:
                args.weight_max = hi
    elif args.method == "gradcam":
        if args.cam_weight_min is None or args.cam_weight_max is None:
            lo, hi = _range_around(cfg_cam_weight, fallback=(0.1, 10.0))
            if args.cam_weight_min is None:
                args.cam_weight_min = lo
            if args.cam_weight_max is None:
                args.cam_weight_max = hi
    elif args.method == "abn_cls":
        if args.abn_cls_weight_min is None or args.abn_cls_weight_max is None:
            lo, hi = _range_around(cfg_abn_cls_weight, fallback=(0.1, 10.0))
            if args.abn_cls_weight_min is None:
                args.abn_cls_weight_min = lo
            if args.abn_cls_weight_max is None:
                args.abn_cls_weight_max = hi
    elif args.method == "abn_att":
        if args.abn_att_weight_min is None or args.abn_att_weight_max is None:
            lo, hi = _range_around(cfg_abn_att_weight, fallback=(0.1, 10.0))
            if args.abn_att_weight_min is None:
                args.abn_att_weight_min = lo
            if args.abn_att_weight_max is None:
                args.abn_att_weight_max = hi

    grad_criteria = []
    if args.method in ("gals", "rrr"):
        grad_criteria = [c.strip().upper() for c in str(args.grad_criteria).split(",") if c.strip()]
        # Keep order stable while deduplicating.
        grad_criteria = list(dict.fromkeys(grad_criteria))
        valid_criteria = {"L1", "L2"}
        invalid = [c for c in grad_criteria if c not in valid_criteria]
        if invalid:
            raise ValueError(f"Invalid --grad-criteria values: {invalid}. Valid: L1,L2")
        if not grad_criteria:
            grad_criteria = ["L1", "L2"]

    print(
        f"[SWEEP] Hyperparameter ranges: "
        f"base_lr=[{args.base_lr_min}, {args.base_lr_max}] "
        f"classifier_lr=[{args.cls_lr_min}, {args.cls_lr_max}] "
        f"weight_decay=[{args.weight_decay_min}, {args.weight_decay_max}]",
        flush=True,
    )
    if args.method in ("gals", "rrr"):
        print(f"[SWEEP] grad_weight=[{args.weight_min}, {args.weight_max}]", flush=True)
        print(f"[SWEEP] grad_criterion={grad_criteria}", flush=True)
    elif args.method == "gradcam":
        print(f"[SWEEP] cam_weight=[{args.cam_weight_min}, {args.cam_weight_max}]", flush=True)
    elif args.method == "abn_cls":
        print(
            f"[SWEEP] abn_cls_weight=[{args.abn_cls_weight_min}, {args.abn_cls_weight_max}]",
            flush=True,
        )
    elif args.method == "abn_att":
        print(
            f"[SWEEP] abn_att_weight=[{args.abn_att_weight_min}, {args.abn_att_weight_max}]",
            flush=True,
        )

    if inferred_attention_dir and inferred_attention_dir.upper() != "NONE":
        expected_attention_dir = os.path.join(args.data_root, args.dataset_dir, inferred_attention_dir)
        if not os.path.isdir(expected_attention_dir):
            raise FileNotFoundError(
                "Missing precomputed attention maps for GALS.\n"
                f"Expected directory: {expected_attention_dir}\n"
                "Run `extract_attention.py` first (stage 1) using the corresponding *_attention.yaml config."
            )

    post_seg_dirs = parse_csv_list(args.post_segmentation_dirs)
    post_seg_labels = parse_csv_list(args.post_segmentation_labels)
    if post_seg_labels and len(post_seg_labels) != len(post_seg_dirs):
        raise ValueError(
            "--post-segmentation-labels must have the same number of entries as --post-segmentation-dirs"
        )
    for seg_dir in post_seg_dirs:
        if not os.path.isdir(seg_dir):
            raise FileNotFoundError(f"Missing post segmentation directory: {seg_dir}")
    if post_seg_dirs:
        print(
            f"[POST] extra segmentation phases: {len(post_seg_dirs)} -> {post_seg_dirs}",
            flush=True,
        )

    if args.sampler == "tpe":
        try:
            import optuna  # noqa: F401
        except Exception as exc:
            print(f"[SWEEP] Optuna not available ({exc}); falling back to random search.", flush=True)
            args.sampler = "random"

    best_row = None
    best_dir = None
    sweep_rows = []

    def cleanup_run_dir(path):
        if not path:
            return
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

    def run_and_record(
        trial_id,
        base_lr,
        classifier_lr,
        grad_weight,
        grad_criterion,
        cam_weight,
        abn_cls_weight,
        abn_att_weight,
        weight_decay,
        sampler_name,
    ):
        nonlocal best_row, best_dir
        run_name = f"{run_name_prefix}_trial_{trial_id:03d}"
        row = run_one_trial(
            trial_id,
            run_name=run_name,
            method=args.method,
            config=args.config,
            data_root=args.data_root,
            dataset_dir=args.dataset_dir,
            dataset_name=dataset_name,
            train_seed=args.train_seed,
            base_lr=base_lr,
            classifier_lr=classifier_lr,
            grad_weight=grad_weight,
            grad_criterion=grad_criterion,
            cam_weight=cam_weight,
            abn_cls_weight=abn_cls_weight,
            abn_att_weight=abn_att_weight,
            weight_decay=weight_decay,
            python_exe=python_exe,
            logs_dir=args.logs_dir,
            extra_overrides=args.overrides,
            optim_beta=args.optim_beta,
        )
        row["sampler"] = sampler_name
        write_row(args.output_csv, row, header)
        sweep_rows.append(row)

        score = row["optim_value"]
        is_new_best = score is not None and (best_row is None or score > best_row["optim_value"])
        run_dir = row.get("run_dir")

        if args.keep == "none":
            cleanup_run_dir(run_dir)
        elif args.keep == "best":
            if is_new_best:
                cleanup_run_dir(best_dir)
                best_dir = run_dir
            else:
                cleanup_run_dir(run_dir)

        if is_new_best:
            best_row = row
        return row

    start_time = time.time()

    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            if args.max_hours is not None and (time.time() - start_time) >= args.max_hours * 3600:
                print(f"[SWEEP] Reached max-hours={args.max_hours}; stopping before trial {trial_id}.", flush=True)
                break
            base_lr = loguniform(rng, args.base_lr_min, args.base_lr_max)
            classifier_lr = loguniform(rng, args.cls_lr_min, args.cls_lr_max)
            weight_decay = (
                loguniform(rng, args.weight_decay_min, args.weight_decay_max)
                if args.tune_weight_decay
                else cfg_weight_decay
            )
            grad_weight = None
            grad_criterion = None
            cam_weight = None
            abn_cls_weight = None
            abn_att_weight = None
            if args.method in ("gals", "rrr"):
                grad_weight = loguniform(rng, args.weight_min, args.weight_max)
                grad_criterion = str(rng.choice(grad_criteria))
            elif args.method == "gradcam":
                cam_weight = loguniform(rng, args.cam_weight_min, args.cam_weight_max)
            elif args.method == "abn_cls":
                abn_cls_weight = loguniform(rng, args.abn_cls_weight_min, args.abn_cls_weight_max)
            elif args.method == "abn_att":
                abn_att_weight = loguniform(rng, args.abn_att_weight_min, args.abn_att_weight_max)
            try:
                row = run_and_record(
                    trial_id,
                    base_lr,
                    classifier_lr,
                    grad_weight,
                    grad_criterion,
                    cam_weight,
                    abn_cls_weight,
                    abn_att_weight,
                    weight_decay,
                    "random",
                )
                print(
                    f"[SWEEP] Trial {trial_id} done. log_optim_num={row['optim_value']:.6f} "
                    f"(val_acc={row['val_acc_for_optim']:.4f}, ig_fwd_kl={row['val_ig_fwd_kl']:.6f})",
                    flush=True,
                )
            except Exception as exc:
                print(f"[SWEEP] Trial {trial_id} failed: {exc}", flush=True)
    else:
        import optuna

        def objective(trial):
            base_lr = float(trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
            classifier_lr = float(trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
            weight_decay = (
                float(trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max, log=True))
                if args.tune_weight_decay
                else cfg_weight_decay
            )
            grad_weight = None
            grad_criterion = None
            cam_weight = None
            abn_cls_weight = None
            abn_att_weight = None
            if args.method in ("gals", "rrr"):
                grad_weight = float(trial.suggest_float("grad_weight", args.weight_min, args.weight_max, log=True))
                grad_criterion = str(trial.suggest_categorical("grad_criterion", grad_criteria))
            elif args.method == "gradcam":
                cam_weight = float(trial.suggest_float("cam_weight", args.cam_weight_min, args.cam_weight_max, log=True))
            elif args.method == "abn_cls":
                abn_cls_weight = float(
                    trial.suggest_float("abn_cls_weight", args.abn_cls_weight_min, args.abn_cls_weight_max, log=True)
                )
            elif args.method == "abn_att":
                abn_att_weight = float(
                    trial.suggest_float("abn_att_weight", args.abn_att_weight_min, args.abn_att_weight_max, log=True)
                )

            row = run_and_record(
                trial.number,
                base_lr,
                classifier_lr,
                grad_weight,
                grad_criterion,
                cam_weight,
                abn_cls_weight,
                abn_att_weight,
                weight_decay,
                "tpe",
            )
            print(
                f"[SWEEP] Trial {trial.number} done. log_optim_num={row['optim_value']:.6f} "
                f"(val_acc={row['val_acc_for_optim']:.4f}, ig_fwd_kl={row['val_ig_fwd_kl']:.6f})",
                flush=True,
            )
            return row["optim_value"] if row["optim_value"] is not None else -1.0

        callbacks = []
        if args.max_hours is not None:
            def _time_limit_cb(study, _trial):
                if (time.time() - start_time) >= args.max_hours * 3600:
                    print(f"[SWEEP] Reached max-hours={args.max_hours}; stopping study.", flush=True)
                    study.stop()
            callbacks.append(_time_limit_cb)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=args.n_trials,
            catch=(RuntimeError, BlockingIOError, OSError),
            callbacks=callbacks,
        )

    if best_row is not None:
        print("[SWEEP] Best trial:", flush=True)
        for k in header:
            if k in best_row:
                print(f"  {k}: {best_row[k]}", flush=True)

    post_rows = []
    if best_row is not None and args.post_seeds > 0:
        post_csv = args.post_output_csv
        if post_csv is None:
            root, ext = os.path.splitext(args.output_csv)
            post_csv = f"{root}_best_seeds{ext or '.csv'}"
        post_logs_dir = args.post_logs_dir or f"{args.logs_dir}_best_seeds"

        post_header = [
            "phase",
            "seed",
            "name",
            "method",
            "segmentation_dir",
            "base_lr",
            "classifier_lr",
            "grad_weight",
            "grad_criterion",
            "cam_weight",
            "abn_cls_weight",
            "abn_att_weight",
            "weight_decay",
            "best_balanced_val_acc",
            "val_acc_for_optim",
            "val_ig_fwd_kl",
            "log_optim_num",
            "optim_value",
            "optim_beta",
            "test_acc",
            "balanced_test_acc",
            "per_group",
            "worst_group",
            "checkpoint",
            "log_path",
            "seconds",
            "run_dir",
            "sweep_best_trial",
        ]

        seeds = list(range(args.post_seed_start, args.post_seed_start + args.post_seeds))
        post_phases = [("default", list(args.overrides or []), None)]
        for i, seg_dir in enumerate(post_seg_dirs):
            phase_label = (
                sanitize_name(post_seg_labels[i])
                if i < len(post_seg_labels)
                else f"segdir{i + 1}"
            )
            if not phase_label:
                phase_label = f"segdir{i + 1}"
            phase_overrides = upsert_override(args.overrides, "DATA.SEGMENTATION_DIR", seg_dir)
            post_phases.append((phase_label, phase_overrides, seg_dir))

        print(
            f"[POST] Rerunning best hyperparameters for {len(seeds)} seeds across {len(post_phases)} phase(s).",
            flush=True,
        )

        for phase_name, phase_overrides, phase_seg_dir in post_phases:
            for s in seeds:
                if phase_name == "default":
                    run_name = f"{run_name_prefix}_best_seed{s}"
                else:
                    run_name = f"{run_name_prefix}_{phase_name}_best_seed{s}"
                row = run_one_trial(
                    100000 + s,
                    run_name=run_name,
                    method=args.method,
                    config=args.config,
                    data_root=args.data_root,
                    dataset_dir=args.dataset_dir,
                    dataset_name=dataset_name,
                    train_seed=s,
                    base_lr=float(best_row["base_lr"]),
                    classifier_lr=float(best_row["classifier_lr"]),
                    grad_weight=float(best_row["grad_weight"]) if best_row.get("grad_weight") is not None else None,
                    grad_criterion=str(best_row["grad_criterion"]) if best_row.get("grad_criterion") is not None else None,
                    cam_weight=float(best_row["cam_weight"]) if best_row.get("cam_weight") is not None else None,
                    abn_cls_weight=float(best_row["abn_cls_weight"]) if best_row.get("abn_cls_weight") is not None else None,
                    abn_att_weight=float(best_row["abn_att_weight"]) if best_row.get("abn_att_weight") is not None else None,
                    weight_decay=float(best_row["weight_decay"]),
                    python_exe=python_exe,
                    logs_dir=post_logs_dir,
                    extra_overrides=phase_overrides,
                    optim_beta=args.optim_beta,
                )

                if args.post_keep == "none":
                    cleanup_run_dir(row.get("run_dir"))

                out_row = {
                    "phase": phase_name,
                    "seed": s,
                    "name": row.get("name"),
                    "method": row.get("method"),
                    "segmentation_dir": phase_seg_dir,
                    "base_lr": row.get("base_lr"),
                    "classifier_lr": row.get("classifier_lr"),
                    "grad_weight": row.get("grad_weight"),
                    "grad_criterion": row.get("grad_criterion"),
                    "cam_weight": row.get("cam_weight"),
                    "abn_cls_weight": row.get("abn_cls_weight"),
                    "abn_att_weight": row.get("abn_att_weight"),
                    "weight_decay": row.get("weight_decay"),
                    "best_balanced_val_acc": row.get("best_balanced_val_acc"),
                    "val_acc_for_optim": row.get("val_acc_for_optim"),
                    "val_ig_fwd_kl": row.get("val_ig_fwd_kl"),
                    "log_optim_num": row.get("log_optim_num"),
                    "optim_value": row.get("optim_value"),
                    "optim_beta": row.get("optim_beta"),
                    "test_acc": row.get("test_acc"),
                    "balanced_test_acc": row.get("balanced_test_acc"),
                    "per_group": row.get("per_group"),
                    "worst_group": row.get("worst_group"),
                    "checkpoint": row.get("checkpoint"),
                    "log_path": row.get("log_path"),
                    "seconds": row.get("seconds"),
                    "run_dir": row.get("run_dir"),
                    "sweep_best_trial": int(best_row["trial"]),
                }
                write_row(post_csv, out_row, post_header)
                post_rows.append(out_row)
                print(
                    f"[POST] phase={phase_name} seed={s} log_optim_num={out_row['optim_value']} "
                    f"per_group={out_row['per_group']} worst_group={out_row['worst_group']}",
                    flush=True,
                )

        def _mean_std(key):
            vals = [r.get(key) for r in post_rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None, None
            arr = np.array(vals, dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        m_pg, s_pg = _mean_std("per_group")
        m_wg, s_wg = _mean_std("worst_group")
        m_bte, s_bte = _mean_std("balanced_test_acc")
        print("[POST] Summary over seeds:", flush=True)
        if m_pg is not None:
            print(f"  per_group: {m_pg:.2f} +/- {s_pg:.2f}", flush=True)
        if m_wg is not None:
            print(f"  worst_group: {m_wg:.2f} +/- {s_wg:.2f}", flush=True)
        if m_bte is not None:
            print(f"  balanced_test_acc: {m_bte:.2f} +/- {s_bte:.2f}", flush=True)
        print(f"[POST] Wrote: {post_csv}", flush=True)

    if not sweep_rows:
        raise RuntimeError(
            "All sweep trials failed (no successful rows were recorded). "
            "Check per-trial logs under --logs-dir for the first failing trace."
        )

    print_runtime_summary("sweep", sweep_rows, cfg_num_epochs)
    if post_rows:
        print_runtime_summary("post_best_seeds", post_rows, cfg_num_epochs)


if __name__ == "__main__":
    main()
