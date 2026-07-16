import argparse
import csv
import os
import re
import shutil
import sys
import time

try:
    from .iter_schedule import count_nonempty_lines, compute_powerlaw_schedule
except ImportError:
    from iter_schedule import count_nonempty_lines, compute_powerlaw_schedule


def _default_repo_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def _default_voc_workspace_root(repo_root, workspace_name):
    return os.path.join(
        os.path.abspath(repo_root),
        "pipelines",
        "generate_r4rr_maps",
        "voc_workspaces",
        workspace_name,
    )


def _resolve_paths(repo_root, voc_workspace_root=None, workspace_name="waterbirds"):
    repo_root = os.path.abspath(repo_root)
    candidates = [
        os.path.join(repo_root, "WeCLIPPlus"),
        os.path.join(repo_root, "code", "WeCLIPPlus"),
    ]
    weclip_root = next((p for p in candidates if os.path.isdir(p)), None)
    if weclip_root is None:
        raise FileNotFoundError(
            "Could not locate WeCLIPPlus. Expected one of: "
            f"{candidates[0]} or {candidates[1]}"
        )
    workspace_root = (
        os.path.abspath(voc_workspace_root)
        if voc_workspace_root
        else _default_voc_workspace_root(repo_root, workspace_name)
    )
    voc_root = os.path.join(workspace_root, "VOCdevkit", "VOC2012")
    return {
        "workspace_root": workspace_root,
        "weclip_root": weclip_root,
        "config": os.path.join(weclip_root, "configs", "voc_attn_reg.yaml"),
        "config_dir": os.path.join(workspace_root, "configs"),
        "voc_root": voc_root,
        "set_dir": os.path.join(voc_root, "ImageSets", "Main"),
        "dest_dir": os.path.join(voc_root, "JPEGImages"),
        "train_work_dir": os.path.join(workspace_root, "work_dir_voc"),
        "clip_pretrain_path": os.path.join(weclip_root, "pretrained", "ViT-B-16.pt"),
    }


def _ensure_weclip_import_path(weclip_root):
    weclip_root = os.path.abspath(weclip_root)
    if weclip_root not in sys.path:
        sys.path.insert(0, weclip_root)


def _format_elapsed(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rem = seconds - (hours * 3600 + minutes * 60)
    if hours:
        return f"{hours}h {minutes:02d}m {rem:05.2f}s"
    if minutes:
        return f"{minutes}m {rem:05.2f}s"
    return f"{rem:.2f}s"


def _resolve_clip_backend_and_pretrained(
    clip_backend,
    clip_pretrained,
    clip_backend_explicit=False,
):
    backend = (clip_backend or "openclip").strip().lower()
    if backend not in {"openai", "openclip", "siglip2"}:
        raise ValueError(
            f"Unsupported --clip-backend '{clip_backend}'. "
            "Expected one of: openai, openclip, siglip2."
        )

    pretrained = clip_pretrained
    if isinstance(pretrained, str):
        pretrained = pretrained.strip()
    if pretrained == "":
        pretrained = None

    if backend == "openclip" and pretrained is None:
        # Default run (no --clip-backend flag): OpenCLIP with OpenAI weights.
        # Explicit --clip-backend openclip/OpenCLIP: prefer LAION defaults.
        pretrained = None if clip_backend_explicit else "openai"

    if backend == "siglip2" and pretrained in (None, "metaclip_fullcc"):
        # Let adapter auto-select SigLIP2 defaults.
        pretrained = None

    return backend, pretrained


def _write_runtime_config(base_config, output_dir, voc_root, clip_pretrain_path, clip_pretrained=None):
    os.makedirs(output_dir, exist_ok=True)
    with open(base_config, "r") as f:
        content = f.read()

    name_list_dir = os.path.join(voc_root, "ImageSets", "Main")

    content = re.sub(
        r"(root_dir:\s*')([^']*)(')",
        rf"\1{voc_root}\3",
        content,
    )
    content = re.sub(
        r"(name_list_dir:\s*')([^']*)(')",
        rf"\1{name_list_dir}\3",
        content,
    )
    content = re.sub(
        r"(clip_pretrain_path:\s*')([^']*)(')",
        rf"\1{clip_pretrain_path}\3",
        content,
    )
    if clip_pretrained is not None:
        if re.search(r"clip_pretrained:\s*'[^']*'", content):
            content = re.sub(
                r"(clip_pretrained:\s*')([^']*)(')",
                rf"\1{clip_pretrained}\3",
                content,
            )
        else:
            content = re.sub(
                r"(clip_pretrain_path:\s*'[^']*'\n)",
                rf"\1  clip_pretrained: '{clip_pretrained}'\n",
                content,
            )

    output_path = os.path.join(output_dir, "voc_attn_reg_runtime.yaml")
    with open(output_path, "w") as f:
        f.write(content)
    return output_path


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _iter_image_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in _IMAGE_EXTS:
                yield os.path.join(dirpath, fname)


def _make_image_id(src_img_dir, image_path):
    rel_path = os.path.relpath(image_path, src_img_dir)
    rel_no_ext = os.path.splitext(rel_path)[0]
    flat = rel_no_ext.replace(os.sep, "_").replace("/", "_")
    flat = re.sub(r"[^A-Za-z0-9_-]+", "_", flat).strip("_")
    return flat


def _default_waterbirds_workspace_name(src_img_dir):
    src_dir = os.path.normpath(os.path.abspath(src_img_dir or ""))
    base = os.path.basename(src_dir) or "dataset"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    if not slug:
        slug = "dataset"
    return f"waterbirds_{slug}"


def _write_imagesets(set_dir, class_name, train_basenames, val_basenames):
    os.makedirs(set_dir, exist_ok=True)
    train_path = os.path.join(set_dir, "train.txt")
    val_path = os.path.join(set_dir, "val.txt")
    cls_train_path = os.path.join(set_dir, f"{class_name}_train.txt")
    cls_val_path = os.path.join(set_dir, f"{class_name}_val.txt")

    with open(train_path, "w") as f_train:
        f_train.write("\n".join(train_basenames) + "\n")
    with open(val_path, "w") as f_val:
        f_val.write("\n".join(val_basenames) + "\n")

    with open(cls_train_path, "w") as f_cls_train:
        f_cls_train.writelines(f"{b} 1\n" for b in train_basenames)
    with open(cls_val_path, "w") as f_cls_val:
        f_cls_val.writelines(f"{b} 1\n" for b in val_basenames)


def _load_waterbirds_split_rel_no_exts(src_img_dir):
    """Return (train_rel_no_exts, eval_rel_no_exts) from metadata.csv.

    train split: split==0
    eval split: split in {0, 1} (train + val)
    """
    metadata_csv = os.path.join(src_img_dir, "metadata.csv")
    if not os.path.isfile(metadata_csv):
        return None, None

    train_rel_no_exts = set()
    eval_rel_no_exts = set()

    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "img_filename" not in fieldnames or "split" not in fieldnames:
            raise ValueError(
                f"metadata.csv missing required columns 'img_filename' and 'split': {metadata_csv}"
            )

        for row in reader:
            split_raw = str(row.get("split", "")).strip().lower()
            rel = str(row["img_filename"]).strip().replace("\\", "/")
            rel_no_ext = os.path.splitext(rel)[0]

            if split_raw in {"0", "0.0", "train"}:
                train_rel_no_exts.add(rel_no_ext)
                eval_rel_no_exts.add(rel_no_ext)
            elif split_raw in {"1", "1.0", "val", "valid", "validation"}:
                eval_rel_no_exts.add(rel_no_ext)

    if not train_rel_no_exts:
        raise RuntimeError(
            f"No training rows found in metadata.csv split column at: {metadata_csv}"
        )
    if not eval_rel_no_exts:
        raise RuntimeError(
            f"No train/val rows found in metadata.csv split column at: {metadata_csv}"
        )

    return train_rel_no_exts, eval_rel_no_exts


def _prepare_single_class_dataset(
    src_img_dir,
    class_name,
    set_dir,
    dest_dir,
    train_rel_no_exts=None,
    eval_rel_no_exts=None,
    copy_images=True,
):
    os.makedirs(dest_dir, exist_ok=True)
    all_basenames = []
    seen = set()
    rel_no_ext_to_unique = {}

    for image_path in _iter_image_files(src_img_dir):
        base_id = _make_image_id(src_img_dir, image_path)
        unique_id = base_id
        suffix = 1
        while unique_id in seen:
            unique_id = f"{base_id}_{suffix}"
            suffix += 1
        seen.add(unique_id)

        ext = os.path.splitext(image_path)[1].lower() or ".jpg"
        dst_path = os.path.join(dest_dir, unique_id + ext)
        if not os.path.exists(dst_path):
            if copy_images:
                shutil.copyfile(image_path, dst_path)
            else:
                shutil.move(image_path, dst_path)

        rel_no_ext = os.path.splitext(os.path.relpath(image_path, src_img_dir))[0].replace("\\", "/")
        rel_no_ext_to_unique[rel_no_ext] = unique_id
        all_basenames.append(unique_id)

    all_basenames = sorted(all_basenames)
    if not all_basenames:
        print(f"No images found under {src_img_dir}")
        return

    if train_rel_no_exts is None:
        train_basenames = list(all_basenames)
    else:
        train_basenames = sorted(
            rel_no_ext_to_unique[rel_no_ext]
            for rel_no_ext in train_rel_no_exts
            if rel_no_ext in rel_no_ext_to_unique
        )
        if not train_basenames:
            raise RuntimeError(
                "Resolved zero training images from metadata split. "
                "Check that --src-img-dir points to the Waterbirds dataset root."
            )

    if eval_rel_no_exts is None:
        val_basenames = list(all_basenames)
    else:
        val_basenames = sorted(
            rel_no_ext_to_unique[rel_no_ext]
            for rel_no_ext in eval_rel_no_exts
            if rel_no_ext in rel_no_ext_to_unique
        )
        if not val_basenames:
            raise RuntimeError(
                "Resolved zero train/val images from metadata split. "
                "Check that --src-img-dir points to the Waterbirds dataset root."
            )

    _write_imagesets(set_dir, class_name, train_basenames, val_basenames)


def main(
    repo_root,
    voc_workspace_root,
    src_img_dir,
    setup_data,
    class_name,
    clip_backend,
    clip_model,
    clip_pretrained,
    results_dir,
    clip_backend_explicit=False,
    seed=None,
):
    clip_backend_effective, clip_pretrained_effective = _resolve_clip_backend_and_pretrained(
        clip_backend=clip_backend,
        clip_pretrained=clip_pretrained,
        clip_backend_explicit=clip_backend_explicit,
    )
    if class_name:
        os.environ["CLIP_TEXT_VERSION"] = class_name
    if clip_backend_effective:
        os.environ["CLIP_BACKEND"] = clip_backend_effective
    if clip_model:
        os.environ["CLIP_MODEL_NAME"] = clip_model
    else:
        os.environ.pop("CLIP_MODEL_NAME", None)

    if clip_pretrained_effective is not None:
        os.environ["CLIP_PRETRAINED"] = str(clip_pretrained_effective)
    else:
        os.environ.pop("CLIP_PRETRAINED", None)
    if seed is not None:
        os.environ["WECLIP_RUN_SEED"] = str(int(seed))

    paths = _resolve_paths(
        repo_root,
        voc_workspace_root=voc_workspace_root,
        workspace_name=_default_waterbirds_workspace_name(src_img_dir),
    )
    _ensure_weclip_import_path(paths["weclip_root"])

    from move_data import moveImageSets, convert_to_jpg
    from scripts import dist_clip_voc
    import test_msc_flip_voc
    from omegaconf import OmegaConf

    dist_clip_voc.args.work_dir = paths["train_work_dir"]
    print(f"Using VOC workspace: {paths['workspace_root']}")

    config = _write_runtime_config(
        paths["config"],
        paths["config_dir"],
        paths["voc_root"],
        clip_model or paths["clip_pretrain_path"],
        clip_pretrained_effective,
    )

    if setup_data:
        print("Setting up data")
        os.makedirs(paths["set_dir"], exist_ok=True)
        moveImageSets.main(paths["set_dir"])
        train_rel_no_exts, eval_rel_no_exts = _load_waterbirds_split_rel_no_exts(src_img_dir)
        _prepare_single_class_dataset(
            src_img_dir,
            class_name,
            paths["set_dir"],
            paths["dest_dir"],
            train_rel_no_exts=train_rel_no_exts,
            eval_rel_no_exts=eval_rel_no_exts,
            copy_images=True,
        )
    else:
        print("Skipping Setup")

    train_txt = os.path.join(paths["set_dir"], "train.txt")
    if not os.path.isfile(train_txt):
        raise FileNotFoundError(
            "ImageSets/Main/train.txt missing. Run once with --setup-data."
        )

    runtime_cfg = OmegaConf.load(config)
    b_eff = int(runtime_cfg.train.samples_per_gpu)
    schedule = compute_powerlaw_schedule(
        n_train=count_nonempty_lines(train_txt),
        b_eff=b_eff,
    )
    runtime_cfg.train.max_iters = int(schedule["max_iters"])
    runtime_cfg.train.cam_iters = int(schedule["cam_iters"])
    runtime_cfg.train.eval_iters = int(schedule["eval_iters"])
    OmegaConf.save(runtime_cfg, config)

    print(
        "Runtime schedule: "
        f"N_train={schedule['n_train']}, "
        f"B_eff={schedule['b_eff']}, "
        f"steps/epoch={schedule['steps_per_epoch']}, "
        f"max_iters={schedule['max_iters']}, "
        f"cam_iters={schedule['cam_iters']}, "
        f"eval_iters={schedule['eval_iters']}, "
        f"coverage_guaranteed={schedule['coverage_guaranteed']}"
    )

    print(
        f"Runtime CLIP: backend={clip_backend_effective}, "
        f"model={clip_model or paths['clip_pretrain_path']}, "
        f"pretrained={clip_pretrained_effective}"
    )
    print(
        f"Runtime seed: {seed if seed is not None else os.environ.get('WECLIP_RUN_SEED', '1')}"
    )

    convert_to_jpg.convert_to_jpg(paths["dest_dir"], True)
    final_path = dist_clip_voc.main(config)
    if results_dir:
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(paths["weclip_root"], results_dir)
        test_msc_flip_voc.args.work_dir = results_dir
    test_msc_flip_voc.outer_main(final_path, config, cfg_override=runtime_cfg)


if __name__ == "__main__":
    _t0 = time.perf_counter()
    _status = "completed"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=_default_repo_root(),
        help="Absolute path to the LearningToLook repo root.",
    )
    parser.add_argument(
        "--voc-workspace-root",
        default=None,
        help=(
            "Workspace root for this run's VOCdevkit/config/work_dir isolation. "
            "Default: <repo-root>/pipelines/generate_r4rr_maps/voc_workspaces/"
            "waterbirds_<src-img-dir-basename>"
        ),
    )
    parser.add_argument(
        "--src-img-dir",
        default="/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2",
        help="Dataset root (expects class subfolders with images).",
    )
    parser.add_argument(
        "--class-name",
        default="bird",
        help="Single foreground class name for Waterbirds (default: bird).",
    )
    parser.add_argument(
        "--setup-data",
        dest="setup_data",
        action="store_true",
        help="Run data setup steps (ImageSets + image moves).",
    )
    parser.add_argument(
        "--no-setup-data",
        dest="setup_data",
        action="store_false",
        help="Skip data setup steps.",
    )
    parser.add_argument(
        "--clip-pretrained",
        default=None,
        help=(
            "Optional pretrained tag override (e.g., openai, laion2b_s34b_b88k, webli). "
            "Default run uses OpenCLIP+OpenAI weights. If you explicitly pass "
            "--clip-backend OpenCLIP/openclip and omit this flag, LAION weights are auto-selected."
        ),
    )
    parser.add_argument(
        "--clip-backend",
        default="openclip",
        type=str.lower,
        choices=["openai", "openclip", "siglip2"],
        help=(
            "CLIP backend family. Default behavior is OpenCLIP+OpenAI weights. "
            "If this flag is explicitly set to OpenCLIP/openclip with no --clip-pretrained, "
            "LAION weights are used."
        ),
    )
    parser.add_argument(
        "--clip-model",
        default=None,
        help="Optional model override (e.g., ViT-B-16-SigLIP2).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory for predictions (prediction_cmap will be under <results-dir>/val).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional run seed override for WeCLIP+ training (default: 1).",
    )
    parser.set_defaults(setup_data=False)
    args = parser.parse_args()
    clip_backend_explicit = any(
        token == "--clip-backend" or token.startswith("--clip-backend=")
        for token in sys.argv[1:]
    )

    try:
        main(
            args.repo_root,
            args.voc_workspace_root,
            args.src_img_dir,
            args.setup_data,
            args.class_name,
            args.clip_backend,
            args.clip_model,
            args.clip_pretrained,
            args.results_dir,
            clip_backend_explicit,
            args.seed,
        )
    except Exception:
        _status = "failed"
        raise
    finally:
        _elapsed = time.perf_counter() - _t0
        print(
            f"[generate_pseudo_masks_waterbirds] {_status} in "
            f"{_format_elapsed(_elapsed)} ({_elapsed:.2f}s)"
        )
