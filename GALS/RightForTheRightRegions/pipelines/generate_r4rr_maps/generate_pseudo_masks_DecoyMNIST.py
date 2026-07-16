import argparse
import os
import re
import sys
import time
from typing import Dict, List

from PIL import Image

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


def _resolve_paths(repo_root, voc_workspace_root=None, workspace_name="decoymnist"):
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


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def _iter_class_images(split_root: str):
    """Yield (class_name, image_path) for split_root/class_name/*.ext."""
    if not os.path.isdir(split_root):
        return
    for class_name in sorted(os.listdir(split_root)):
        class_dir = os.path.join(split_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            src = os.path.join(class_dir, fname)
            if not os.path.isfile(src):
                continue
            if os.path.splitext(fname)[1].lower() not in _IMAGE_EXTS:
                continue
            yield class_name, src


def _sanitize_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def _build_id(split_name: str, source_class: str, image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return (
        f"{_sanitize_token(split_name)}_"
        f"{_sanitize_token(source_class)}_"
        f"{_sanitize_token(stem)}"
    )


def _prepare_decoymnist_dataset(
    src_png_root: str,
    class_name: str,
    set_dir: str,
    dest_dir: str,
) -> Dict[str, int]:
    """
    Ingest DecoyMNIST_png/{train,test}/<digit>/*.png into VOC-style JPEGImages.
    train list uses train split only; val list also uses train split only
    so map export runs on the 60k training images.
    """
    os.makedirs(set_dir, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)

    split_roots = [
        ("train", os.path.join(src_png_root, "train")),
        ("test", os.path.join(src_png_root, "test")),
    ]
    for _, split_root in split_roots:
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Missing required split directory: {split_root}")

    val_basenames: List[str] = []
    train_basenames: List[str] = []
    seen_ids = set()
    copied = 0
    skipped = 0

    for split_name, split_root in split_roots:
        for src_class, image_path in _iter_class_images(split_root):
            base_id = _build_id(split_name, src_class, image_path)
            unique_id = base_id
            suffix = 1
            while unique_id in seen_ids:
                unique_id = f"{base_id}_{suffix}"
                suffix += 1
            seen_ids.add(unique_id)

            dst_path = os.path.join(dest_dir, unique_id + ".jpg")
            if os.path.exists(dst_path):
                skipped += 1
            else:
                Image.open(image_path).convert("RGB").save(dst_path, "JPEG", quality=95)
                copied += 1

            if split_name == "train":
                train_basenames.append(unique_id)

    train_basenames = sorted(train_basenames)
    val_basenames = list(train_basenames)
    if not val_basenames:
        raise RuntimeError(
            f"No images found under train/test class folders in: {src_png_root}"
        )
    if not train_basenames:
        raise RuntimeError(
            f"No train images found under: {os.path.join(src_png_root, 'train')}"
        )

    train_path = os.path.join(set_dir, "train.txt")
    val_path = os.path.join(set_dir, "val.txt")
    cls_train_path = os.path.join(set_dir, f"{class_name}_train.txt")
    cls_val_path = os.path.join(set_dir, f"{class_name}_val.txt")

    with open(train_path, "w") as f:
        f.write("\n".join(train_basenames) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val_basenames) + "\n")
    with open(cls_train_path, "w") as f:
        f.writelines(f"{b} 1\n" for b in train_basenames)
    with open(cls_val_path, "w") as f:
        f.writelines(f"{b} 1\n" for b in val_basenames)

    return {
        "num_train_ids": len(train_basenames),
        "num_eval_ids": len(val_basenames),
        "copied": copied,
        "skipped_existing": skipped,
    }


def _write_runtime_config(
    base_config,
    output_dir,
    voc_root,
    clip_pretrain_path,
    dino_model=None,
    dino_fts_dim=None,
    dino_decoder_layers=None,
):
    os.makedirs(output_dir, exist_ok=True)
    name_list_dir = os.path.join(voc_root, "ImageSets", "Main")

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(base_config)
        cfg.dataset.root_dir = voc_root
        cfg.dataset.name_list_dir = name_list_dir
        cfg.clip_init.clip_pretrain_path = clip_pretrain_path
        if dino_model:
            cfg.dino_init.dino_model = dino_model
        if dino_fts_dim is not None:
            cfg.dino_init.dino_fts_fuse_dim = int(dino_fts_dim)
        if dino_decoder_layers is not None:
            cfg.dino_init.decoder_layer = int(dino_decoder_layers)

        output_path = os.path.join(output_dir, "voc_attn_reg_runtime.yaml")
        OmegaConf.save(cfg, output_path)
        return output_path
    except Exception:
        # Fallback to text replacement if OmegaConf is unavailable.
        with open(base_config, "r") as f:
            content = f.read()

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

        if dino_model:
            content = re.sub(
                r"(dino_model:\s*')([^']*)(')",
                rf"\1{dino_model}\3",
                content,
            )
        if dino_fts_dim is not None:
            content = re.sub(
                r"(dino_fts_fuse_dim:\s*)([0-9]+)",
                rf"\1{int(dino_fts_dim)}",
                content,
            )
        if dino_decoder_layers is not None:
            content = re.sub(
                r"(decoder_layer:\s*)([0-9]+)",
                rf"\1{int(dino_decoder_layers)}",
                content,
            )

        output_path = os.path.join(output_dir, "voc_attn_reg_runtime.yaml")
        with open(output_path, "w") as f:
            f.write(content)
        return output_path


def main(
    repo_root,
    voc_workspace_root,
    src_png_root,
    class_name,
    setup_data,
    results_dir,
    clip_backend,
    clip_model,
    clip_pretrained,
    dino_model,
    dino_fts_dim,
    dino_decoder_layers,
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
    os.environ["CLIP_TEXT_DATASET"] = "decoymnist"
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
        workspace_name="decoymnist",
    )
    _ensure_weclip_import_path(paths["weclip_root"])

    from scripts import dist_clip_voc
    import test_msc_flip_voc

    dist_clip_voc.args.work_dir = paths["train_work_dir"]
    print(f"Using VOC workspace: {paths['workspace_root']}")
    clip_pretrain_path = clip_model or paths["clip_pretrain_path"]

    train_txt = os.path.join(paths["set_dir"], "train.txt")
    if setup_data:
        stats = _prepare_decoymnist_dataset(
            src_png_root=src_png_root,
            class_name=class_name,
            set_dir=paths["set_dir"],
            dest_dir=paths["dest_dir"],
        )
        print(
            "Prepared DecoyMNIST dataset: "
            f"train IDs={stats['num_train_ids']}, "
            f"eval IDs={stats['num_eval_ids']}, "
            f"{stats['copied']} copied, "
            f"{stats['skipped_existing']} skipped existing."
        )
    else:
        val_txt = os.path.join(paths["set_dir"], "val.txt")
        if not os.path.isfile(train_txt) or not os.path.isfile(val_txt):
            raise FileNotFoundError(
                "ImageSets/Main train.txt/val.txt missing. "
                "Run once with --setup-data."
            )

    if not os.path.isfile(train_txt):
        raise FileNotFoundError(
            "ImageSets/Main/train.txt missing. Run once with --setup-data."
        )

    config = _write_runtime_config(
        paths["config"],
        paths["config_dir"],
        paths["voc_root"],
        clip_pretrain_path,
        dino_model=dino_model,
        dino_fts_dim=dino_fts_dim,
        dino_decoder_layers=dino_decoder_layers,
    )

    # Load and verify the runtime config in memory (guards against
    # NFS caching or OmegaConf serialisation quirks on HPC clusters).
    from omegaconf import OmegaConf

    runtime_cfg = OmegaConf.load(config)
    if dino_model:
        actual = runtime_cfg.dino_init.dino_model
        if actual != dino_model:
            print(
                f"WARNING: runtime config dino_model mismatch: "
                f"expected '{dino_model}', got '{actual}'. Patching in-memory."
            )
        runtime_cfg.dino_init.dino_model = dino_model
    if dino_fts_dim is not None:
        runtime_cfg.dino_init.dino_fts_fuse_dim = int(dino_fts_dim)
    if dino_decoder_layers is not None:
        runtime_cfg.dino_init.decoder_layer = int(dino_decoder_layers)

    schedule = compute_powerlaw_schedule(
        n_train=count_nonempty_lines(train_txt),
        b_eff=int(runtime_cfg.train.samples_per_gpu),
    )
    runtime_cfg.train.max_iters = int(schedule["max_iters"])
    runtime_cfg.train.cam_iters = int(schedule["cam_iters"])
    runtime_cfg.train.eval_iters = int(schedule["eval_iters"])
    OmegaConf.save(runtime_cfg, config)

    print(
        f"Runtime config: clip_backend={clip_backend_effective}, "
        f"clip_pretrained={clip_pretrained_effective}, "
        f"seed={seed if seed is not None else os.environ.get('WECLIP_RUN_SEED', '1')}, "
        f"dino_model={runtime_cfg.dino_init.dino_model}, "
        f"dino_fts_fuse_dim={runtime_cfg.dino_init.dino_fts_fuse_dim}, "
        f"decoder_layer={runtime_cfg.dino_init.decoder_layer}, "
        f"N_train={schedule['n_train']}, "
        f"steps/epoch={schedule['steps_per_epoch']}, "
        f"max_iters={schedule['max_iters']}, "
        f"cam_iters={schedule['cam_iters']}, "
        f"eval_iters={schedule['eval_iters']}, "
        f"coverage_guaranteed={schedule['coverage_guaranteed']}"
    )

    final_path = dist_clip_voc.main(config)

    if results_dir:
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(paths["weclip_root"], results_dir)
        test_msc_flip_voc.args.work_dir = results_dir

    # Pass the verified config object directly to test phase, bypassing
    # any file-system caching issues on NFS-backed HPC clusters.
    test_msc_flip_voc.outer_main(final_path, config, cfg_override=runtime_cfg)


if __name__ == "__main__":
    _t0 = time.perf_counter()
    _status = "completed"
    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo masks for DecoyMNIST from DecoyMNIST_png split folders. "
            "Use --setup-data once to populate WeCLIPPlus JPEGImages/ and ImageSets/."
        )
    )
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
            "Default: <repo-root>/pipelines/generate_r4rr_maps/voc_workspaces/decoymnist"
        ),
    )
    parser.add_argument(
        "--src-png-root",
        default=os.path.join(_default_repo_root(), "data", "DecoyMNIST_png"),
        help="Path to DecoyMNIST_png directory containing train/ and test/ digit folders.",
    )
    parser.add_argument(
        "--class-name",
        default="digit",
        help="Foreground class name for DecoyMNIST (default: digit).",
    )
    parser.add_argument(
        "--setup-data",
        dest="setup_data",
        action="store_true",
        help="Prepare VOC-style JPEGImages/ImageSets (train list=train split; val list=train split).",
    )
    parser.add_argument(
        "--no-setup-data",
        dest="setup_data",
        action="store_false",
        help="Skip data setup and reuse current VOC-style data.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory for prediction_cmap (default: results).",
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
        help=(
            "Override CLIP model identifier. For openai, this can be a checkpoint path. "
            "For openclip/siglip2, this can be an open_clip model name."
        ),
    )
    parser.add_argument(
        "--clip-pretrained",
        default=None,
        help=(
            "Override open_clip pretrained tag (e.g., openai, laion2b_s34b_b88k, webli). "
            "Only used by openclip/siglip2 backends."
        ),
    )
    parser.add_argument(
        "--dino-model",
        default=None,
        help="Override DINO model name in config (e.g., xcit_medium_24_p16).",
    )
    parser.add_argument(
        "--dino-fts-dim",
        type=int,
        default=None,
        help="Override dino_fts_fuse_dim in config (e.g., 512 for XCiT-Medium).",
    )
    parser.add_argument(
        "--dino-decoder-layers",
        type=int,
        default=None,
        help="Override decoder_layer in config.",
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
            args.src_png_root,
            args.class_name,
            args.setup_data,
            args.results_dir,
            args.clip_backend,
            args.clip_model,
            args.clip_pretrained,
            args.dino_model,
            args.dino_fts_dim,
            args.dino_decoder_layers,
            clip_backend_explicit=clip_backend_explicit,
            seed=args.seed,
        )
    except Exception:
        _status = "failed"
        raise
    finally:
        _elapsed = time.perf_counter() - _t0
        print(
            f"[generate_pseudo_masks_DecoyMNIST] {_status} in "
            f"{_format_elapsed(_elapsed)} ({_elapsed:.2f}s)"
        )
