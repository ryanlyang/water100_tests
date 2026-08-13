#!/usr/bin/env python3
"""Train the ImageNet-9 WeCLIP+ teacher used to generate R4RR maps."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence


CLASS_NAMES = (
    "dog",
    "bird",
    "vehicle",
    "reptile",
    "carnivore",
    "insect",
    "instrument",
    "primate",
    "fish",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--weclip-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke-iters", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--dino-model", default="dinov2_vitl14_reg")
    parser.add_argument("--dino-feature-dim", type=int, default=1024)
    parser.add_argument("--dino-decoder-layers", type=int, default=5)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def load_prompt_spec(path: Path):
    spec = importlib.util.spec_from_file_location("clip_text_imagenet9_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_contract(path: Path, contract: Dict[str, object]) -> None:
    encoded = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != contract:
            raise RuntimeError(
                f"Refusing to resume WeCLIP training with a changed contract: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def validate_workspace(args: argparse.Namespace):
    workspace_root = args.workspace_root.resolve(strict=True)
    voc_root = workspace_root / "VOCdevkit" / "VOC2012"
    set_dir = voc_root / "ImageSets" / "Main"
    image_dir = voc_root / "JPEGImages"
    annotation_dir = voc_root / "Annotations"
    workspace_contract_path = workspace_root / "metadata" / "workspace_contract.json"
    workspace_audit_path = workspace_root / "metadata" / "workspace_audit.json"
    prompt_path = (
        args.weclip_root
        / "clip"
        / "clip_texts"
        / "clip_text_imagenet9.py"
    ).resolve(strict=True)

    for path in (
        voc_root,
        set_dir / "train.txt",
        set_dir / "val.txt",
        set_dir / "classes.txt",
        workspace_contract_path,
        workspace_audit_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    audit = json.loads(workspace_audit_path.read_text())
    if audit.get("status") != "ok" or int(audit.get("num_images", -1)) != 45405:
        raise RuntimeError(f"Workspace audit is not valid: {workspace_audit_path}")
    if audit.get("held_out_validation_included") or audit.get("official_variants_included"):
        raise RuntimeError("R4RR teacher workspace includes forbidden evaluation images")

    classes = tuple(
        line.strip() for line in (set_dir / "classes.txt").read_text().splitlines()
        if line.strip()
    )
    if classes != CLASS_NAMES:
        raise RuntimeError(f"Unexpected ImageNet-9 class order: {classes}")
    if count_lines(set_dir / "train.txt") != 45405:
        raise RuntimeError("ImageNet-9 WeCLIP train.txt must contain exactly 45,405 IDs")
    if count_lines(set_dir / "val.txt") != 45405:
        raise RuntimeError("ImageNet-9 WeCLIP val.txt must alias exactly 45,405 training IDs")
    if sum(1 for path in image_dir.iterdir() if path.is_symlink()) != 45405:
        raise RuntimeError("ImageNet-9 WeCLIP JPEGImages symlink count is not 45,405")
    if sum(1 for _ in annotation_dir.glob("*.xml")) != 45405:
        raise RuntimeError(
            "Image-level class XML annotations are missing. Re-run the updated VOC workspace builder."
        )

    prompt_spec = load_prompt_spec(prompt_path)
    if tuple(prompt_spec._all_class_names) != CLASS_NAMES:
        raise RuntimeError("Prompt class order does not match the workspace")
    if len(prompt_spec._all_new_class_names) != len(CLASS_NAMES):
        raise RuntimeError("Prompt foreground concept count is not nine")
    if not prompt_spec.BACKGROUND_CATEGORY:
        raise RuntimeError("ImageNet-9 background prompt list is empty")

    return voc_root, set_dir, prompt_path, workspace_contract_path, workspace_audit_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("Batch size and checkpoint interval must be positive")
    if args.mode == "smoke" and args.smoke_iters <= 0:
        raise ValueError("--smoke-iters must be positive")

    weclip_root = args.weclip_root.resolve(strict=True)
    base_config = (weclip_root / "configs" / "voc_attn_reg.yaml").resolve(strict=True)
    voc_root, set_dir, prompt_path, workspace_contract_path, workspace_audit_path = validate_workspace(args)

    pipeline_root = weclip_root.parent / "pipelines" / "generate_r4rr_maps"
    if str(pipeline_root) not in sys.path:
        sys.path.insert(0, str(pipeline_root))
    from iter_schedule import compute_powerlaw_schedule

    full_schedule = compute_powerlaw_schedule(45405, args.batch_size)
    max_iters = args.smoke_iters if args.mode == "smoke" else int(full_schedule["max_iters"])
    schedule = dict(full_schedule)
    schedule["configured_max_iters"] = max_iters

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    runtime_config = run_root / "weclip_imagenet9_runtime.yaml"
    state_path = run_root / "latest_training_state.pt"
    result_path = run_root / "training_result.json"
    contract_path = run_root / "training_contract.json"

    contract = {
        "schema_version": 1,
        "dataset": "imagenet9_backgrounds_challenge",
        "source_split": "reconstructed_original_train",
        "num_source_images": 45405,
        "foreground_class_order": list(CLASS_NAMES),
        "segmentation_num_classes_including_background": 10,
        "workspace_contract_sha256": sha256_file(workspace_contract_path),
        "workspace_audit_sha256": sha256_file(workspace_audit_path),
        "prompt_config_sha256": sha256_file(prompt_path),
        "mode": args.mode,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_iters": max_iters,
        "checkpoint_interval": args.checkpoint_interval,
        "clip_backend": "openclip_adapter",
        "clip_model": "ViT-B/16",
        "clip_pretrained": "openai",
        "dino_model": args.dino_model,
        "dino_feature_dim": args.dino_feature_dim,
        "dino_decoder_layers": args.dino_decoder_layers,
        "held_out_validation_used": False,
        "official_variants_used": False,
    }
    write_contract(contract_path, contract)

    if result_path.is_file():
        result = json.loads(result_path.read_text())
        checkpoint = Path(result.get("checkpoint", ""))
        if result.get("status") == "complete" and checkpoint.is_file():
            print(f"[SKIP] WeCLIP training is already complete: {checkpoint}")
            return 0

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(base_config)
    cfg.dataset.root_dir = str(voc_root)
    cfg.dataset.name_list_dir = str(set_dir)
    cfg.dataset.num_classes = 10
    cfg.train.samples_per_gpu = args.batch_size
    cfg.train.max_iters = max_iters
    cfg.train.cam_iters = min(int(full_schedule["cam_iters"]), max_iters)
    cfg.train.eval_iters = min(int(full_schedule["eval_iters"]), max_iters)
    cfg.clip_init.clip_pretrain_path = "ViT-B/16"
    cfg.clip_init.clip_pretrained = "openai"
    cfg.dino_init.dino_model = args.dino_model
    cfg.dino_init.dino_fts_fuse_dim = args.dino_feature_dim
    cfg.dino_init.decoder_layer = args.dino_decoder_layers
    cfg.work_dir.dir = str(run_root / "weclip_work")
    OmegaConf.save(cfg, runtime_config)

    os.environ["CLIP_TEXT_DATASET"] = "imagenet9"
    os.environ.pop("CLIP_TEXT_VERSION", None)
    os.environ["CLIP_BACKEND"] = "openclip"
    os.environ["CLIP_MODEL_NAME"] = "ViT-B-16"
    os.environ["CLIP_PRETRAINED"] = "openai"
    os.environ["WECLIP_RUN_SEED"] = str(args.seed)
    os.environ["WECLIP_NUM_WORKERS"] = str(args.num_workers)
    os.environ["WECLIP_STABLE_RUN_ID"] = "imagenet9_smoke" if args.mode == "smoke" else "imagenet9_full"
    os.environ["WECLIP_STATE_PATH"] = str(state_path)
    os.environ["WECLIP_CHECKPOINT_INTERVAL"] = str(args.checkpoint_interval)
    os.environ["WECLIP_KEEP_ITER_CHECKPOINTS"] = "0"
    if state_path.is_file():
        os.environ["WECLIP_RESUME_STATE"] = str(state_path)
        print(f"[RESUME] state={state_path}")
    else:
        os.environ.pop("WECLIP_RESUME_STATE", None)

    if str(weclip_root) not in sys.path:
        sys.path.insert(0, str(weclip_root))
    os.chdir(weclip_root)
    from scripts import dist_clip_voc

    dist_clip_voc.args.work_dir = str(run_root / "weclip_work")
    print(f"[INFO] mode={args.mode} seed={args.seed}")
    print(f"[INFO] runtime_config={runtime_config}")
    print(f"[INFO] schedule={json.dumps(schedule, sort_keys=True)}")
    print("[INFO] teacher=OpenCLIP ViT-B/16 (OpenAI weights) + DINOv2 ViT-L/14-register")
    checkpoint = dist_clip_voc.main(str(runtime_config))
    if not checkpoint or not Path(checkpoint).is_file():
        raise RuntimeError(f"WeCLIP training did not produce a final checkpoint: {checkpoint}")

    result = {
        "status": "complete",
        "mode": args.mode,
        "checkpoint": str(Path(checkpoint).resolve()),
        "runtime_config": str(runtime_config),
        "training_state": str(state_path),
        "schedule": schedule,
    }
    temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)
    print(f"[DONE] checkpoint={result['checkpoint']}")
    print(f"[DONE] result={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
