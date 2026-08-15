#!/usr/bin/env python3
"""Export sharded WeCLIP+ teacher maps for ImageNet-9 Original training images."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


EXPECTED_IMAGES = 45405
SCHEMA_VERSION = 1
CLASS_NAMES: Tuple[str, ...] = (
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
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-scales", type=float, nargs="+", default=(1.0, 1.5))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def shard_bounds(task_index: int, chunk_size: int, total: int) -> Tuple[int, int]:
    if task_index < 0 or chunk_size <= 0 or total <= 0:
        raise ValueError("task_index, chunk_size, and total must define a positive shard")
    start = task_index * chunk_size
    return start, min(start + chunk_size, total)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_workspace_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} workspace rows, found {len(rows)} in {path}"
        )
    return rows


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _ensure_contract(path: Path, contract: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            stored = json.loads(path.read_text())
            if stored != contract:
                raise RuntimeError(f"Refusing to mix maps with a changed contract: {path}")
        else:
            _atomic_json(path, contract)


def _valid_existing_map(map_path: Path, source_path: Path) -> bool:
    if not map_path.is_file():
        return False
    try:
        with Image.open(source_path) as source:
            expected_size = source.size
        with Image.open(map_path) as prediction:
            prediction.verify()
        with Image.open(map_path) as prediction:
            return prediction.mode == "RGB" and prediction.size == expected_size
    except Exception:
        return False


def _voc_colormap(count: int = 256) -> np.ndarray:
    colors = np.zeros((count, 3), dtype=np.uint8)
    for index in range(count):
        red = green = blue = 0
        value = index
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        colors[index] = (red, green, blue)
    return colors


def _writable_rgb_array(image: Image.Image) -> np.ndarray:
    """Return the writable C-order RGB buffer required by pydensecrf."""
    return np.array(image.convert("RGB"), dtype=np.uint8, copy=True, order="C")


def _save_prediction(
    destination: Path,
    source_path: Path,
    logits,
    post_processor,
) -> Dict[str, object]:
    import torch.nn.functional as F

    with Image.open(source_path) as image_file:
        image = _writable_rgb_array(image_file)
    height, width = image.shape[:2]
    resized = F.interpolate(
        logits.detach().cpu(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    probabilities = np.array(
        F.softmax(resized, dim=1)[0].numpy(),
        dtype=np.float32,
        copy=True,
        order="C",
    )
    probabilities = post_processor(image, probabilities)
    labels = np.argmax(probabilities, axis=0).astype(np.uint8)
    encoded = _voc_colormap()[labels]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.png")
    Image.fromarray(encoded, mode="RGB").save(temporary, format="PNG")
    temporary.replace(destination)
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = int(labels.size)
    return {
        "height": height,
        "width": width,
        "unique_labels": [int(value) for value in unique_labels],
        "background_fraction": float(counts[unique_labels == 0].sum() / total),
    }


def _inference_contract(
    args: argparse.Namespace,
    result: Mapping[str, object],
    runtime_config: Path,
    checkpoint: Path,
    workspace_manifest: Path,
    workspace_contract: Path,
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "imagenet9_backgrounds_challenge",
        "source_split": "reconstructed_original_train",
        "num_source_images": EXPECTED_IMAGES,
        "class_names": list(CLASS_NAMES),
        "training_result": str(args.training_result.resolve()),
        "training_result_sha256": _sha256(args.training_result),
        "training_status": result["status"],
        "training_iterations": result["schedule"]["configured_max_iters"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "runtime_config": str(runtime_config.resolve()),
        "runtime_config_sha256": _sha256(runtime_config),
        "workspace_manifest": str(workspace_manifest.resolve()),
        "workspace_manifest_sha256": _sha256(workspace_manifest),
        "workspace_contract_sha256": _sha256(workspace_contract),
        "model": "WeCLIP+",
        "clip_backend": "openclip_adapter",
        "clip_model": "ViT-B/16",
        "clip_pretrained": "openai",
        "dino_model": "dinov2_vitl14_reg",
        "prediction": "mean of scale 1.0 and 1.5, each flip-averaged, then DenseCRF",
        "test_scales": [float(value) for value in args.test_scales],
        "output_format": "RGB PNG using VOC colormap indices 0..9",
        "official_validation_used": False,
        "official_test_variants_used": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if 1.0 not in args.test_scales:
        raise ValueError("--test-scales must include 1.0 to preserve the WeCLIP protocol")

    result = json.loads(args.training_result.read_text())
    if result.get("status") != "complete" or result.get("mode") != "full":
        raise RuntimeError(f"Training result is not a completed full run: {args.training_result}")
    checkpoint = Path(str(result["checkpoint"]))
    runtime_config = Path(str(result["runtime_config"]))
    if not checkpoint.is_file() or not runtime_config.is_file():
        raise FileNotFoundError(f"Missing checkpoint/config: {checkpoint}, {runtime_config}")

    voc_root = args.workspace_root / "VOCdevkit" / "VOC2012"
    set_dir = voc_root / "ImageSets" / "Main"
    workspace_manifest = args.workspace_root / "metadata" / "workspace_manifest.csv"
    workspace_contract = args.workspace_root / "metadata" / "workspace_contract.json"
    rows = _read_workspace_manifest(workspace_manifest)
    ids = _read_ids(set_dir / "val.txt")
    if len(ids) != EXPECTED_IMAGES or ids != [row["sample_id"] for row in rows]:
        raise RuntimeError("val.txt does not exactly match the ordered workspace manifest")
    if len(set(ids)) != EXPECTED_IMAGES:
        raise RuntimeError("Workspace contains duplicate sample IDs")

    start = max(args.start_index, 0)
    end = min(args.end_index, EXPECTED_IMAGES)
    if start >= end:
        raise ValueError(f"Empty inference range [{start}, {end})")
    args.output_root.mkdir(parents=True, exist_ok=True)
    map_root = args.output_root / "prediction_cmap"
    contract = _inference_contract(
        args,
        result,
        runtime_config,
        checkpoint,
        workspace_manifest,
        workspace_contract,
    )
    _ensure_contract(args.output_root / "inference_contract.json", contract)

    selected_indices = list(range(start, end))
    pending_indices = []
    reused = 0
    for index in selected_indices:
        row = rows[index]
        source = Path(row["source_path"])
        destination = map_root / f"{row['sample_id']}.png"
        if not args.force and _valid_existing_map(destination, source):
            reused += 1
        else:
            pending_indices.append(index)

    print(
        f"[SHARD] range=[{start},{end}) total={len(selected_indices)} "
        f"pending={len(pending_indices)} reused={reused}",
        flush=True,
    )
    shard_path = args.output_root / "shards" / f"{start:06d}_{end:06d}.json"
    if not pending_indices:
        _atomic_json(
            shard_path,
            {
                "status": "complete",
                "start_index": start,
                "end_index": end,
                "samples": len(selected_indices),
                "generated": 0,
                "reused": reused,
            },
        )
        print(f"[DONE] shard already complete: {shard_path}", flush=True)
        return 0

    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    from datasets import voc
    from utils.dcrf import DenseCRF
    from WeCLIP_Plus.model_attn_aff_voc import WeCLIP_Plus

    cfg = OmegaConf.load(runtime_config)
    cfg.dataset.root_dir = str(voc_root)
    cfg.dataset.name_list_dir = str(set_dir)
    dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split="val",
        stage="val",
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    subset = Subset(dataset, pending_indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=max(args.num_workers, 0),
        pin_memory=False,
    )
    clip_pretrained = cfg.clip_init.get("clip_pretrained", None)
    model = WeCLIP_Plus(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        clip_pretrained=clip_pretrained,
        dino_model=cfg.dino_init.dino_model,
        dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
        decoder_layers=cfg.dino_init.decoder_layer,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        clip_flag=cfg.clip_init.clip_flag,
        device="cuda",
    )
    incompatible = model.load_state_dict(
        torch.load(checkpoint, map_location="cpu"), strict=False
    )
    print(
        f"[MODEL] checkpoint={checkpoint} missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}",
        flush=True,
    )
    model.to(args.device)
    model.eval()
    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=3,
        pos_w=3,
        bi_xy_std=64,
        bi_rgb_std=5,
        bi_w=4,
    )

    generated_rows: List[Dict[str, object]] = []
    for position, batch in enumerate(loader, start=1):
        sample_started = time.time()
        names, inputs, _labels, _class_labels = batch
        sample_id = names[0]
        row = rows[pending_indices[position - 1]]
        if sample_id != row["sample_id"]:
            raise RuntimeError(f"Dataset order mismatch: {sample_id} != {row['sample_id']}")
        inputs = inputs.to(args.device)
        _, _, height, width = inputs.shape
        ratio = float(cfg.clip_init.resize_long) / max(height, width)
        resized_inputs = F.interpolate(
            inputs,
            size=(int(height * ratio), int(width * ratio)),
            mode="bilinear",
            align_corners=False,
        )
        base_pair = torch.cat([resized_inputs, resized_inputs.flip(-1)], dim=0)
        model.zero_grad(set_to_none=True)
        clip_logits, dino_logits, _cam, _attention = model(
            base_pair, tuple(names) + tuple(names), mode="val"
        )
        combined = (0.5 * dino_logits + 0.5 * clip_logits).detach()
        target_size = combined.shape[-2:]
        scale_logits = [
            (combined[0] + combined[1].flip(-1)) / 2.0
        ]
        del clip_logits, dino_logits, combined, _cam, _attention
        torch.cuda.empty_cache()

        for scale in args.test_scales:
            if float(scale) == 1.0:
                continue
            scaled = F.interpolate(
                resized_inputs,
                scale_factor=float(scale),
                mode="bilinear",
                align_corners=False,
            )
            scaled_pair = torch.cat([scaled, scaled.flip(-1)], dim=0)
            model.zero_grad(set_to_none=True)
            clip_logits, dino_logits, _cam, _attention = model(
                scaled_pair, tuple(names) + tuple(names), mode="val"
            )
            combined = (0.5 * dino_logits + 0.5 * clip_logits).detach()
            combined = F.interpolate(
                combined, size=target_size, mode="bilinear", align_corners=False
            )
            scale_logits.append((combined[0] + combined[1].flip(-1)) / 2.0)
            del clip_logits, dino_logits, combined, _cam, _attention
            torch.cuda.empty_cache()

        mean_logits = torch.stack(scale_logits, dim=0).mean(dim=0).unsqueeze(0)
        destination = map_root / f"{sample_id}.png"
        stats = _save_prediction(
            destination,
            Path(row["source_path"]),
            mean_logits,
            post_processor,
        )
        expected_label = int(row["label"]) + 1
        generated_rows.append(
            {
                "sample_id": sample_id,
                "class_name": row["class_name"],
                "expected_segmentation_label": expected_label,
                "map_path": str(destination),
                "seconds": time.time() - sample_started,
                **stats,
            }
        )
        del mean_logits, scale_logits, resized_inputs, inputs
        torch.cuda.empty_cache()
        if position == 1 or position % 25 == 0 or position == len(pending_indices):
            print(
                f"[MAP] {position}/{len(pending_indices)} sample={sample_id} "
                f"labels={stats['unique_labels']} bg={stats['background_fraction']:.4f}",
                flush=True,
            )

    _atomic_json(
        shard_path,
        {
            "status": "complete",
            "start_index": start,
            "end_index": end,
            "samples": len(selected_indices),
            "generated": len(generated_rows),
            "reused": reused,
            "generated_rows": generated_rows,
        },
    )
    print(f"[DONE] shard={shard_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
