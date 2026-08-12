#!/usr/bin/env python3
"""Generate manifest-keyed CLIP ViT transformer relevance maps for IN-9 GALS."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from imagenet9_data import CLASS_NAMES, ImageNet9Sample, load_original_samples


MODEL_NAME = "ViT-B/32"
MAP_METHOD = "clip_transformer_relevance"
MAP_SCHEMA_VERSION = 1
PIL_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
PROMPT_CONCEPTS: Mapping[str, str] = {
    "dog": "dog",
    "bird": "bird",
    "vehicle": "vehicle",
    "reptile": "reptile",
    "carnivore": "carnivore",
    "insect": "insect",
    "instrument": "musical instrument",
    "primate": "primate",
    "fish": "fish",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection", choices=("diagnostic", "all"), default="diagnostic")
    parser.add_argument("--diagnostic-per-class", type=int, default=20)
    parser.add_argument("--diagnostic-seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-checkpoint", default=MODEL_NAME)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--write-qa", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompts_for_class(class_name: str) -> List[str]:
    concept = PROMPT_CONCEPTS[class_name]
    article = "an" if concept[0].lower() in "aeiou" else "a"
    return [f"an image of {article} {concept}", f"a photo of {article} {concept}"]


def _contract(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "dataset": "imagenet9_backgrounds_challenge",
        "source_split": "reconstructed_original_train",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "model": MODEL_NAME,
        "model_checkpoint_argument": args.clip_checkpoint,
        "method": MAP_METHOD,
        "map_tensor_key": "unnormalized_attentions",
        "expected_map_shape": [2, 1, 7, 7],
        "prompt_concepts": dict(PROMPT_CONCEPTS),
        "prompts_by_class": {
            class_name: prompts_for_class(class_name) for class_name in CLASS_NAMES
        },
        "class_names": list(CLASS_NAMES),
        "background_specific_prompts": False,
        "official_variants_generated": False,
    }


def _ensure_contract(path: Path, contract: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            stored = json.loads(path.read_text())
            if stored != contract:
                raise RuntimeError(
                    f"Refusing to mix GALS maps with a changed contract: {path}"
                )
        else:
            temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            temporary.write_text(encoded)
            temporary.replace(path)


def _diagnostic_samples(
    samples: Sequence[ImageNet9Sample],
    per_class: int,
    seed: int,
) -> List[ImageNet9Sample]:
    if per_class <= 0:
        raise ValueError("--diagnostic-per-class must be positive")
    selected: List[ImageNet9Sample] = []
    for class_name in CLASS_NAMES:
        class_samples = [sample for sample in samples if sample.class_name == class_name]
        ranked = sorted(
            class_samples,
            key=lambda sample: hashlib.sha256(
                f"{seed}:{sample.sample_id}".encode()
            ).hexdigest(),
        )
        if len(ranked) < per_class:
            raise RuntimeError(
                f"Class {class_name} has {len(ranked)} samples, fewer than {per_class}"
            )
        selected.extend(ranked[:per_class])
    return sorted(selected, key=lambda sample: (sample.label, sample.sample_id))


def select_samples(args: argparse.Namespace) -> List[ImageNet9Sample]:
    samples = load_original_samples(args.manifest, "train", verify_files=True)
    if args.selection == "diagnostic":
        samples = _diagnostic_samples(
            samples, args.diagnostic_per_class, args.diagnostic_seed
        )
    else:
        samples = sorted(samples, key=lambda sample: (sample.label, sample.sample_id))
    start = max(args.start_index, 0)
    end = len(samples) if args.end_index < 0 else min(args.end_index, len(samples))
    if start >= end:
        raise ValueError(
            f"Empty map-generation range [{start}, {end}) for {len(samples)} selected samples"
        )
    return samples[start:end]


def _write_selection(samples: Sequence[ImageNet9Sample], path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "label", "class_name", "source_path"),
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "class_name": sample.class_name,
                    "source_path": str(sample.path),
                }
            )
    try:
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)


def map_path(output_root: Path, sample: ImageNet9Sample) -> Path:
    return output_root / "maps" / sample.class_name / f"{sample.sample_id}.pth"


def _validate_map_payload(
    payload: Mapping[str, object],
    sample: ImageNet9Sample,
) -> Dict[str, object]:
    attention = payload.get("unnormalized_attentions")
    if not torch.is_tensor(attention):
        raise RuntimeError(f"Map has no tensor 'unnormalized_attentions': {sample.sample_id}")
    if tuple(attention.shape) != (2, 1, 7, 7):
        raise RuntimeError(
            f"Unexpected map shape for {sample.sample_id}: {tuple(attention.shape)}"
        )
    finite = bool(torch.isfinite(attention).all().item())
    nonzero_prompts = int(
        torch.count_nonzero(attention.reshape(attention.shape[0], -1), dim=1)
        .gt(0)
        .sum()
        .item()
    )
    return {
        "finite": finite,
        "nonzero_prompts": nonzero_prompts,
        "minimum": float(attention.min().item()),
        "maximum": float(attention.max().item()),
        "mean": float(attention.mean().item()),
    }


def _heatmap(normalized_map: np.ndarray) -> np.ndarray:
    value = np.clip(normalized_map, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    return np.uint8(255 * np.stack((red, green, blue), axis=-1))


def _qa_triptych(
    sample: ImageNet9Sample,
    payload: Mapping[str, object],
    output_path: Path,
) -> None:
    attention = payload["unnormalized_attentions"]
    combined = attention.float().mean(dim=0, keepdim=True)
    combined = F.interpolate(combined, size=(224, 224), mode="bilinear", align_corners=False)
    combined = combined[0, 0]
    minimum = combined.min()
    maximum = combined.max()
    normalized = (combined - minimum) / torch.clamp(maximum - minimum, min=1e-12)
    heat = _heatmap(normalized.numpy())
    with Image.open(sample.path) as image_file:
        image = image_file.convert("RGB").resize((224, 224), PIL_BICUBIC)
    image_array = np.asarray(image, dtype=np.float32)
    overlay = np.uint8(np.clip(0.55 * image_array + 0.45 * heat, 0, 255))
    header = 28
    canvas = Image.new("RGB", (224 * 3, 224 + header), "white")
    canvas.paste(image, (0, header))
    canvas.paste(Image.fromarray(heat), (224, header))
    canvas.paste(Image.fromarray(overlay), (448, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((4, 7), f"{sample.class_name}: {sample.sample_id}", fill="black", font=font)
    draw.text((228, 7), "CLIP ViT map", fill="black", font=font)
    draw.text((452, 7), "overlay", fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=88, optimize=True)


def _class_qa_sheets(output_root: Path, samples: Sequence[ImageNet9Sample]) -> None:
    for class_name in CLASS_NAMES:
        triptychs = [
            output_root / "qa" / "samples" / class_name / f"{sample.sample_id}.jpg"
            for sample in samples
            if sample.class_name == class_name
        ]
        triptychs = [path for path in triptychs if path.is_file()]
        if not triptychs:
            continue
        images = [Image.open(path).convert("RGB") for path in triptychs]
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        sheet = Image.new("RGB", (width, height), "white")
        offset = 0
        for image in images:
            sheet.paste(image, (0, offset))
            offset += image.height
            image.close()
        sheet_path = output_root / "qa" / f"{class_name}_{len(triptychs)}_samples.jpg"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_path, quality=88, optimize=True)


def _write_rows(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "sample_id", "label", "class_name", "source_path", "map_path", "prompts",
        "finite", "nonzero_prompts", "minimum", "maximum", "mean", "reused",
        "seconds",
    )
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract = _contract(args)
    _ensure_contract(args.output_root / "map_contract.json", contract)

    all_selected = (
        _diagnostic_samples(
            load_original_samples(args.manifest, "train", verify_files=True),
            args.diagnostic_per_class,
            args.diagnostic_seed,
        )
        if args.selection == "diagnostic"
        else sorted(
            load_original_samples(args.manifest, "train", verify_files=True),
            key=lambda sample: (sample.label, sample.sample_id),
        )
    )
    if args.selection == "diagnostic":
        _write_selection(
            all_selected,
            args.output_root / "diagnostic_selection_seed0.csv",
        )
    samples = select_samples(args)

    from CLIP.clip import clip
    from utils.attention_utils import transformer_attention

    print(
        f"[MAPS] selection={args.selection} shard={args.start_index}:{args.end_index} "
        f"samples={len(samples)} output={args.output_root}",
        flush=True,
    )
    print(f"[MAPS] model={args.clip_checkpoint} method={MAP_METHOD}", flush=True)
    model, preprocess = clip.load(args.clip_checkpoint, device=args.device, jit=False)
    model.eval()
    preprocess_no_crop = []
    for transform in preprocess.transforms:
        if isinstance(transform, transforms.Resize):
            preprocess_no_crop.append(
                transforms.Resize((224, 224), interpolation=PIL_BICUBIC)
            )
        elif not isinstance(transform, transforms.CenterCrop):
            preprocess_no_crop.append(transform)
    preprocess = transforms.Compose(preprocess_no_crop)
    tokens = {
        class_name: clip.tokenize(prompts_for_class(class_name)).to(args.device)
        for class_name in CLASS_NAMES
    }

    rows: List[Dict[str, object]] = []
    for index, sample in enumerate(samples, start=1):
        destination = map_path(args.output_root, sample)
        prompts = prompts_for_class(sample.class_name)
        started = time.time()
        reused = False
        if args.skip_existing and destination.is_file():
            payload = torch.load(destination, map_location="cpu")
            reused = True
        else:
            payload = transformer_attention(
                model,
                preprocess,
                str(sample.path),
                text_list=prompts,
                tokenized_text=tokens[sample.class_name],
                device=args.device,
                plot_vis=False,
                resize=False,
            )
            payload.update(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "class_name": sample.class_name,
                    "source_path": str(sample.path),
                    "model_name": MODEL_NAME,
                    "map_method": MAP_METHOD,
                    "map_schema_version": MAP_SCHEMA_VERSION,
                }
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)
        stats = _validate_map_payload(payload, sample)
        if not stats["finite"]:
            raise RuntimeError(f"Non-finite GALS map: {destination}")
        if args.write_qa:
            _qa_triptych(
                sample,
                payload,
                args.output_root / "qa" / "samples" / sample.class_name / f"{sample.sample_id}.jpg",
            )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "class_name": sample.class_name,
                "source_path": str(sample.path),
                "map_path": str(destination.resolve()),
                "prompts": json.dumps(prompts),
                **stats,
                "reused": reused,
                "seconds": time.time() - started,
            }
        )
        print(
            f"[MAP] {index}/{len(samples)} class={sample.class_name} "
            f"sample={sample.sample_id} nonzero={stats['nonzero_prompts']}/2 reused={reused}",
            flush=True,
        )
        del payload
        if args.device.startswith("cuda") and index % 25 == 0:
            torch.cuda.empty_cache()

    shard_end = args.start_index + len(samples)
    manifest_path = (
        args.output_root
        / "manifests"
        / f"{args.selection}_{args.start_index:06d}_{shard_end:06d}.csv"
    )
    _write_rows(rows, manifest_path)
    if args.selection == "diagnostic" and args.write_qa:
        _class_qa_sheets(args.output_root, all_selected)
    valid = sum(bool(row["finite"]) and int(row["nonzero_prompts"]) > 0 for row in rows)
    summary = {
        "selection": args.selection,
        "start_index": args.start_index,
        "end_index": shard_end,
        "samples": len(rows),
        "finite_nonzero_maps": valid,
        "zero_maps": len(rows) - valid,
        "manifest": str(manifest_path.resolve()),
    }
    summary_path = manifest_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[DONE] {json.dumps(summary, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
