#!/usr/bin/env python3
"""Evaluate OpenAI CLIP ViT-B/16 and ViT-B/32 on official ImageNet-9 tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MODEL_NAMES: Tuple[str, ...] = ("ViT-B/16", "ViT-B/32")
OPENCLIP_MODEL_NAMES: Mapping[str, str] = {
    "ViT-B/16": "ViT-B-16",
    "ViT-B/32": "ViT-B-32",
}
PROMPT_TEMPLATES: Tuple[str, ...] = (
    "an image of {article} {concept}",
    "a photo of {article} {concept}",
)
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
SCHEMA_VERSION = 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ryreu/guided_cnn/data/imagenet9"),
    )
    parser.add_argument("--protocol-name", default="reconstructed_original_bbox1_v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def prompts_for_class(class_name: str) -> List[str]:
    concept = PROMPT_CONCEPTS[class_name]
    article = "an" if concept[0].lower() in "aeiou" else "a"
    return [
        template.format(article=article, concept=concept)
        for template in PROMPT_TEMPLATES
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _load_open_clip():
    try:
        import open_clip  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "This evaluator requires open_clip. Run it in the r4rr-weclip environment."
        ) from error
    return open_clip


def _create_openclip_model(open_clip, model_name: str, device: str, download_root: Optional[Path]):
    openclip_name = OPENCLIP_MODEL_NAMES[model_name]
    kwargs = {
        "model_name": openclip_name,
        "pretrained": "openai",
        "device": device,
    }
    if download_root is not None:
        download_root.mkdir(parents=True, exist_ok=True)
        parameters = inspect.signature(open_clip.create_model_and_transforms).parameters
        if "cache_dir" in parameters:
            kwargs["cache_dir"] = str(download_root)
        else:
            print(
                "[INFO] open_clip does not expose cache_dir; using its default cache",
                flush=True,
            )
    model, _train_preprocess, eval_preprocess = open_clip.create_model_and_transforms(
        **kwargs
    )
    tokenizer = open_clip.get_tokenizer(openclip_name)
    return model, eval_preprocess, tokenizer


def _slug(model_name: str) -> str:
    return model_name.lower().replace("/", "_").replace("-", "_")


def build_contract(args: argparse.Namespace) -> Dict[str, object]:
    from imagenet9_data import CLASS_NAMES, FORBIDDEN_SELECTION_VARIANTS

    metadata_root = args.data_root / "metadata" / args.protocol_name
    official_manifest = metadata_root / "official_test_manifest.csv"
    if not official_manifest.is_file():
        raise FileNotFoundError(official_manifest)
    concepts = dict(PROMPT_CONCEPTS)
    if tuple(concepts) != tuple(CLASS_NAMES):
        raise RuntimeError("Prompt concepts do not match the fixed ImageNet-9 class order")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "imagenet9_backgrounds_challenge",
        "evaluation_split": "official_test_only",
        "official_manifest": str(official_manifest.resolve()),
        "official_manifest_sha256": _sha256(official_manifest),
        "official_test_root": str(
            (args.data_root / "official_test" / "bg_challenge").resolve()
        ),
        "models": list(args.models),
        "weights": "openai",
        "implementation": "open_clip",
        "class_names": list(CLASS_NAMES),
        "prompt_concepts": concepts,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "prompts_by_class": {
            class_name: prompts_for_class(class_name) for class_name in CLASS_NAMES
        },
        "text_ensemble": "normalize_each_prompt_then_mean_then_normalize",
        "variants": list(FORBIDDEN_SELECTION_VARIANTS),
        "validation_or_tuning_data_used": False,
        "prompt_selection_on_official_test": False,
        "seed": args.seed,
    }


def _build_text_classifier(model, tokenizer, class_names: Sequence[str], device: str):
    import torch
    import torch.nn.functional as F

    class_features = []
    with torch.no_grad():
        for class_name in class_names:
            tokens = tokenizer(prompts_for_class(class_name)).to(device)
            prompt_features = model.encode_text(tokens).float()
            prompt_features = F.normalize(prompt_features, dim=1)
            class_feature = F.normalize(prompt_features.mean(dim=0), dim=0)
            class_features.append(class_feature)
    return torch.stack(class_features, dim=0)


def _evaluate_variant(
    model,
    preprocess,
    text_classifier,
    samples,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Dict[str, object]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from imagenet9_data import classification_metrics

    class ClipImageDataset(Dataset):
        def __init__(self, source_samples, transform):
            self.samples = list(source_samples)
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            sample = self.samples[index]
            with Image.open(sample.path) as image_file:
                image = self.transform(image_file.convert("RGB"))
            return image, sample.label

    loader = DataLoader(
        ClipImageDataset(samples, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    prediction_batches = []
    target_batches = []
    started = time.time()
    with torch.no_grad():
        for images, targets in loader:
            image_features = model.encode_image(
                images.to(device, non_blocking=True)
            ).float()
            image_features = F.normalize(image_features, dim=1)
            predictions = (image_features @ text_classifier.t()).argmax(dim=1)
            prediction_batches.append(predictions.cpu())
            target_batches.append(targets.cpu())
    metrics = classification_metrics(
        torch.cat(prediction_batches), torch.cat(target_batches)
    )
    metrics["samples"] = len(samples)
    metrics["seconds"] = time.time() - started
    return metrics


def summarize_model(variant_results: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    macro = {
        variant: float(metrics["macro_class_accuracy"])
        for variant, metrics in variant_results.items()
    }
    mixed_variants = ("mixed_same", "mixed_rand", "mixed_next")
    worst_variant = min(macro, key=macro.get)
    return {
        "original_macro_class_accuracy": macro["original"],
        "mixed_average_macro_class_accuracy": float(
            np.mean([macro[variant] for variant in mixed_variants])
        ),
        "original_minus_mixed_rand": macro["original"] - macro["mixed_rand"],
        "only_fg_macro_class_accuracy": macro["only_fg"],
        "only_bg_average_macro_class_accuracy": float(
            np.mean([macro["only_bg_b"], macro["only_bg_t"]])
        ),
        "no_fg_macro_class_accuracy": macro["no_fg"],
        "worst_variant": worst_variant,
        "worst_variant_macro_class_accuracy": macro[worst_variant],
    }


def _write_tables(
    output_root: Path,
    class_names: Sequence[str],
    model_results: Mapping[str, Mapping[str, object]],
) -> None:
    variant_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for model_name, payload in model_results.items():
        variants = payload["variant_results"]
        for variant, metrics in variants.items():
            variant_rows.append(
                {
                    "model": model_name,
                    "weights": "openai",
                    "variant": variant,
                    "samples": metrics["samples"],
                    "accuracy_pct": 100.0 * float(metrics["accuracy"]),
                    "macro_class_accuracy_pct": 100.0
                    * float(metrics["macro_class_accuracy"]),
                    "seconds": metrics["seconds"],
                }
            )
            for index, class_name in enumerate(class_names):
                class_rows.append(
                    {
                        "model": model_name,
                        "weights": "openai",
                        "variant": variant,
                        "class_index": index,
                        "class_name": class_name,
                        "support": metrics["class_support"][index],
                        "accuracy_pct": 100.0
                        * float(metrics["per_class_accuracy"][index]),
                    }
                )
        summary = payload["robustness_summary"]
        summary_rows.append(
            {
                "model": model_name,
                "weights": "openai",
                "original_macro_pct": 100.0
                * float(summary["original_macro_class_accuracy"]),
                "mixed_average_macro_pct": 100.0
                * float(summary["mixed_average_macro_class_accuracy"]),
                "original_minus_mixed_rand_points": 100.0
                * float(summary["original_minus_mixed_rand"]),
                "only_fg_macro_pct": 100.0
                * float(summary["only_fg_macro_class_accuracy"]),
                "only_bg_average_macro_pct": 100.0
                * float(summary["only_bg_average_macro_class_accuracy"]),
                "no_fg_macro_pct": 100.0
                * float(summary["no_fg_macro_class_accuracy"]),
                "worst_variant": summary["worst_variant"],
                "worst_variant_macro_pct": 100.0
                * float(summary["worst_variant_macro_class_accuracy"]),
            }
        )
    _atomic_csv(
        output_root / "variant_results.csv",
        (
            "model",
            "weights",
            "variant",
            "samples",
            "accuracy_pct",
            "macro_class_accuracy_pct",
            "seconds",
        ),
        variant_rows,
    )
    _atomic_csv(
        output_root / "per_class_results.csv",
        (
            "model",
            "weights",
            "variant",
            "class_index",
            "class_name",
            "support",
            "accuracy_pct",
        ),
        class_rows,
    )
    _atomic_csv(
        output_root / "robustness_summary.csv",
        (
            "model",
            "weights",
            "original_macro_pct",
            "mixed_average_macro_pct",
            "original_minus_mixed_rand_points",
            "only_fg_macro_pct",
            "only_bg_average_macro_pct",
            "no_fg_macro_pct",
            "worst_variant",
            "worst_variant_macro_pct",
        ),
        summary_rows,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    invalid_models = sorted(set(args.models) - set(MODEL_NAMES))
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}; expected {MODEL_NAMES}")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    _seed_everything(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract = build_contract(args)
    _atomic_json(args.output_root / "evaluation_contract.json", contract)

    from imagenet9_data import (
        CLASS_NAMES,
        FORBIDDEN_SELECTION_VARIANTS,
        load_official_variant_samples,
    )

    official_manifest = Path(str(contract["official_manifest"]))
    official_root = Path(str(contract["official_test_root"]))
    variant_samples = {
        variant: load_official_variant_samples(
            official_manifest, official_root, variant, verify_files=True
        )
        for variant in FORBIDDEN_SELECTION_VARIANTS
    }
    open_clip = _load_open_clip()
    model_results: Dict[str, Dict[str, object]] = {}

    for model_name in args.models:
        model_path = args.output_root / "models" / f"{_slug(model_name)}.json"
        if model_path.is_file() and not args.force:
            stored = json.loads(model_path.read_text())
            if stored.get("contract") != contract:
                raise RuntimeError(
                    f"Refusing to resume {model_name} with a changed contract: {model_path}"
                )
            expected = set(FORBIDDEN_SELECTION_VARIANTS)
            if set(stored.get("variant_results", {})) == expected:
                model_results[model_name] = stored
                print(f"[SKIP] Complete result already exists for {model_name}", flush=True)
                continue

        print(f"[MODEL] {model_name} weights=openai device={args.device}", flush=True)
        model, preprocess, tokenizer = _create_openclip_model(
            open_clip, model_name, args.device, args.download_root
        )
        model.eval()
        text_classifier = _build_text_classifier(
            model, tokenizer, CLASS_NAMES, args.device
        )
        variant_results: Dict[str, Dict[str, object]] = {}
        for variant in FORBIDDEN_SELECTION_VARIANTS:
            metrics = _evaluate_variant(
                model,
                preprocess,
                text_classifier,
                variant_samples[variant],
                args.batch_size,
                args.num_workers,
                args.device,
            )
            variant_results[variant] = metrics
            print(
                f"[RESULT] model={model_name} variant={variant} n={metrics['samples']} "
                f"acc={100.0 * float(metrics['accuracy']):.2f} "
                f"macro={100.0 * float(metrics['macro_class_accuracy']):.2f}",
                flush=True,
            )
        payload = {
            "contract": contract,
            "model": model_name,
            "weights": "openai",
            "variant_results": variant_results,
            "robustness_summary": summarize_model(variant_results),
        }
        _atomic_json(model_path, payload)
        model_results[model_name] = payload
        del model, text_classifier
        import torch

        torch.cuda.empty_cache()

    _write_tables(args.output_root, CLASS_NAMES, model_results)
    _atomic_json(
        args.output_root / "results.json",
        {"contract": contract, "models": model_results},
    )
    print(f"[DONE] {args.output_root / 'variant_results.csv'}", flush=True)
    print(f"[DONE] {args.output_root / 'per_class_results.csv'}", flush=True)
    print(f"[DONE] {args.output_root / 'robustness_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
